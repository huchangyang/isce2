#
# Author: Heresh Fattahi, Cunren Liang
#
#
import argparse
import logging
import os
import isce
import isceobj
from isceobj.Constants import SPEED_OF_LIGHT
import numpy as np
from osgeo import gdal
import shelve

from scipy import ndimage
try:
    import cv2
except ImportError:
    print('OpenCV2 does not appear to be installed / is not importable.')
    print('OpenCV2 is needed for this step. You may experience failures ...')


logger = logging.getLogger('isce.insar.runDispersive')

# Default γ threshold for sub-band coherence mask and polynomial phase weights (softer than ALOS ion_filt 0.97)
DEFAULT_IONO_COHERENCE_THRESHOLD = 0.5
DEFAULT_ADJUST_PHASE_COHERENCE_THRESHOLD = 0.7
DEFAULT_JUMP_GLOBAL_INTEGER = True


def resolve_coherence_path(cor_path):
    """
    Return path to an existing sub-band coherence product (.cor + .cor.xml).
    Config may point to filt_*.cor while FilterAndCoherence with filtStrength==0 wrote *.cor, or the reverse.
    cor_path: full path to the .cor file (no .xml suffix), as in stack configs / ISCE.
    """
    if not cor_path:
        return cor_path
    if os.path.exists(cor_path + '.xml'):
        return cor_path
    directory, fname = os.path.split(cor_path)
    if not fname.endswith('.cor'):
        return cor_path
    stem = fname[:-4]
    alts = []
    if stem.startswith('filt_'):
        alts.append(os.path.join(directory, stem[5:] + '.cor'))
    else:
        alts.append(os.path.join(directory, 'filt_' + stem + '.cor'))
    for c in alts:
        if os.path.exists(c + '.xml'):
            logger.info('Resolved coherence file: {} -> {}'.format(cor_path, c))
            return c
    return cor_path


def read_coherence_2d(cor_path, length, width):
    """
    Load sub-band coherence as (length, width) in [0, 1].

    Matches alosStack/ion_filt.py and adjust_phase_polynomial: supports
    single-band float32 (length * width) or BIL amp/coherence interleaved
    (length * 2, width; coherence on odd lines).

    Returns None if the file is missing or the float count does not match.
    """
    if not cor_path or not os.path.exists(str(cor_path) + '.xml'):
        return None
    data = np.fromfile(cor_path, dtype=np.float32)
    n_single = length * width
    n_bil = length * 2 * width
    if data.size == n_single:
        return np.clip(data.reshape(length, width), 0.0, 1.0).astype(np.float32)
    if data.size == n_bil:
        arr = data.reshape(length * 2, width)
        return np.clip(arr[1:length * 2:2, :], 0.0, 1.0).astype(np.float32)
    logger.warning(
        'read_coherence_2d: unexpected size for {}: got {} floats, expected {} or {}. Skipping.'.format(
            cor_path, data.size, n_single, n_bil))
    return None


def apply_alos_style_dual_band_invalid(ion, std, cor_low, cor_high):
    """
    Alos2Proc ion_filt (before std / adaptive Gaussian): if either sub-band
    coherence is zero, that pixel is invalid — std must be 0 so it gets no
    weight in adaptive_gaussian (wgt = 1/std^2 with wgt[index]=0).
    """
    if cor_low is None or cor_high is None:
        return
    if cor_low.shape != ion.shape or cor_high.shape != ion.shape:
        logger.warning('Coherence array shape mismatch; skipping dual-band invalid mask.')
        return
    invalid = (cor_low <= 1e-6) | (cor_high <= 1e-6)
    idx = np.nonzero(invalid)
    if idx[0].size:
        ion[idx] = 0
        std[idx] = 0


def polyfit_variance_weights_from_std_coherence(std, cor_low, cor_high, cor_threshold_fit):
    """
    ALOS runIonFilt global polynomial: wgt = std**2, then zero where averaged
    coherence (with dual-band zeros) is below corThresholdFit; caller inverts
    nonzeros to 1/wgt for polyfit_2d.
    """
    wgt = np.array(std, dtype=np.float64) ** 2
    wgt[np.nonzero(std == 0)] = 0.0
    if cor_low is not None and cor_high is not None:
        cor = (cor_low.astype(np.float64) + cor_high.astype(np.float64)) / 2.0
        cor[np.nonzero((cor_low <= 1e-6) | (cor_high <= 1e-6))] = 0.0
        cor = np.clip(cor, 0.0, 1.0)
        wgt[np.nonzero(cor < float(cor_threshold_fit))] = 0.0
    return wgt.astype(np.float32)


def resolve_ifg_prefix_for_unw(prefix, unw_method):
    """
    Return prefix such that prefix_unwmethod.unw exists (stripmap stack: snaphu -> *_snaphu.unw).
    Handles filt_ vs no-filt_ mismatch with FilterAndCoherence naming.
    """
    if not prefix or not unw_method:
        return prefix
    unw = prefix + '_' + unw_method + '.unw'
    if os.path.exists(unw + '.xml'):
        return prefix
    directory, base = os.path.split(prefix)
    alts = []
    if base.startswith('filt_'):
        alts.append(os.path.join(directory, base[5:]))
    else:
        alts.append(os.path.join(directory, 'filt_' + base))
    for p in alts:
        u = p + '_' + unw_method + '.unw'
        if os.path.exists(u + '.xml'):
            logger.info('Resolved interferogram prefix for unwrap: {} -> {}'.format(prefix, p))
            return p
    return prefix


def createParser():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='split the range spectrum of SLC')
    parser.add_argument('-L', '--low_band_igram_prefix', dest='lowBandIgramPrefix', type=str, required=True,
            help='prefix of unwrapped low band interferogram')
    parser.add_argument('-Lu', '--low_band_igram_unw_method', dest='lowBandIgramUnwMethod', type=str, required=True,
            help='unwrap method used for low band interferogram')
    parser.add_argument('-H', '--high_band_igram_prefix', dest='highBandIgramPrefix', type=str, required=True,
            help='prefix of unwrapped high band interferogram')
    parser.add_argument('-Hu', '--high_band_igram_unw_method', dest='highBandIgramUnwMethod', type=str, required=True,
            help='unwrap method used for high band interferogram')
    parser.add_argument('-o', '--outDir', dest='outDir', type=str, required=True,
            help='output directory')
    parser.add_argument('-a', '--low_band_shelve', dest='lowBandShelve', type=str, default=None,
            help='shelve file used to extract metadata')
    parser.add_argument('-b', '--high_band_shelve', dest='highBandShelve', type=str, default=None,
            help='shelve file used to extract metadata')
    parser.add_argument('-c', '--full_band_coherence', dest='fullBandCoherence', type=str, default=None,
            help='full band coherence')
    parser.add_argument('--low_band_coherence', dest='lowBandCoherence', type=str, default=None,
            help='low band coherence')
    parser.add_argument('--high_band_coherence', dest='highBandCoherence', type=str, default=None,
            help='high band coherence')
    parser.add_argument('--azimuth_looks', dest='azLooks', type=float, default=14.0,
            help='high band coherence')
    parser.add_argument('--range_looks', dest='rngLooks', type=float, default=4.0,
            help='high band coherence')

    parser.add_argument('--dispersive_filter_mask_type', dest='dispersive_filter_mask_type', type=str, default='coh_and_conncomp',
            help='mask for iono filtering: '
                 'coh_and_conncomp (default, coherence AND both conncomp > 0, recommended when unwrapping errors exist near boundaries); '
                 'coherence (both sub-bands must exceed coherence threshold); '
                 'connected_components (conncomp > 0 only); '
                 'unw (phase != 0 fallback).')

    parser.add_argument('--dispersive_filter_coherence_threshold', dest='dispersive_filter_coherence_threshold', type=float, default=DEFAULT_IONO_COHERENCE_THRESHOLD,
            help='sub-band coherence threshold for the ionosphere mask (default {:.2f}): '
            'both sub-bands must exceed it when mask_type=coherence'.format(DEFAULT_IONO_COHERENCE_THRESHOLD))
    parser.add_argument('--adjust_phase_coherence_threshold', dest='adjustPhaseCoherenceThreshold', type=float,
            default=DEFAULT_ADJUST_PHASE_COHERENCE_THRESHOLD,
            help='coherence threshold used only by polynomial phase adjustment weights (default {:.2f}, '
            'matches ALOS corThresholdAdj)'.format(DEFAULT_ADJUST_PHASE_COHERENCE_THRESHOLD))

    #parser.add_argument('-f', '--filter_sigma', dest='filterSigma', type=float, default=100.0,
    #        help='sigma of the gaussian filter')

    parser.add_argument('--filter_sigma_x', dest='kernel_sigma_x', type=float, default=100.0,
                help='sigma of the gaussian filter in X direction, default=100')

    parser.add_argument('--filter_sigma_y', dest='kernel_sigma_y', type=float, default=100.0,
                    help='sigma of the gaussian filter in Y direction, default=100')

    parser.add_argument('--filter_size_x', dest='kernel_x_size', type=float, default=800.0,
                            help='size of the gaussian kernel in X direction, default = 800')

    parser.add_argument('--filter_size_y', dest='kernel_y_size', type=float, default=800.0,
                        help='size of the gaussian kernel in Y direction, default=800')

    parser.add_argument('--filter_kernel_rotation', dest='kernel_rotation', type=float, default=0.0,
                        help='rotation angle of the filter kernel in degrees (default = 0.0)')

    parser.add_argument('-i', '--iteration', dest='dispersive_filter_iterations', type=int, default=5,
            help='number of iteration for filtering and interpolation')

    parser.add_argument('-m', '--mask_file', dest='maskFile', type=str, default=None,
            help='a mask file with one for valid pixels and zero for non valid pixels.')
    parser.add_argument('-u', '--outlier_sigma', dest='outlierSigma', type=float, default=1.0,
            help='number of sigma for removing outliers. data outside (avergae +/- u*sigma) are considered as outliers. sigma is calculated from data/coherence. u is the user input. default u =1')
    parser.add_argument('-p', '--min_pixel_connected_component', dest='minPixelConnComp', type=int, default=1000.0,
            help='minimum number of pixels in a connected component to consider the component as valid. components with less pixel will be masked out')
    parser.add_argument('-r', '--ref', dest='ref', type=str, default=None, help='refernce pixel : row, column')

    # Adaptive Gaussian filtering parameters (matching StripmapProc defaults)
    parser.add_argument('--filtering_winsize_max_ion', dest='filteringWinsizeMaxIon', type=int, default=501,
            help='maximum window size for adaptive Gaussian filtering (default=501)')
    parser.add_argument('--filtering_winsize_min_ion', dest='filteringWinsizeMinIon', type=int, default=51,
            help='minimum window size for adaptive Gaussian filtering (default=51)')
    parser.add_argument('--filtering_winsize_secondary_ion', dest='filteringWinsizeSecondaryIon', type=int, default=5,
            help='window size for secondary Gaussian filtering (default=5)')
    parser.add_argument('--filter_std_ion', dest='filterStdIon', type=float, default=None,
            help='target standard deviation for adaptive filtering (default=None, auto-determined)')
    parser.add_argument('--fit_adaptive_ion', dest='fitAdaptiveIon', type=bool, default=True,
            help='apply polynomial fit in adaptive filtering window (default=True)')
    parser.add_argument('--filt_secondary_ion', dest='filtSecondaryIon', type=bool, default=True,
            help='apply secondary filtering after adaptive filtering (default=True)')
    parser.add_argument('--use_adaptive_gaussian', dest='useAdaptiveGaussian', type=bool, default=True,
            help='use adaptive Gaussian filtering instead of iterative filtering (default=True)')
    parser.add_argument('--adjust_phase_polynomial', dest='adjustPhasePolynomial', type=bool, default=True,
            help='adjust phase using polynomial fitting before computing ionosphere (ALOS-style, default=True)')
    parser.add_argument('--fit_ion', dest='fitIon', type=bool, default=True,
            help='apply global polynomial fit to ionosphere before filtering (ALOS-style, default=True)')
    parser.add_argument('--filt_ion', dest='filtIon', type=bool, default=True,
            help='apply adaptive Gaussian filtering to ionosphere (ALOS-style, default=True)')
    parser.add_argument('--fit_ion_coherence_threshold', dest='fitIonCoherenceThreshold', type=float, default=0.25,
            help='coherence threshold for global polynomial fitting (default=0.25)')
    parser.add_argument('--jump_global_integer', dest='jumpGlobalInteger', type=bool,
            default=DEFAULT_JUMP_GLOBAL_INTEGER,
            help='force the entire scene to use a single global integer jump (rounded global median). '
                 'Prevents spatial discontinuities in jumps.bil caused by large-scale phase ramps '
                 'spanning integer boundaries, which would cause dense-fringe artefacts in ion. '
                 'Residual offset is absorbed by unwrapp_error_correction downstream (default=True). '
                 'Set False only when different disconnected conncomp regions genuinely need '
                 'different integer corrections.')

    # Separate controls for non-dispersive component (optional / often unnecessary)
    parser.add_argument('--fit_nonDispersive', dest='fitNonDispersive', type=bool, default=False,
            help='apply global polynomial fit to non-dispersive phase (ALOS-style, default=False)')
    parser.add_argument('--filt_nonDispersive', dest='filtNonDispersive', type=bool, default=False,
            help='apply adaptive Gaussian filtering to non-dispersive phase (ALOS-style, default=False)')
    
    # Ionospheric looks parameters (for multilooked interferograms)
    parser.add_argument('--number_range_looks_ion', dest='numberRangeLooksIon', type=int, default=16,
            help='number of range looks for ionospheric estimation (default=16)')
    parser.add_argument('--number_azimuth_looks_ion', dest='numberAzimuthLooksIon', type=int, default=16,
            help='number of azimuth looks for ionospheric estimation (default=16)')
    
    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def getValue(dataFile, band, y_ref, x_ref):
    ds = gdal.Open(dataFile, gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    b = ds.GetRasterBand(band)
    ref = b.ReadAsArray(x_ref,y_ref,1,1)
    
    ds = None
    return ref[0][0]

def polyfit_2d(data, weight, order):
    '''
    Fit a surface to a 2-d matrix (from Alos2Proc)
    '''
    if order < 1:
        raise Exception('order must >= 1!')
    
    if data.shape != weight.shape:
        raise Exception('data and weight must be of same size!')
    
    (length, width) = data.shape
    n = data.size
    
    # Number of coefficients
    ncoeff = 1
    for i in range(1, order+1):
        for j in range(i+1):
            ncoeff += 1
    
    # Row, column
    y, x = np.indices((length, width))
    x = x.flatten()
    y = y.flatten()
    z = data.flatten()
    weight = np.sqrt(weight.flatten())
    
    # Linear functions: H theta = s
    H = np.zeros((n, ncoeff))
    H[:,0] += 1
    k = 1
    for i in range(1, order+1):
        for j in range(i+1):
            H[:, k] = x**(i-j)*y**(j)
            k += 1
    
    # Least squares
    coeff = np.linalg.lstsq(H*weight[:,None], z*weight, rcond=-1)[0]
    
    # Fit surface
    data_fit = (np.dot(H, coeff)).reshape(length, width)
    
    return (data_fit, coeff)

def adaptive_gaussian(data, std, size_min, size_max, std_out0, fit=True):
    '''
    Adaptive Gaussian filtering (from Alos2Proc)
    This program performs Gaussian filtering with adaptive window size.
    
    data:     input raw data, numpy array
    std:      standard deviation of raw data, numpy array
    size_min: minimum filter window size
    size_max: maximum filter window size (size_min <= size_max, size_min == size_max is allowed)
    std_out0: standard deviation of output data
    fit:      whether do fitting before gaussian filtering
    '''
    import scipy.signal as ss
    
    (length, width) = data.shape
    
    # Assume zero-value samples are invalid
    index = np.nonzero(np.logical_or(data==0, std==0))
    data[index] = 0
    std[index] = 0
    # Compute weight using standard deviation
    wgt = 1.0 / (std**2 + (std==0))
    wgt[index] = 0
    
    # Compute number of gaussian filters
    if size_min > size_max:
        raise Exception('size_min: {} > size_max: {}'.format(size_min, size_max))
    
    if size_min % 2 == 0:
        size_min += 1
    if size_max % 2 == 0:
        size_max += 1
    
    size_num = int((size_max - size_min) / 2 + 1)
    
    # Create gaussian filters
    print('compute Gaussian filters')
    gaussian_filters = []
    for i in range(size_num):
        size = int(size_min + i * 2)
        # Gaussian kernel
        hsize = (size - 1) / 2
        x = np.arange(-hsize, hsize + 1)
        f = np.exp(-x**2/(2.0*(size/2.0)**2)) / ((size/2.0) * np.sqrt(2.0*np.pi))
        # Use np.outer for 2D Gaussian kernel
        f2d = np.outer(f, f)
        gaussian_filters.append(f2d/np.sum(f2d))
    
    # Compute standard deviation after filtering
    print('compute standard deviation after filtering for each filtering window size')
    std_filt = np.zeros((length, width, size_num))
    for i in range(size_num):
        size = int(size_min + i * 2)
        print('current window size: %4d, min window size: %4d, max window size: %4d' % (size, size_min, size_max), end='\r', flush=True)
        index = np.nonzero(ss.fftconvolve(wgt!=0, gaussian_filters[i]!=0, mode='same') < 0.5)
        scale = ss.fftconvolve(wgt, gaussian_filters[i], mode='same')
        scale[index] = 0
        var_filt = ss.fftconvolve(wgt, gaussian_filters[i]**2, mode='same') / (scale**2 + (scale==0))
        var_filt[index] = 0
        std_filt[:, :, i] = np.sqrt(var_filt)
    print('\n')
    
    # Find gaussian window size
    print('find Gaussian window size to use')
    gaussian_index = np.zeros((length, width), dtype=np.int32)
    std_filt2 = np.zeros((length, width))
    for i in range(length):
        if (((i+1)%50) == 0):
            print('processing line %6d of %6d' % (i+1, length), end='\r', flush=True)
        for j in range(width):
            if np.sum(std_filt[i, j, :]) == 0:
                gaussian_index[i, j] = -1
            else:
                gaussian_index[i, j] = size_num - 1
                for k in range(size_num):
                    if (std_filt[i, j, k] != 0) and (std_filt[i, j, k] <= std_out0):
                        gaussian_index[i, j] = k
                        break
            if gaussian_index[i, j] != -1:
                std_filt2[i, j] = std_filt[i, j, gaussian_index[i, j]]
    del std_filt
    print("processing line %6d of %6d\n" % (length, length))
    
    # Adaptive gaussian filtering
    print('filter image')
    data_out = np.zeros((length, width))
    std_out = np.zeros((length, width))
    window_size_out = np.zeros((length, width), dtype=np.int16)
    # Reduce print frequency for better performance
    print_interval = max(100, length // 20)  # Print at most 20 times
    for i in range(length):
        # Print progress less frequently to reduce I/O overhead
        if (((i+1) % print_interval == 0) or (i == 0) or (i == length-1)):
            progress_pct = 100.0 * (i+1) / length
            print('processing line %6d of %6d (%.1f%%)' % (i+1, length, progress_pct), end='\r', flush=True)
        for j in range(width):
            if gaussian_index[i, j] == -1:
                continue
            
            size = int(size_min + gaussian_index[i, j] * 2)
            size_half = int((size - 1) / 2)
            window_size_out[i, j] = size
            
            first_line = max(i-size_half, 0)
            last_line = min(i+size_half, length-1)
            first_column = max(j-size_half, 0)
            last_column = min(j+size_half, width-1)
            length_valid = last_line - first_line + 1
            width_valid = last_column - first_column + 1
            
            if first_line == 0:
                last_line2 = size - 1
                first_line2 = last_line2 - (length_valid - 1)
            else:
                first_line2 = 0
                last_line2 = first_line2 + (length_valid - 1)
            if first_column == 0:
                last_column2 = size - 1
                first_column2 = last_column2 - (width_valid - 1)
            else:
                first_column2 = 0
                last_column2 = first_column2 + (width_valid - 1)
            
            data_window = np.zeros((size, size))
            wgt_window = np.zeros((size, size))
            data_window[first_line2:last_line2+1, first_column2:last_column2+1] = data[first_line:last_line+1, first_column:last_column+1]
            wgt_window[first_line2:last_line2+1, first_column2:last_column2+1] = wgt[first_line:last_line+1, first_column:last_column+1]
            n_valid = np.sum(data_window!=0)
            
            order, n_coeff = (2, 6)
            if fit:
                if n_valid > n_coeff * 3:
                    data_fit, coeff = polyfit_2d(data_window, wgt_window, order)
                    index = np.nonzero(data_window!=0)
                    data_window[index] -= data_fit[index]
            
            wgt_window_2 = wgt_window * gaussian_filters[gaussian_index[i, j]]
            scale = 1.0/np.sum(wgt_window_2)
            wgt_window_2 *= scale
            data_out[i, j] = np.sum(wgt_window_2 * data_window)
            std_out[i, j] = std_filt2[i, j]
            
            if fit:
                if n_valid > n_coeff * 3:
                    data_out[i, j] += data_fit[size_half, size_half]
    print('\n')
    
    return (data_out, std_out, window_size_out)

def adjust_phase_polynomial(lowBandIgram, highBandIgram, outputDir, lowBandCoherence=None, highBandCoherence=None,
                            coherence_weight_threshold=DEFAULT_IONO_COHERENCE_THRESHOLD):
    '''
    Adjust phase using polynomial fitting (similar to ALOS processing)
    This function adjusts the upper band phase to remove relative phase unwrapping errors
    using polynomial fitting, similar to computeIonosphere in runIonFilt.py.
    When coherence files are available, pixels where either sub-band coherence is below
    coherence_weight_threshold get zero weight in the polynomial fit (ALOS uses a high
    threshold on diff coherence before computeIonosphere; here we use both sub-bands).
    
    Returns: adjusted high band interferogram file path
    '''
    logger.info('Adjusting phase using polynomial fitting (ALOS-style)')
    
    # Read unwrapped interferograms
    img_low = isceobj.createImage()
    img_low.load(lowBandIgram + '.xml')
    width = img_low.width
    length = img_low.length

    def _read_unw_phase(unw_path):
        """
        Read unwrapped interferogram phase robustly.

        Supports two common layouts:
        - single-band float32 phase:            size = length * width
        - BIL amp/phase interleaved (phase in odd lines): size = length * 2 * width
        """
        data = np.fromfile(unw_path, dtype=np.float32)
        n_single = length * width
        n_bil = length * 2 * width
        if data.size == n_single:
            return data.reshape(length, width), 'single'
        if data.size == n_bil:
            arr = data.reshape(length * 2, width)
            return arr[1:length * 2:2, :], 'bil'
        raise ValueError(
            f'Unexpected unw binary size for {unw_path}: got {data.size} float32, '
            f'expected {n_single} (single) or {n_bil} (BIL interleaved).'
        )

    def _read_coherence_for_weights(cor_path):
        """
        Read coherence for weighting as an array of shape (length, width).
        Coherence files are typically single-band float32 with size = length*width.
        Some pipelines may store two-band BIL interleaved arrays; support that too.
        """
        data = np.fromfile(cor_path, dtype=np.float32)
        n_single = length * width
        n_bil = length * 2 * width
        if data.size == n_single:
            cor = data.reshape(length, width)
        elif data.size == n_bil:
            arr = data.reshape(length * 2, width)
            cor = arr[1:length * 2:2, :]
        else:
            raise ValueError(
                f'Unexpected coherence binary size for {cor_path}: got {data.size} float32, '
                f'expected {n_single} (single) or {n_bil} (BIL interleaved).'
            )
        return np.clip(cor, 0.0, 1.0)

    lowerUnw, lowFmt = _read_unw_phase(lowBandIgram)
    upperUnw, highFmt = _read_unw_phase(highBandIgram)
    if lowFmt != highFmt:
        raise ValueError(
            f'Inconsistent unw layouts between low/high: {lowFmt} vs {highFmt}. '
            f'Please verify low/high band .unw binary formats.'
        )
    
    # Prepare weight using coherence if available
    if lowBandCoherence and highBandCoherence and os.path.exists(lowBandCoherence + '.xml') and os.path.exists(highBandCoherence + '.xml'):
        cor_low = _read_coherence_for_weights(lowBandCoherence)
        cor_high = _read_coherence_for_weights(highBandCoherence)
        # Use average coherence as weight, with high power (similar to ALOS corOrderAdj=20)
        cor = (cor_low + cor_high) / 2.0
        wgt = cor**20  # Similar to corOrderAdj=20 in ALOS
        # ALOS-style gating: exclude pixels where either sub-band is below threshold (matches coherence mask logic)
        if coherence_weight_threshold is not None:
            low_ok = cor_low > float(coherence_weight_threshold)
            high_ok = cor_high > float(coherence_weight_threshold)
            wgt[np.nonzero(~(low_ok & high_ok))] = 0.0
        wgt[np.nonzero(lowerUnw==0)] = 0
        wgt[np.nonzero(upperUnw==0)] = 0
    else:
        # Use binary mask if coherence not available
        wgt = np.ones((length, width), dtype=np.float32)
        wgt[np.nonzero(lowerUnw==0)] = 0
        wgt[np.nonzero(upperUnw==0)] = 0
    
    # Compute phase difference
    phase_diff = lowerUnw - upperUnw
    
    # Fit polynomial surface to phase difference (order 2, similar to ALOS)
    diff_fit, coeff = polyfit_2d(phase_diff, wgt, 2)
    
    # Adjust upper band phase
    flag2 = (lowerUnw != 0)
    index2 = np.nonzero(flag2)
    
    # Phase for adjustment: round the difference to nearest 2π
    unwd = ((phase_diff - diff_fit)[index2]) / (2.0*np.pi)
    unw_adj = np.around(unwd) * (2.0*np.pi)
    
    # Adjust upper band phase
    upperUnw_adjusted = upperUnw.copy()
    upperUnw_adjusted[index2] += unw_adj
    
    # Check adjustment results
    unw_diff_adj = (lowerUnw - upperUnw_adjusted)[index2]
    logger.info('After polynomial adjustment:')
    logger.info('  Max phase difference: {:.4f}'.format(np.amax(unw_diff_adj)))
    logger.info('  Min phase difference: {:.4f}'.format(np.amin(unw_diff_adj)))
    logger.info('  Max-min: {:.4f}'.format(np.amax(unw_diff_adj) - np.amin(unw_diff_adj)))
    
    # Save adjusted high band interferogram
    highBandIgramAdjusted = os.path.join(outputDir, os.path.basename(highBandIgram) + '.adjusted')

    # Save adjusted file with matching layout + XML
    if highFmt == 'bil':
        # In this codebase, BIL "amp/phase interleaved" is stored as (length*2, width)
        # floats on disk, while XML usually keeps length==original_length and uses bands==2.
        original_data = np.fromfile(highBandIgram, dtype=np.float32).reshape(length * 2, width)
        original_data[1:length * 2:2, :] = upperUnw_adjusted
        original_data.astype(np.float32).tofile(highBandIgramAdjusted)
        # Important: do NOT double the XML length here.
        write_xml(highBandIgramAdjusted, width, length, 2, "FLOAT", "BIL")
    else:
        # Single-band phase only: disk size matches length*width floats.
        upperUnw_adjusted.astype(np.float32).tofile(highBandIgramAdjusted)
        write_xml(highBandIgramAdjusted, width, length, 1, "FLOAT", "BIL")
    
    logger.info('Adjusted high band interferogram saved to: {}'.format(highBandIgramAdjusted))
    
    return highBandIgramAdjusted


def check_consistency(lowBandIgram, highBandIgram, outputDir,
                      global_integer=DEFAULT_JUMP_GLOBAL_INTEGER):
    """
    Estimate the relative 2π integer jumps between low- and high-band sub-band
    interferograms.

    When global_integer=True (default), the entire scene is forced to use a
    single integer value (the rounded global median of the per-pixel estimate).
    This prevents spatial discontinuities in jumps.bil that arise when the
    (low−high)/(2π) field has a large-scale gradient crossing an integer
    boundary – the most common cause of dense-fringe artefacts in the ion
    output.  Any residual constant offset introduced by the global rounding is
    handled by the subsequent unwrapp_error_correction() step.

    When global_integer=False, per-pixel rounding is used (original behaviour),
    which is appropriate when different disconnected connected-component regions
    genuinely require different integer corrections.
    """
    jumpFile = os.path.join(outputDir, "jumps.bil")

    if not global_integer:
        cmd = 'imageMath.py -e="round((a_1-b_1)/(2.0*PI))" --a={0} --b={1} -o {2} -t float -s BIL'.format(
            lowBandIgram, highBandIgram, jumpFile)
        print(cmd)
        os.system(cmd)
        return jumpFile

    # ---- global-integer mode ----
    img = isceobj.createImage()
    img.load(lowBandIgram + '.xml')
    length = img.length
    width = img.width

    def _read_phase(path):
        data = np.fromfile(path, dtype=np.float32)
        n_single = length * width
        n_bil = length * 2 * width
        if data.size == n_single:
            return data.reshape(length, width)
        if data.size == n_bil:
            arr = data.reshape(length * 2, width)
            return arr[1:length * 2:2, :]
        raise ValueError(
            'Unexpected binary size for {}: {} floats (expected {} or {})'.format(
                path, data.size, n_single, n_bil))

    lowerUnw = _read_phase(lowBandIgram)
    upperUnw = _read_phase(highBandIgram)

    valid = (lowerUnw != 0) & (upperUnw != 0)
    diff_cycles = (lowerUnw - upperUnw) / (2.0 * np.pi)

    global_median = float(np.median(diff_cycles[valid]))
    global_jump = int(np.round(global_median))

    per_pixel = np.round(diff_cycles[valid]).astype(np.int32)
    disagree_frac = float(np.mean(per_pixel != global_jump))
    logger.info(
        'check_consistency (global_integer=True): global jump = {} '
        '(median = {:.4f}), per-pixel disagreement = {:.1f}%'.format(
            global_jump, global_median, 100.0 * disagree_frac))
    if disagree_frac > 0.15:
        logger.warning(
            '{:.1f}% of valid pixels differ from global jump {}. '
            'A large-scale phase ramp exists between sub-bands. '
            'The constant offset will be corrected by unwrapp_error_correction(). '
            'Set --jump_global_integer=False to revert to per-pixel mode.'.format(
                100.0 * disagree_frac, global_jump))

    jumps = np.where(valid, float(global_jump), 0.0).astype(np.float32)
    jumps.tofile(jumpFile)
    write_xml(jumpFile, width, length, 1, "FLOAT", "BIL")

    return jumpFile



def dispersive_nonDispersive(lowBandIgram, highBandIgram, f0, fL, fH, outDispersive, outNonDispersive, jumpFile, y_ref=None, x_ref=None, m=None , d=None):
    
    if y_ref and x_ref:
        refL = getValue(lowBandIgram, 2, y_ref, x_ref)
        refH = getValue(highBandIgram, 2, y_ref, x_ref)

    else:
        refL = 0.0
        refH = 0.0
    
    # m : common phase unwrapping error
    # d : differential phase unwrapping error

    if m and d:

        coef = (fL*fH)/(f0*(fH**2 - fL**2))
        #cmd = 'imageMath.py -e="{0}*((a_1-{8}-2*PI*c)*{1}-(b_1-{9}-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} -o {7} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, m , d, outDispersive, refL, refH)
        cmd = 'imageMath.py -e="{0}*((a_1-2*PI*c)*{1}-(b_1+(2.0*PI*g)-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} --g={7} -o {8} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, m , d, jumpFile, outDispersive)
        print(cmd)
        os.system(cmd)

        coefn = f0/(fH**2-fL**2)
        #cmd = 'imageMath.py -e="{0}*((a_1-{8}-2*PI*c)*{1}-(b_1-{9}-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} -o {7} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, m , d, outNonDispersive, refH, refL)
        cmd = 'imageMath.py -e="{0}*((a_1+(2.0*PI*g)-2*PI*c)*{1}-(b_1-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} --g={7} -o {8} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, m , d, jumpFile, outNonDispersive)
        print(cmd)
        os.system(cmd)

    else:
        
        coef = (fL*fH)/(f0*(fH**2 - fL**2))
        #cmd = 'imageMath.py -e="{0}*((a_1-{6})*{1}-(b_1-{7})*{2})" --a={3} --b={4} -o {5} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, outDispersive, refL, refH)
        cmd = 'imageMath.py -e="{0}*(a_1*{1}-(b_1+2.0*PI*c)*{2})" --a={3} --b={4} --c={5}  -o {6} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, jumpFile, outDispersive)

        print(cmd)
        os.system(cmd)

        coefn = f0/(fH**2-fL**2)
        #cmd = 'imageMath.py -e="{0}*((a_1-{6})*{1}-(b_1-{7})*{2})" --a={3} --b={4} -o {5} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, outNonDispersive, refH, refL) 
        cmd = 'imageMath.py -e="{0}*((a_1+2.0*PI*c)*{1}-(b_1)*{2})" --a={3} --b={4} --c={5} -o {6} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, jumpFile, outNonDispersive)
        print(cmd)
        os.system(cmd)


    return None

def theoretical_variance_fromSubBands(inps, f0, fL, fH, B, Sig_phi_iono, Sig_phi_nonDisp,N):
    # Calculating the theoretical variance of the 
    # ionospheric phase based on the coherence of
    # the sub-band interferograns 
    #ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
    lowBandCoherence = inps.lowBandCoherence 
    Sig_phi_L = inps.Sig_phi_L 

    #ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)
    #highBandIgram = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename + ".unw")

    #ifgDirname = os.path.dirname(self.insar.lowBandIgram)
    #lowBandCoherence = os.path.join(ifgDirname , self.insar.coherenceFilename)
    #Sig_phi_L = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename + ".sig")

    #ifgDirname = os.path.dirname(self.insar.highBandIgram)
    #highBandCoherence = os.path.join(ifgDirname , self.insar.coherenceFilename)
    #Sig_phi_H = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename + ".sig")

    highBandCoherence = inps.highBandCoherence
    Sig_phi_H = inps.Sig_phi_H

    #N = self.numberAzimuthLooks*self.numberRangeLooks
    #PI = np.pi
    #fL,f0,fH,B = getBandFrequencies(inps)
    #cL = read(inps.lowBandCoherence,bands=[1])
    #cL = cL[0,:,:]
    #cL[cL==0.0]=0.001
    
    # Guard against coherence == 0 to avoid inf/NaN in sigma maps.
    # Important: if coherence is 0, the corresponding sigma should be 0 (invalid),
    # not a huge finite value caused by the epsilon floor.
    cmd = 'imageMath.py -e="(a>1.0e-6)*sqrt(1.0-(a+(a<=1.0e-6)*1.0e-6)**2)/(a+(a<=1.0e-6)*1.0e-6)/sqrt(2.0*{0})" --a={1} -o {2} -t float -s BIL'.format(N, lowBandCoherence, Sig_phi_L)
    print(cmd)
    os.system(cmd)
    #Sig_phi_L = np.sqrt(1-cL**2)/cL/np.sqrt(2.*N)

    #cH = read(inps.highBandCoherence,bands=[1])
    #cH = cH[0,:,:]
    #cH[cH==0.0]=0.001

    cmd = 'imageMath.py -e="(a>1.0e-6)*sqrt(1.0-(a+(a<=1.0e-6)*1.0e-6)**2)/(a+(a<=1.0e-6)*1.0e-6)/sqrt(2.0*{0})" --a={1} -o {2} -t float -s BIL'.format(N, highBandCoherence, Sig_phi_H)
    print(cmd)
    os.system(cmd)
    #Sig_phi_H = np.sqrt(1-cH**2)/cH/np.sqrt(2.0*N)

    coef = (fL*fH)/(f0*(fH**2 - fL**2))

    cmd = 'imageMath.py -e="sqrt(({0}**2)*({1}**2)*(a**2) + ({0}**2)*({2}**2)*(b**2))" --a={3} --b={4} -o {5} -t float -s BIL'.format(coef, fL, fH, Sig_phi_L, Sig_phi_H, Sig_phi_iono)
    os.system(cmd)

    #Sig_phi_iono = np.sqrt((coef**2)*(fH**2)*Sig_phi_H**2 + (coef**2)*(fL**2)*Sig_phi_L**2)
    #length, width = Sig_phi_iono.shape

    #outFileIono = os.path.join(inps.outDir, 'Sig_iono.bil')
    #write(Sig_phi_iono, outFileIono, 1, 6)
    #write_xml(outFileIono, length, width)

    coef_non = f0/(fH**2 - fL**2)
    cmd = 'imageMath.py -e="sqrt(({0}**2)*({1}**2)*(a**2) + ({0}**2)*({2}**2)*(b**2))" --a={3} --b={4} -o {5} -t float -s BIL'.format(coef_non, fL, fH, Sig_phi_L, Sig_phi_H, Sig_phi_nonDisp)
    os.system(cmd)

    #Sig_phi_non_dis = np.sqrt((coef_non**2) * (fH**2) * Sig_phi_H**2 + (coef_non**2) * (fL**2) * Sig_phi_L**2)

    #outFileNonDis = os.path.join(inps.outDir, 'Sig_nonDis.bil')
    #write(Sig_phi_non_dis, outFileNonDis, 1, 6)
    #write_xml(outFileNonDis, length, width)

    return None #Sig_phi_iono, Sig_phi_nonDisp

def lowPassFilter(dataFile, sigDataFile, maskFile, Sx, Sy, sig_x, sig_y, iteration=5, theta=0.0):
    ds = gdal.Open(dataFile + '.vrt', gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    dataIn = np.memmap(dataFile, dtype=np.float32, mode='r', shape=(length,width))
    sigData = np.memmap(sigDataFile, dtype=np.float32, mode='r', shape=(length,width))
    mask = np.memmap(maskFile, dtype=np.byte, mode='r', shape=(length,width))

    dataF, sig_dataF = iterativeFilter(dataIn[:,:], mask[:,:], sigData[:,:], iteration, Sx, Sy, sig_x, sig_y, theta)

    filtDataFile = dataFile + ".filt"
    sigFiltDataFile  = sigDataFile + ".filt"
    filtData = np.memmap(filtDataFile, dtype=np.float32, mode='w+', shape=(length,width))
    filtData[:,:] = dataF[:,:]
    filtData.flush()

    sigFilt= np.memmap(sigFiltDataFile, dtype=np.float32, mode='w+', shape=(length,width))
    sigFilt[:,:] = sig_dataF[:,:]
    sigFilt.flush()

    # writing xml and vrt files
    write_xml(filtDataFile, width, length, 1, "FLOAT", "BIL")
    write_xml(sigFiltDataFile, width, length, 1, "FLOAT", "BIL")

    return filtDataFile, sigFiltDataFile

def write_xml(fileName,width,length,bands,dataType,scheme):

    img = isceobj.createImage()
    img.setFilename(fileName)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = bands
    img.dataType = dataType
    img.scheme = scheme
    img.renderHdr()
    img.renderVRT()
    
    return None

def iterativeFilter(dataIn, mask, Sig_dataIn, iteration, Sx, Sy, sig_x, sig_y, theta=0.0):
    data = np.zeros(dataIn.shape)
    data[:,:] = dataIn[:,:]
    Sig_data = np.zeros(dataIn.shape)
    Sig_data[:,:] = Sig_dataIn[:,:]

    print ('masking the data')
    data[mask==0]=np.nan
    Sig_data[mask==0]=np.nan
    print ('Filling the holes with nearest neighbor interpolation')
    dataF = fill(data)
    Sig_data = fill(Sig_data)
    print ('Low pass Gaussian filtering the interpolated data')
    dataF, Sig_dataF = Filter(dataF, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0)
    for i in range(iteration):
       print ('iteration: ', i , ' of ',iteration)
       print ('masking the interpolated and filtered data')
       dataF[mask==0]=np.nan
       print('Filling the holes with nearest neighbor interpolation of the filtered data from previous step')
       dataF = fill(dataF)
       print('Replace the valid pixels with original unfiltered data')
       dataF[mask==1]=data[mask==1]
       dataF, Sig_dataF = Filter(dataF, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0)

    return dataF, Sig_dataF

def Filter(data, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0):
    kernel = Gaussian_kernel(Sx, Sy, sig_x, sig_y) #(800, 800, 15.0, 100.0)
    kernel = rotate(kernel , theta)

    data = data/Sig_data**2
    data = cv2.filter2D(data,-1,kernel)
    W1 = cv2.filter2D(1.0/Sig_data**2,-1,kernel)
    W2 = cv2.filter2D(1.0/Sig_data**2,-1,kernel**2)

    #data = ndimage.convolve(data,kernel, mode='nearest')
    #W1 = ndimage.convolve(1.0/Sig_data**2,kernel, mode='nearest')
    #W2 = ndimage.convolve(1.0/Sig_data**2,kernel**2, mode='nearest')


    return data/W1, np.sqrt(W2/(W1**2))

def Gaussian_kernel(Sx, Sy, sig_x,sig_y):
    if np.mod(Sx,2) == 0:
        Sx = Sx + 1

    if np.mod(Sy,2) ==0:
            Sy = Sy + 1

    x,y = np.meshgrid(np.arange(Sx),np.arange(Sy))
    x = x + 1
    y = y + 1
    x0 = (Sx+1)/2
    y0 = (Sy+1)/2
    fx = ((x-x0)**2.)/(2.*sig_x**2.)
    fy = ((y-y0)**2.)/(2.*sig_y**2.)
    k = np.exp(-1.0*(fx+fy))
    a = 1./np.sum(k)
    k = a*k
    return k

def rotate(k , theta):

    Sy,Sx = np.shape(k)
    x,y = np.meshgrid(np.arange(Sx),np.arange(Sy))

    x = x + 1
    y = y + 1
    x0 = (Sx+1)/2
    y0 = (Sy+1)/2
    x = x - x0
    y = y - y0

    A=np.vstack((x.flatten(), y.flatten()))
    if theta!=0:
        from scipy.interpolate import griddata
        theta = theta*np.pi/180.
        R = np.array([[np.cos(theta), -1.0*np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        AR = np.dot(R,A)
        xR = AR[0,:].reshape(Sy,Sx)
        yR = AR[1,:].reshape(Sy,Sx)

        k = griddata((x.flatten(),y.flatten()),k.flatten(),(xR,yR), method='linear')
        #k = f(xR, yR)
        #k = k.data
        k[np.isnan(k)] = 0.0
        a = 1./np.sum(k)
        k = a*k
    return k

def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell
    
    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'.
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)
       
    Output:
        Return a filled array.
    """
    if invalid is None: invalid = np.isnan(data)

    ind = ndimage.distance_transform_edt(invalid,
                                    return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]


def getMask(inps, maskFile, lowBandIgram=None, highBandIgram=None):
    '''
    Generate mask file for filtering, with support for water body masking
    '''
    if lowBandIgram is None:
        lowBandIgram = inps.lowBandIgram 
    if highBandIgram is None:
        highBandIgram = inps.highBandIgram
    
    lowBandCor = inps.lowBandCoherence
    highBandCor = inps.highBandCoherence
    th = getattr(inps, 'dispersive_filter_coherence_threshold', DEFAULT_IONO_COHERENCE_THRESHOLD)

    mask_type = inps.dispersive_filter_mask_type

    # ---- resolve availability flags ----
    has_coherence = bool(lowBandCor and highBandCor
                         and os.path.exists(str(lowBandCor) + '.xml')
                         and os.path.exists(str(highBandCor) + '.xml'))
    has_conncomp = (os.path.exists(lowBandIgram + '.conncomp')
                    and os.path.exists(highBandIgram + '.conncomp'))

    # ---- auto-degrade when requested files are not available ----
    if mask_type == 'coherence' and not has_coherence:
        logger.warning(
            'dispersive_filter_mask_type=coherence but sub-band coherence files are missing; '
            'falling back to mask from unwrapped phases (phase != 0).')
        mask_type = 'unw_fallback'

    if mask_type == 'coh_and_conncomp':
        if not has_coherence and not has_conncomp:
            logger.warning(
                'dispersive_filter_mask_type=coh_and_conncomp: neither coherence nor conncomp files '
                'found; falling back to unw mask.')
            mask_type = 'unw_fallback'
        elif not has_coherence:
            logger.warning(
                'dispersive_filter_mask_type=coh_and_conncomp: coherence files missing; '
                'using conncomp only.')
            mask_type = 'connected_components'
        elif not has_conncomp:
            logger.warning(
                'dispersive_filter_mask_type=coh_and_conncomp: conncomp files missing; '
                'using coherence only.')
            mask_type = 'coherence'

    if mask_type == 'connected_components' and not has_conncomp:
        logger.warning(
            'dispersive_filter_mask_type=connected_components but conncomp files not found; '
            'falling back to coherence mask.' if has_coherence else
            'dispersive_filter_mask_type=connected_components but conncomp files not found; '
            'falling back to unw mask.')
        mask_type = 'coherence' if has_coherence else 'unw_fallback'

    # ---- generate the mask ----
    if mask_type == 'coh_and_conncomp':
        # Step 1: coherence-based mask
        print('generating mask: coherence AND conncomp (threshold={})'.format(th))
        coh_maskFile = maskFile + '.coh_tmp'
        cmd = 'imageMath.py -e="(a>{0})*(b>{0})" --a={1} --b={2} -t byte -s BIL -o {3}'.format(
            th, lowBandCor, highBandCor, coh_maskFile)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError('Failed to generate coherence mask. Command: {}'.format(cmd))

        # Step 2: load coherence mask and intersect with conncomp > 0
        img_tmp = isceobj.createImage()
        img_tmp.load(coh_maskFile + '.xml')
        _w = img_tmp.width
        _l = img_tmp.length
        coh_mask = np.fromfile(coh_maskFile, dtype=np.byte).reshape(_l, _w)

        # conncomp files are typically uint8; any value > 0 means "unwrapped and valid"
        conncomp_low = np.fromfile(lowBandIgram + '.conncomp', dtype=np.uint8).reshape(_l, _w)
        conncomp_high = np.fromfile(highBandIgram + '.conncomp', dtype=np.uint8).reshape(_l, _w)
        combined = ((coh_mask != 0) & (conncomp_low > 0) & (conncomp_high > 0)).astype(np.byte)

        coh_valid = int(np.sum(coh_mask != 0))
        conncomp_valid = int(np.sum((conncomp_low > 0) & (conncomp_high > 0)))
        combined_valid = int(np.sum(combined))
        logger.info(
            'coh_and_conncomp mask: coherence-valid={}, conncomp-valid={}, combined={} ({:.1f}% of coh)'.format(
                coh_valid, conncomp_valid, combined_valid,
                100.0 * combined_valid / coh_valid if coh_valid > 0 else 0.0))
        combined.tofile(maskFile)
        write_xml(maskFile, _w, _l, 1, 'BYTE', 'BIL')

        # clean up temp file
        for _ext in ('', '.xml', '.vrt'):
            _f = coh_maskFile + _ext
            if os.path.exists(_f):
                os.remove(_f)

    elif mask_type == 'coherence':
        print('generating a mask based on coherence files of sub-band interferograms with a threshold of {}'.format(th))
        cmd = 'imageMath.py -e="(a>{0})*(b>{0})" --a={1} --b={2} -t byte -s BIL -o {3}'.format(th, lowBandCor, highBandCor, maskFile)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError('Failed to generate mask file using coherence files. Command: {}'.format(cmd))

    elif mask_type == 'connected_components':
        print('generating a mask based on .conncomp files')
        cmd = 'imageMath.py -e="(a>0)*(b>0)" --a={0} --b={1} -t byte -s BIL -o {2}'.format(
            lowBandIgram + '.conncomp', highBandIgram + '.conncomp', maskFile)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError('Failed to generate mask file using connected components. Command: {}'.format(cmd))

    else:
        print('generating a mask based on unwrapped files. Pixels with phase = 0 are masked out.')
        cmd = 'imageMath.py -e="(a_1!=0)*(b_1!=0)" --a={0} --b={1} -t byte -s BIL -o {2}'.format(lowBandIgram, highBandIgram, maskFile)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError('Failed to generate mask file using unwrapped files. Command: {}'.format(cmd))
    
    # Apply water body mask if available (matching StripmapProc behavior)
    # Check for water body file in the interferogram directory
    ifgDirname = os.path.dirname(lowBandIgram)
    
    # Try to find water body file with multilook suffix
    numberRangeLooksIon = getattr(inps, 'numberRangeLooksIon', None)
    numberAzimuthLooksIon = getattr(inps, 'numberAzimuthLooksIon', None)
    
    if numberRangeLooksIon and numberAzimuthLooksIon:
        azLooks = getattr(inps, 'azLooks', 1)
        rngLooks = getattr(inps, 'rngLooks', 1)
        totalAzLooks = int(azLooks * numberAzimuthLooksIon)
        totalRgLooks = int(rngLooks * numberRangeLooksIon)
        ml2 = '_{}rlks_{}alks'.format(totalRgLooks, totalAzLooks)
        wbdFile = os.path.join(ifgDirname, 'wbd' + ml2 + '.wbd')
    else:
        # Try without multilook suffix
        wbdFile = os.path.join(ifgDirname, 'wbd.wbd')
    
    # Also check in parent directory
    if not os.path.exists(wbdFile + '.xml'):
        parentDir = os.path.dirname(ifgDirname)
        wbdFile = os.path.join(parentDir, 'wbd.wbd')
    
    # Apply water body mask if found
    if os.path.exists(wbdFile + '.xml'):
        logger.info('Applying water body mask from: {}'.format(wbdFile))
        # Load mask and water body files
        img_mask = isceobj.createImage()
        img_mask.load(maskFile + '.xml')
        width = img_mask.width
        length = img_mask.length
        
        mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
        wbd = np.fromfile(wbdFile, dtype=np.int8).reshape(length, width)
        
        # Mask out water body regions (wbd==-1 means water)
        mask[np.nonzero(wbd==-1)] = 0
        
        # Save updated mask
        mask.astype(np.byte).tofile(maskFile)
        logger.info('Water body mask applied: {} pixels masked out'.format(np.sum(wbd==-1)))

    # Verify that mask file was created
    if not os.path.exists(maskFile):
        raise RuntimeError('Mask file was not created: {}'.format(maskFile))
    if not os.path.exists(maskFile + '.xml'):
        raise RuntimeError('Mask file XML was not created: {}'.format(maskFile + '.xml'))

def unwrapp_error_correction(f0, B, dispFile, nonDispFile,lowBandIgram, highBandIgram, jumpsFile, y_ref=None, x_ref=None):

    dFile = os.path.join(os.path.dirname(dispFile) , "dJumps.bil")
    mFile = os.path.join(os.path.dirname(dispFile) , "mJumps.bil")

    if y_ref and x_ref:
        refL = getValue(lowBandIgram, 2, y_ref, x_ref)
        refH = getValue(highBandIgram, 2, y_ref, x_ref)

    else:
        refL = 0.0
        refH = 0.0

    #cmd = 'imageMath.py -e="round(((a_1-{7}) - (b_1-{8}) - (2.0*{0}/3.0/{1})*c + (2.0*{0}/3.0/{1})*f )/2.0/PI)" --a={2} --b={3} --c={4} --f={5}  -o {6} -t float32 -s BIL'.format(B, f0, highBandIgram, lowBandIgram, nonDispFile, dispFile, dFile, refH, refL)

    cmd = 'imageMath.py -e="round(((a_1+(2.0*PI*g)) - (b_1) - (2.0*{0}/3.0/{1})*c + (2.0*{0}/3.0/{1})*f )/2.0/PI)" --a={2} --b={3} --c={4} --f={5} --g={6}  -o {7} -t float32 -s BIL'.format(B, f0, highBandIgram, lowBandIgram, nonDispFile, dispFile, jumpsFile, dFile)

    print(cmd)

    os.system(cmd)
    #d = (phH - phL - (2.*B/3./f0)*ph_nondis + (2.*B/3./f0)*ph_iono )/2./PI
    #d = np.round(d)

    #cmd = 'imageMath.py -e="round(((a_1 - {6}) + (b_1-{7}) - 2.0*c - 2.0*f )/4.0/PI - g/2)" --a={0} --b={1} --c={2} --f={3} --g={4} -o {5} -t float32 -s BIL'.format(lowBandIgram, highBandIgram, nonDispFile, dispFile, dFile, mFile, refL, refH)

    cmd = 'imageMath.py -e="round(((a_1 ) + (b_1+(2.0*PI*k)) - 2.0*c - 2.0*f )/4.0/PI - g/2)" --a={0} --b={1} --c={2} --f={3} --g={4} --k={5} -o {6} -t float32 -s BIL'.format(lowBandIgram, highBandIgram, nonDispFile, dispFile, dFile, jumpsFile, mFile)

    print(cmd)

    os.system(cmd)


    #m = (phL + phH - 2*ph_nondis - 2*ph_iono)/4./PI - d/2.
    #m = np.round(m)

    return mFile , dFile

def getBandFrequencies(inps):

    with shelve.open(inps.lowBandShelve, flag='r') as db:
          frameL = db['frame']
          wvl0 = frameL.radarWavelegth
          wvlL = frameL.subBandRadarWavelength

    with shelve.open(inps.highBandShelve, flag='r') as db:
       frameH = db['frame']
       wvlH = frameH.subBandRadarWavelength

       pulseLength = frameH.instrument.pulseLength
       chirpSlope = frameH.instrument.chirpSlope
       # Total Bandwidth
       B = np.abs(chirpSlope)*pulseLength

    return wvl0, wvlL, wvlH, B


def computeNumberOfLooks(inps, wvl0, wvlL, wvlH, B, f0, fL, fH):
    '''
    Compute more accurate number of looks for subband interferograms (ALOS-style)
    This considers bandwidth, azimuth bandwidth, and subband characteristics
    '''
    # Get looks from input parameters
    azLooks = getattr(inps, 'azLooks', 1)
    rgLooks = getattr(inps, 'rngLooks', 1)
    numberRangeLooksIon = getattr(inps, 'numberRangeLooksIon', 16)
    numberAzimuthLooksIon = getattr(inps, 'numberAzimuthLooksIon', 16)
    simpleTotalLooks = float(azLooks) * float(rgLooks) * float(numberRangeLooksIon) * float(numberAzimuthLooksIon)
    
    # Try to get azimuth bandwidth from shelve files
    try:
        with shelve.open(inps.lowBandShelve, flag='r') as db:
            frameL = db['frame']
            # Try to get azimuth bandwidth (may not be available for all sensors)
            if hasattr(frameL.instrument, 'pulseRepetitionFrequency'):
                prf = frameL.instrument.pulseRepetitionFrequency
                # Estimate azimuth bandwidth (for stripmap, typically ~PRF)
                # This is a simplified estimate
                azimuthBandwidth = prf * 0.85  # Typical factor for stripmap
            else:
                # Fallback: use wavelength-based estimate
                azimuthBandwidth = SPEED_OF_LIGHT / wvl0 * 0.1  # Rough estimate
    except:
        # Fallback if shelve access fails
        azimuthBandwidth = SPEED_OF_LIGHT / wvl0 * 0.1
    
    # Try to get range sampling rate
    try:
        with shelve.open(inps.lowBandShelve, flag='r') as db:
            frameL = db['frame']
            if hasattr(frameL.instrument, 'rangeSamplingRate'):
                rangeSamplingRate = frameL.instrument.rangeSamplingRate
            else:
                # Estimate from bandwidth
                rangeSamplingRate = B * 1.2  # Typical oversampling factor
    except:
        rangeSamplingRate = B * 1.2
    
    # Try to get azimuth line interval (pixel spacing in azimuth)
    try:
        with shelve.open(inps.lowBandShelve, flag='r') as db:
            frameL = db['frame']
            if hasattr(frameL, 'azimuthLineInterval'):
                azimuthLineInterval = frameL.azimuthLineInterval
            else:
                # Estimate: typically PRF / ground speed
                azimuthLineInterval = 1.0 / (azimuthBandwidth / (SPEED_OF_LIGHT / wvl0))
    except:
        azimuthLineInterval = 1.0 / (azimuthBandwidth / (SPEED_OF_LIGHT / wvl0))
    
    # Compute number of looks (ALOS-style formula)
    # Assume subband range bandwidth is 1/3 of original range bandwidth
    # This matches the subband splitting approach
    subbandRangeBandwidth = B / 3.0
    
    numberOfLooks = (azimuthLineInterval * azLooks * numberAzimuthLooksIon / (1.0/azimuthBandwidth)) * \
                    (subbandRangeBandwidth / rangeSamplingRate * rgLooks * numberRangeLooksIon)

    # This heuristic estimate is fragile for some sensors/stacks. If it becomes wildly
    # inconsistent with the straightforward multilook count, trust the simple count.
    if (not np.isfinite(numberOfLooks)) or (numberOfLooks <= 0):
        logger.warning('Computed number of looks is invalid ({}). Falling back to simple count {:.2f}.'.format(
            numberOfLooks, simpleTotalLooks))
        numberOfLooks = simpleTotalLooks
    else:
        ratio = numberOfLooks / simpleTotalLooks if simpleTotalLooks > 0 else np.inf
        if ratio < 0.25 or ratio > 4.0:
            logger.warning(
                'Computed number of looks {:.2f} is outside a reasonable range relative to simple count {:.2f} '
                '(ratio {:.3f}). Falling back to simple count.'.format(
                    numberOfLooks, simpleTotalLooks, ratio))
            numberOfLooks = simpleTotalLooks

    logger.info('Computed number of looks for subband interferograms: {:.2f}'.format(numberOfLooks))
    logger.info('  Azimuth bandwidth: {:.2f} Hz'.format(azimuthBandwidth))
    logger.info('  Range sampling rate: {:.2e} Hz'.format(rangeSamplingRate))
    logger.info('  Subband range bandwidth: {:.2e} Hz'.format(subbandRangeBandwidth))
    
    return numberOfLooks


def main(iargs=None):


    inps = cmdLineParse(iargs)

    # Match actual FilterAndCoherence filenames (filt_ vs no filt_) when config was hand-edited or from an older stack
    if getattr(inps, 'lowBandIgramPrefix', None):
        inps.lowBandIgramPrefix = resolve_ifg_prefix_for_unw(
            inps.lowBandIgramPrefix, inps.lowBandIgramUnwMethod)
    if getattr(inps, 'highBandIgramPrefix', None):
        inps.highBandIgramPrefix = resolve_ifg_prefix_for_unw(
            inps.highBandIgramPrefix, inps.highBandIgramUnwMethod)
    if getattr(inps, 'lowBandCoherence', None):
        inps.lowBandCoherence = resolve_coherence_path(inps.lowBandCoherence)
    if getattr(inps, 'highBandCoherence', None):
        inps.highBandCoherence = resolve_coherence_path(inps.highBandCoherence)

    '''
    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
    lowBandIgram = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename)

    if '.flat' in lowBandIgram:
        lowBandIgram = lowBandIgram.replace('.flat', '.unw')
    elif '.int' in lowBandIgram:
        lowBandIgram = lowBandIgram.replace('.int', '.unw')
    else:
        lowBandIgram += '.unw'

    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)
    highBandIgram = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename)

    if '.flat' in highBandIgram:
        highBandIgram = highBandIgram.replace('.flat', '.unw')
    elif '.int' in highBandIgram:
        highBandIgram = highBandIgram.replace('.int', '.unw')
    else:
        highBandIgram += '.unw'

    '''

    ##########

    # construct the unwrap and  unwrap connected component filenames for both high and low band interferogams
    # allow for different connected component files for the low and high band images depending what the user preferred
    #       for snaphu2stage: use snaphu connected component
    #       for snaphu: use snaphu connected component
    #       for icu: use icu connected component
    # lowband file
    if inps.lowBandIgramUnwMethod == 'snaphu' or inps.lowBandIgramUnwMethod == 'snaphu2stage':
        lowBandconncomp = inps.lowBandIgramPrefix + '_snaphu.unw.conncomp'
    elif inps.lowBandIgramUnwMethod == 'icu':
        lowBandconncomp = inps.lowBandIgramPrefix + '_icu.unw.conncomp'
    inps.lowBandconncomp = lowBandconncomp
    inps.lowBandIgram = inps.lowBandIgramPrefix + '_' + inps.lowBandIgramUnwMethod + '.unw'
    # highband file
    if inps.highBandIgramUnwMethod == 'snaphu' or inps.highBandIgramUnwMethod == 'snaphu2stage':
        highBandconncomp = inps.highBandIgramPrefix + '_snaphu.unw.conncomp'
    elif inps.highBandIgramUnwMethod == 'icu':
        highBandconncomp = inps.highBandIgramPrefix + '_icu.unw.conncomp'
    inps.highBandconncomp = highBandconncomp
    inps.highBandIgram = inps.highBandIgramPrefix + '_' + inps.highBandIgramUnwMethod + '.unw'
    # print a summary for the user
    print('Files to be used for estimating ionosphere:')
    print('**Low band files:')
    print(inps.lowBandIgram)
    print(inps.lowBandconncomp)
    print('**High band files:')
    print(inps.highBandIgram)
    print(inps.highBandconncomp)

    # generate the output directory if it does not exist yet, and back-up the shelve files
    os.makedirs(inps.outDir, exist_ok=True)
    lowBandShelve = os.path.join(inps.outDir, 'lowBandShelve')
    highBandShelve = os.path.join(inps.outDir, 'highBandShelve')
    os.makedirs(lowBandShelve, exist_ok=True)
    os.makedirs(highBandShelve, exist_ok=True)
    cmdCp = 'cp ' + inps.lowBandShelve + '* ' + lowBandShelve
    os.system(cmdCp)
    cmdCp = 'cp ' + inps.highBandShelve + '* ' + highBandShelve
    os.system(cmdCp)
    inps.lowBandShelve = os.path.join(lowBandShelve, 'data')
    inps.highBandShelve = os.path.join(highBandShelve, 'data')

    
 
    '''
    outputDir = self.insar.ionosphereDirname
    os.makedirs(outputDir, exist_ok=True)
    '''

    outDispersive = os.path.join(inps.outDir, 'iono.bil')
    sigmaDispersive = outDispersive + ".sig"

    outNonDispersive = os.path.join(inps.outDir, 'nonDispersive.bil') 
    sigmaNonDispersive = outNonDispersive + ".sig"

    inps.Sig_phi_L = os.path.join(inps.outDir, 'lowBand.Sigma')
    inps.Sig_phi_H = os.path.join(inps.outDir, 'highBand.Sigma')

    maskFile = os.path.join(inps.outDir, "mask.bil")

    #referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)
    wvl, wvlL, wvlH, B = getBandFrequencies(inps)
    
    f0 = SPEED_OF_LIGHT/wvl
    fL = SPEED_OF_LIGHT/wvlL
    fH = SPEED_OF_LIGHT/wvlH

    ###Determine looks
    #azLooks, rgLooks = self.insar.numberOfLooks( referenceFrame, self.posting,
    #                                    self.numberAzimuthLooks, self.numberRangeLooks)

    #########################################################
    # Look for multilooked unwrapped interferograms for ionosphere estimation
    # These should have been created by unwrapping multilooked .int files (from crossmul step)
    numberRangeLooksIon = getattr(inps, 'numberRangeLooksIon', None)
    numberAzimuthLooksIon = getattr(inps, 'numberAzimuthLooksIon', None)
    
    # Use default values if not specified
    if numberRangeLooksIon is None:
        numberRangeLooksIon = 16
    if numberAzimuthLooksIon is None:
        numberAzimuthLooksIon = 16
    
    useMultilookedUnw = False
    lowBandIgramForIono = inps.lowBandIgram
    highBandIgramForIono = inps.highBandIgram
    
    if numberRangeLooksIon > 1 or numberAzimuthLooksIon > 1:
        # Check if unwrapped files from multilooked interferograms exist
        # The unwrapped files should have the same resolution as the multilooked .int files
        # We check by looking for files that might have been unwrapped from multilooked .int files
        # Since the unwrap step uses multilooked .int files, the .unw files will have multilooked resolution
        # but may not have the multilook suffix in the filename
        
        # First, try to find unwrapped files that match the multilooked pattern
        # The pattern would be: original_name_6rlks_6alks.unw (if unwrapped from multilooked .int)
        # But the unwrap step might not add this suffix, so we need to check dimensions
        
        # For now, we'll use the regular unwrapped files if they exist
        # The dimensions should already be multilooked if the unwrap step used multilooked .int files
        logger.info('Using unwrapped interferograms (should be multilooked if unwrapped from multilooked .int files)')
        logger.info('Low band: {}'.format(lowBandIgramForIono))
        logger.info('High band: {}'.format(highBandIgramForIono))
        
        # Verify that the files exist and have multilooked dimensions
        if os.path.exists(lowBandIgramForIono + '.xml') and os.path.exists(highBandIgramForIono + '.xml'):
            # Check dimensions to verify they are multilooked
            imgLow = isceobj.createImage()
            imgLow.load(lowBandIgramForIono + '.xml')
            imgHigh = isceobj.createImage()
            imgHigh.load(highBandIgramForIono + '.xml')
            
            # Get original looks
            azLooks = getattr(inps, 'azLooks', 1)
            rgLooks = getattr(inps, 'rngLooks', 1)
            
            # Expected multilooked dimensions
            # We can't easily determine original dimensions here, so we'll assume
            # the unwrapped files are already at the correct resolution if they exist
            useMultilookedUnw = True
            logger.info('Using unwrapped interferograms for ionosphere estimation')

    #########################################################
    # Adjust phase using polynomial fitting (ALOS-style) if requested
    # This adjusts the upper band phase to remove relative phase unwrapping errors
    adjustPhase = getattr(inps, 'adjustPhasePolynomial', True)
    highBandIgramForIonoAdjusted = highBandIgramForIono
    
    if adjustPhase:
        logger.info('Applying polynomial phase adjustment (ALOS-style)')
        try:
            highBandIgramForIonoAdjusted = adjust_phase_polynomial(
                lowBandIgramForIono,
                highBandIgramForIono,
                inps.outDir,
                lowBandCoherence=inps.lowBandCoherence,
                highBandCoherence=inps.highBandCoherence,
                coherence_weight_threshold=getattr(inps, 'adjustPhaseCoherenceThreshold', DEFAULT_ADJUST_PHASE_COHERENCE_THRESHOLD),
            )
        except Exception as e:
            logger.warning('Polynomial phase adjustment failed: {}. Using original interferograms.'.format(e))
            highBandIgramForIonoAdjusted = highBandIgramForIono
    else:
        logger.info('Skipping polynomial phase adjustment')
    
    #########################################################
    # make sure the low-band and high-band interferograms have consistent unwrapping errors. 
    # For this we estimate jumps as the difference of lowBand and highBand phases divided by 2PI
    # The assumprion is that bothe interferograms are flattened and the phase difference between them
    # is less than 2PI. This assumprion is valid for current sensors. It needs to be evaluated for
    # future sensors like NISAR.
    # Use adjusted high band interferogram if available
    jumpsFile = check_consistency(
        lowBandIgramForIono,
        highBandIgramForIonoAdjusted,
        inps.outDir,
        global_integer=getattr(inps, 'jumpGlobalInteger', DEFAULT_JUMP_GLOBAL_INTEGER),
    )

    #########################################################
    # estimating the dispersive and non-dispersive components
    # Use adjusted high band interferogram if available
    dispersive_nonDispersive(lowBandIgramForIono, highBandIgramForIonoAdjusted, f0, fL, fH, outDispersive, outNonDispersive, jumpsFile)

    # generating a mask which will help filtering the estimated dispersive and non-dispersive phase
    # Use multilooked interferograms for mask generation if they were used for ionosphere estimation
    getMask(inps, maskFile, lowBandIgram=lowBandIgramForIono, highBandIgram=highBandIgramForIono)

    # Calculating the theoretical standard deviation of the estimation based on the coherence of the interferograms
    # Use more accurate number of looks calculation (ALOS-style) if possible
    try:
        numberOfLooks = computeNumberOfLooks(inps, wvl, wvlL, wvlH, B, f0, fL, fH)
        # Use the computed numberOfLooks for variance calculation
        # Note: theoretical_variance_fromSubBands uses totalLooks, so we'll pass numberOfLooks
        # But we need to check if the function can handle this properly
        # For now, we'll compute a conversion factor
        azLooks = getattr(inps, 'azLooks', 1)
        rgLooks = getattr(inps, 'rngLooks', 1)
        simpleTotalLooks = azLooks * rgLooks
        if useMultilookedUnw and numberRangeLooksIon and numberAzimuthLooksIon:
            simpleTotalLooks = simpleTotalLooks * numberRangeLooksIon * numberAzimuthLooksIon
        # Use the more accurate calculation only if it stays reasonably close to
        # the straightforward multilook count. Some sensors/stacks produce wildly
        # unrealistic values here due to metadata/unit mismatches.
        totalLooks = numberOfLooks if numberOfLooks > 0 else simpleTotalLooks
        if (not np.isfinite(totalLooks)) or (totalLooks <= 0):
            logger.warning('Using invalid computed number of looks {}. Falling back to simple count {:.2f}.'.format(
                totalLooks, simpleTotalLooks))
            totalLooks = simpleTotalLooks
        else:
            ratio = totalLooks / simpleTotalLooks if simpleTotalLooks > 0 else np.inf
            if ratio < 0.25 or ratio > 4.0:
                logger.warning(
                    'Computed number of looks {:.2f} is inconsistent with simple count {:.2f} '
                    '(ratio {:.3f}); using simple count instead.'.format(
                        totalLooks, simpleTotalLooks, ratio))
                totalLooks = simpleTotalLooks
        logger.info('Using number of looks: {:.2f} (simple calculation: {:.2f})'.format(totalLooks, simpleTotalLooks))
    except Exception as e:
        logger.warning('Failed to compute accurate number of looks: {}. Using simple calculation.'.format(e))
        azLooks = getattr(inps, 'azLooks', 1)
        rgLooks = getattr(inps, 'rngLooks', 1)
        totalLooks = azLooks * rgLooks
        if useMultilookedUnw and numberRangeLooksIon and numberAzimuthLooksIon:
            totalLooks = totalLooks * numberRangeLooksIon * numberAzimuthLooksIon
    theoretical_variance_fromSubBands(inps, f0, fL, fH, B, sigmaDispersive, sigmaNonDispersive, totalLooks) 

    # Use adaptive Gaussian filtering if explicitly requested, otherwise use original iterative filtering
    useAdaptiveFilter = getattr(inps, 'useAdaptiveGaussian', True)
    fitNonDisp = getattr(inps, 'fitNonDispersive', False)
    filtNonDisp = getattr(inps, 'filtNonDispersive', False)
    if useAdaptiveFilter:
        # Use adaptive Gaussian filtering (similar to StripmapProc)
        logger.info('Using adaptive Gaussian filtering for ionospheric phase')
        import scipy.signal as ss
        
        # Read data and std - need to get dimensions first
        img = isceobj.createImage()
        img.load(outDispersive + '.xml')
        width = img.width
        length = img.length
        
        ionos = np.fromfile(outDispersive, dtype=np.float32).reshape(length, width)
        std = np.fromfile(sigmaDispersive, dtype=np.float32).reshape(length, width)
        mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
        
        # Apply mask: mask==0 marks invalid samples
        ionos[mask==0] = 0
        std[mask==0] = 0

        # Alos2Proc ion_filt: std/ion invalid where either sub-band coherence ~0 (single-band .cor OK)
        g2d = None
        cor_low_ion = read_coherence_2d(inps.lowBandCoherence, length, width)
        cor_high_ion = read_coherence_2d(inps.highBandCoherence, length, width)
        apply_alos_style_dual_band_invalid(ionos, std, cor_low_ion, cor_high_ion)
        
        # Get filtering parameters (defaults match StripmapProc/alosStack.xml)
        size_max = getattr(inps, 'filteringWinsizeMaxIon', 501)
        size_min = getattr(inps, 'filteringWinsizeMinIon', 51)
        size_secondary = getattr(inps, 'filteringWinsizeSecondaryIon', 5)
        std_out0 = getattr(inps, 'filterStdIon', None)
        fitAdaptive = getattr(inps, 'fitAdaptiveIon', True)
        filtSecondary = getattr(inps, 'filtSecondaryIon', True)
        fitIon = getattr(inps, 'fitIon', True)
        filtIon = getattr(inps, 'filtIon', True)
        corThresholdFit = getattr(inps, 'fitIonCoherenceThreshold', 0.25)
        
        # Check that at least one of fit or filt is enabled
        if (not fitIon) and (not filtIon):
            raise Exception('either fit_ion or filt_ion should be True when doing ionospheric correction')
        
        # If std_out0 is None, use a reasonable default
        if std_out0 is None:
            std_out0 = 0.005  # Default fallback
        
        if size_min > size_max:
            size_max = size_min
        if size_secondary % 2 != 1:
            size_secondary += 1
            logger.info('Window size of secondary filtering should be odd, changed to {}'.format(size_secondary))
        
        # Global polynomial fitting (ALOS-style) before filtering
        ionos_fit = None
        if fitIon:
            logger.info('Applying global polynomial fit to ionospheric phase (ALOS-style)')
            wgt = polyfit_variance_weights_from_std_coherence(
                std, cor_low_ion, cor_high_ion, corThresholdFit)
            
            # Normalize weight
            index = np.nonzero(wgt!=0)
            if len(index[0]) > 0:
                wgt[index] = 1.0/(wgt[index])
                # Fit 2nd order polynomial
                ionos_fit, coeff = polyfit_2d(ionos.copy(), wgt, 2)
                # Subtract fit from original data (only where data is valid)
                ionos = ionos - ionos_fit * (ionos!=0)
                logger.info('Global polynomial fit completed')
            else:
                logger.warning('No valid pixels for global polynomial fitting, skipping fit step')
                fitIon = False
        
        # Filter dispersive phase (only if filtIon is enabled)
        ionos_filt = None
        std_filt = None
        window_size = None
        if filtIon:
            ionos_filt, std_filt, window_size = adaptive_gaussian(
                ionos.copy(), std.copy(), size_min, size_max, std_out0, fit=fitAdaptive,
                )
        
            # Apply secondary filtering if requested
            if filtSecondary:
                logger.info('Applying secondary filtering with window size {}'.format(size_secondary))
                # Create Gaussian kernel for secondary filtering
                hsize = (size_secondary - 1) / 2
                x = np.arange(-hsize, hsize + 1)
                f = np.exp(-x**2/(2.0*(size_secondary/2.0)**2)) / ((size_secondary/2.0) * np.sqrt(2.0*np.pi))
                g2d = np.outer(f, f)
                g2d = g2d / np.sum(g2d)
                # Apply secondary filtering
                scale = ss.fftconvolve((ionos_filt!=0).astype(np.float32), g2d, mode='same')
                ionos_filt = (ionos_filt!=0) * ss.fftconvolve(ionos_filt, g2d, mode='same') / (scale + (scale==0))
        
        # Combine fit and filt results (ALOS-style)
        if fitIon and filtIon:
            ionos_final = ionos_filt + ionos_fit * (ionos_filt!=0)
        elif fitIon and not filtIon:
            ionos_final = ionos_fit
        elif not fitIon and filtIon:
            ionos_final = ionos_filt
        else:
            ionos_final = ionos

        # Save filtered results
        ionos_final.astype(np.float32).tofile(outDispersive + ".filt")
        write_xml(outDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtIon and std_filt is not None:
            std_filt.astype(np.float32).tofile(sigmaDispersive + ".filt")
            write_xml(sigmaDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtIon and window_size is not None:
            window_size.astype(np.float32).tofile(outDispersive + ".filt.win")
            write_xml(outDispersive + ".filt.win", width, length, 1, "FLOAT", "BIL")
        
        # Filter non-dispersive phase
        nonDisp = np.fromfile(outNonDispersive, dtype=np.float32).reshape(length, width)
        std_nonDisp = np.fromfile(sigmaNonDispersive, dtype=np.float32).reshape(length, width)
        nonDisp[mask==0] = 0
        std_nonDisp[mask==0] = 0
        apply_alos_style_dual_band_invalid(nonDisp, std_nonDisp, cor_low_ion, cor_high_ion)
        
        # Global polynomial fitting for non-dispersive phase
        nonDisp_fit = None
        if fitNonDisp:
            wgt = polyfit_variance_weights_from_std_coherence(
                std_nonDisp, cor_low_ion, cor_high_ion, corThresholdFit)
            index = np.nonzero(wgt!=0)
            if len(index[0]) > 0:
                wgt[index] = 1.0/(wgt[index])
                nonDisp_fit, _ = polyfit_2d(nonDisp.copy(), wgt, 2)
                nonDisp = nonDisp - nonDisp_fit * (nonDisp!=0)
        
        nonDisp_filt = None
        std_nonDisp_filt = None
        if filtNonDisp:
            nonDisp_filt, std_nonDisp_filt, _ = adaptive_gaussian(
                nonDisp.copy(), std_nonDisp.copy(), size_min, size_max, std_out0, fit=fitAdaptive,
                )
            
            # Apply secondary filtering to non-dispersive phase if requested
            if filtSecondary:
                # Create Gaussian kernel if not already created (from dispersive phase filtering)
                if g2d is None:
                    hsize = (size_secondary - 1) / 2
                    x = np.arange(-hsize, hsize + 1)
                    f = np.exp(-x**2/(2.0*(size_secondary/2.0)**2)) / ((size_secondary/2.0) * np.sqrt(2.0*np.pi))
                    g2d = np.outer(f, f)
                    g2d = g2d / np.sum(g2d)
                scale = ss.fftconvolve((nonDisp_filt!=0).astype(np.float32), g2d, mode='same')
                nonDisp_filt = (nonDisp_filt!=0) * ss.fftconvolve(nonDisp_filt, g2d, mode='same') / (scale + (scale==0))
        
        # Combine fit and filt results for non-dispersive phase
        if fitNonDisp and filtNonDisp:
            nonDisp_final = nonDisp_filt + nonDisp_fit * (nonDisp_filt!=0)
        elif fitNonDisp and not filtNonDisp:
            nonDisp_final = nonDisp_fit
        elif not fitNonDisp and filtNonDisp:
            nonDisp_final = nonDisp_filt
        else:
            nonDisp_final = nonDisp

        nonDisp_final.astype(np.float32).tofile(outNonDispersive + ".filt")
        write_xml(outNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtNonDisp and std_nonDisp_filt is not None:
            std_nonDisp_filt.astype(np.float32).tofile(sigmaNonDispersive + ".filt")
            write_xml(sigmaNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        
        del ionos, std, mask, nonDisp, std_nonDisp
        if ionos_filt is not None:
            del ionos_filt, std_filt
        if nonDisp_filt is not None:
            del nonDisp_filt, std_nonDisp_filt
        if ionos_fit is not None:
            del ionos_fit
        if nonDisp_fit is not None:
            del nonDisp_fit
    else:
        # Original iterative filtering method
        # low pass filtering the dispersive phase
        lowPassFilter(outDispersive, sigmaDispersive, maskFile, 
                        inps.kernel_x_size, inps.kernel_y_size, 
                        inps.kernel_sigma_x, inps.kernel_sigma_y, 
                        iteration = inps.dispersive_filter_iterations, 
                        theta = inps.kernel_rotation)

        # low pass filtering the  non-dispersive phase
        lowPassFilter(outNonDispersive, sigmaNonDispersive, maskFile, 
                        inps.kernel_x_size, inps.kernel_y_size,
                        inps.kernel_sigma_x, inps.kernel_sigma_y,
                        iteration = inps.dispersive_filter_iterations,
                        theta = inps.kernel_rotation)
            
    # Estimating phase unwrapping errors
    # Use adjusted high band interferogram if available
    mFile , dFile = unwrapp_error_correction(f0, B, outDispersive+".filt", outNonDispersive+".filt", 
                                                    inps.lowBandIgram, highBandIgramForIonoAdjusted, jumpsFile)

    # re-estimate the dispersive and non-dispersive phase components by taking into account the unwrapping errors
    # Use adjusted high band interferogram if available
    outDispersive = outDispersive + ".unwCor"
    outNonDispersive = outNonDispersive + ".unwCor"
    dispersive_nonDispersive(inps.lowBandIgram, highBandIgramForIonoAdjusted, f0, fL, fH, outDispersive, outNonDispersive, jumpsFile, m=mFile , d=dFile)

    # Filter the corrected estimates
    if useAdaptiveFilter:
        # Use adaptive Gaussian filtering again
        ionos = np.fromfile(outDispersive, dtype=np.float32).reshape(length, width)
        std = np.fromfile(sigmaDispersive, dtype=np.float32).reshape(length, width)
        mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
        ionos[mask==0] = 0
        std[mask==0] = 0

        cor_low_ion = read_coherence_2d(inps.lowBandCoherence, length, width)
        cor_high_ion = read_coherence_2d(inps.highBandCoherence, length, width)
        apply_alos_style_dual_band_invalid(ionos, std, cor_low_ion, cor_high_ion)
        
        # Global polynomial fitting for corrected dispersive phase
        ionos_fit = None
        if fitIon:
            wgt = polyfit_variance_weights_from_std_coherence(
                std, cor_low_ion, cor_high_ion, corThresholdFit)
            index = np.nonzero(wgt!=0)
            if len(index[0]) > 0:
                wgt[index] = 1.0/(wgt[index])
                ionos_fit, _ = polyfit_2d(ionos.copy(), wgt, 2)
                ionos = ionos - ionos_fit * (ionos!=0)
        
        ionos_filt = None
        std_filt = None
        window_size = None
        g2d = None
        if filtIon:
            ionos_filt, std_filt, window_size = adaptive_gaussian(
                ionos.copy(), std.copy(), size_min, size_max, std_out0, fit=fitAdaptive,
                )
            
            if filtSecondary:
                # Create Gaussian kernel for secondary filtering
                hsize = (size_secondary - 1) / 2
                x = np.arange(-hsize, hsize + 1)
                f = np.exp(-x**2/(2.0*(size_secondary/2.0)**2)) / ((size_secondary/2.0) * np.sqrt(2.0*np.pi))
                g2d = np.outer(f, f)
                g2d = g2d / np.sum(g2d)
                scale = ss.fftconvolve((ionos_filt!=0).astype(np.float32), g2d, mode='same')
                ionos_filt = (ionos_filt!=0) * ss.fftconvolve(ionos_filt, g2d, mode='same') / (scale + (scale==0))
        
        # Combine fit and filt results
        if fitIon and filtIon:
            ionos_final = ionos_filt + ionos_fit * (ionos_filt!=0)
        elif fitIon and not filtIon:
            ionos_final = ionos_fit
        elif not fitIon and filtIon:
            ionos_final = ionos_filt
        else:
            ionos_final = ionos

        ionos_final.astype(np.float32).tofile(outDispersive + ".filt")
        write_xml(outDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtIon and std_filt is not None:
            std_filt.astype(np.float32).tofile(sigmaDispersive + ".filt")
            write_xml(sigmaDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtIon and window_size is not None:
            window_size.astype(np.float32).tofile(outDispersive + ".filt.win")
            write_xml(outDispersive + ".filt.win", width, length, 1, "FLOAT", "BIL")
        
        nonDisp = np.fromfile(outNonDispersive, dtype=np.float32).reshape(length, width)
        std_nonDisp = np.fromfile(sigmaNonDispersive, dtype=np.float32).reshape(length, width)
        nonDisp[mask==0] = 0
        std_nonDisp[mask==0] = 0
        apply_alos_style_dual_band_invalid(nonDisp, std_nonDisp, cor_low_ion, cor_high_ion)
        
        # Global polynomial fitting for corrected non-dispersive phase
        nonDisp_fit = None
        if fitNonDisp:
            wgt = polyfit_variance_weights_from_std_coherence(
                std_nonDisp, cor_low_ion, cor_high_ion, corThresholdFit)
            index = np.nonzero(wgt!=0)
            if len(index[0]) > 0:
                wgt[index] = 1.0/(wgt[index])
                nonDisp_fit, _ = polyfit_2d(nonDisp.copy(), wgt, 2)
                nonDisp = nonDisp - nonDisp_fit * (nonDisp!=0)
        
        nonDisp_filt = None
        std_nonDisp_filt = None
        if filtNonDisp:
            nonDisp_filt, std_nonDisp_filt, _ = adaptive_gaussian(
                nonDisp.copy(), std_nonDisp.copy(), size_min, size_max, std_out0, fit=fitAdaptive,
                )
            
            if filtSecondary:
                # Create Gaussian kernel if not already created
                if g2d is None:
                    hsize = (size_secondary - 1) / 2
                    x = np.arange(-hsize, hsize + 1)
                    f = np.exp(-x**2/(2.0*(size_secondary/2.0)**2)) / ((size_secondary/2.0) * np.sqrt(2.0*np.pi))
                    g2d = np.outer(f, f)
                    g2d = g2d / np.sum(g2d)
                scale = ss.fftconvolve((nonDisp_filt!=0).astype(np.float32), g2d, mode='same')
                nonDisp_filt = (nonDisp_filt!=0) * ss.fftconvolve(nonDisp_filt, g2d, mode='same') / (scale + (scale==0))
        
        # Combine fit and filt results for non-dispersive phase
        if fitNonDisp and filtNonDisp:
            nonDisp_final = nonDisp_filt + nonDisp_fit * (nonDisp_filt!=0)
        elif fitNonDisp and not filtNonDisp:
            nonDisp_final = nonDisp_fit
        elif not fitNonDisp and filtNonDisp:
            nonDisp_final = nonDisp_filt
        else:
            nonDisp_final = nonDisp

        nonDisp_final.astype(np.float32).tofile(outNonDispersive + ".filt")
        write_xml(outNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtNonDisp and std_nonDisp_filt is not None:
            std_nonDisp_filt.astype(np.float32).tofile(sigmaNonDispersive + ".filt")
            write_xml(sigmaNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        
        del ionos, std, mask, nonDisp, std_nonDisp
        if ionos_filt is not None:
            del ionos_filt, std_filt
        if nonDisp_filt is not None:
            del nonDisp_filt, std_nonDisp_filt
        if ionos_fit is not None:
            del ionos_fit
        if nonDisp_fit is not None:
            del nonDisp_fit
    else:
        # Original iterative filtering
        lowPassFilter(outDispersive, sigmaDispersive, maskFile, 
                        inps.kernel_x_size, inps.kernel_y_size,
                        inps.kernel_sigma_x, inps.kernel_sigma_y,
                        iteration = inps.dispersive_filter_iterations,
                        theta = inps.kernel_rotation)

        lowPassFilter(outNonDispersive, sigmaNonDispersive, maskFile,
                        inps.kernel_x_size, inps.kernel_y_size,
                        inps.kernel_sigma_x, inps.kernel_sigma_y,
                        iteration = inps.dispersive_filter_iterations,
                        theta = inps.kernel_rotation)
    
    # Resample ionospheric phase back to original interferogram resolution (first multilook, before extra ionospheric looks)
    # The final ionospheric phase should have the same dimensions as the original interferogram
    if useMultilookedUnw and numberRangeLooksIon and numberAzimuthLooksIon and (numberRangeLooksIon > 1 or numberAzimuthLooksIon > 1):
        # Get dimensions of multilooked ionosphere (at extra multilooked resolution)
        img_ion = isceobj.createImage()
        img_ion.load(outDispersive + '.filt.xml')
        width_ion = img_ion.width
        length_ion = img_ion.length
        
        # Get dimensions of original interferogram (first multilook, before extra ionospheric looks)
        # The original interferogram is the one before extra multilooking (e.g., filt_xxx.int, not filt_xxx_6rlks_6alks.int)
        # Find the original interferogram file (first multilook)
        originalIntFile = None
        ifgDirname = os.path.dirname(inps.lowBandIgram)
        
        # Try to find the original interferogram file (first multilook, before extra multilooking)
        # The lowBandIgramPrefix may contain the multilook suffix (e.g., filt_20250813_20250910_6rlks_6alks)
        # We need to remove the multilook suffix to find the original file
        # Original file could be: filt_20250813_20250910.int (filtered) or 20250813_20250910.int (unfiltered)
        import glob
        import re
        baseName = inps.lowBandIgramPrefix
        
        # Remove the multilook suffix from baseName if present
        # Pattern: _Xrlks_Yalks where X and Y are numbers
        ml2_pattern = r'_\d+rlks_\d+alks$'
        if re.search(ml2_pattern, baseName):
            # Remove the multilook suffix
            baseName = re.sub(ml2_pattern, '', baseName)
            logger.info('Removed multilook suffix from baseName, using: {}'.format(baseName))
        
        # First, try to find filtered original interferogram (filt_xxx.int)
        # This is the first multilook + filtered version (before extra multilooking)
        pattern_filt = os.path.join(ifgDirname, baseName + '.int')
        if os.path.exists(pattern_filt + '.xml'):
            originalIntFile = pattern_filt
            logger.info('Found original filtered interferogram: {}'.format(originalIntFile))
        else:
            # If filtered version doesn't exist, try to find unfiltered original (xxx.int)
            # Remove 'filt_' prefix if present
            baseNameUnfilt = baseName
            if baseNameUnfilt.startswith('filt_'):
                baseNameUnfilt = baseNameUnfilt[5:]  # Remove 'filt_' prefix
            pattern_unfilt = os.path.join(ifgDirname, baseNameUnfilt + '.int')
            if os.path.exists(pattern_unfilt + '.xml'):
                originalIntFile = pattern_unfilt
                logger.info('Found original unfiltered interferogram: {}'.format(originalIntFile))
            else:
                # Last resort: search all .int files in directory
                # Look for files that don't have the extra multilook pattern
                allIntFiles = glob.glob(os.path.join(ifgDirname, '*.int'))
                ml2 = '_{}rlks_{}alks'.format(numberRangeLooksIon, numberAzimuthLooksIon)
                for intFile in allIntFiles:
                    # Remove .int extension and .xml if present for comparison
                    intFileBase = os.path.basename(intFile).replace('.int', '').replace('.xml', '')
                    # Check if this file doesn't have the multilook suffix
                    # and matches either the filtered or unfiltered base name
                    if ml2 not in intFileBase:
                        if baseName in intFileBase or baseNameUnfilt in intFileBase:
                            originalIntFile = intFile.replace('.xml', '')
                            logger.info('Found original interferogram (alternative search): {}'.format(originalIntFile))
                            break
        
        # If we found the original interferogram, use its dimensions for resampling
        if originalIntFile and os.path.exists(originalIntFile + '.xml'):
            img_orig = isceobj.createImage()
            img_orig.load(originalIntFile + '.xml')
            width_orig = img_orig.width
            length_orig = img_orig.length
            
            logger.info('Original interferogram found: {} ({}x{})'.format(originalIntFile, length_orig, width_orig))
            logger.info('Ionospheric phase current resolution: {}x{}'.format(length_ion, width_ion))
            
            # Always resample to match original interferogram dimensions
            from scipy.interpolate import interp1d
            
            logger.info('Resampling ionospheric phase from {}x{} to {}x{} (original interferogram resolution)'.format(
                width_ion, length_ion, width_orig, length_orig))
            
            # Resample dispersive phase
            ionos_ml = np.fromfile(outDispersive + '.filt', dtype=np.float32).reshape(length_ion, width_ion)
            
            # Resample in range direction first
            index_rg_ml = np.linspace(0, width_ion-1, num=width_ion, endpoint=True)
            if width_orig != width_ion:
                index_rg_orig = np.linspace(0, width_orig-1, num=width_orig, endpoint=True) * (width_ion-1)/(width_orig-1) if width_orig > 1 else np.array([0])
            else:
                index_rg_orig = index_rg_ml
            
            ionos_resampled_rg = np.zeros((length_ion, width_orig), dtype=np.float32)
            for i in range(length_ion):
                if width_orig == width_ion:
                    ionos_resampled_rg[i, :] = ionos_ml[i, :]
                else:
                    f = interp1d(index_rg_ml, ionos_ml[i, :], kind='cubic', fill_value="extrapolate", bounds_error=False)
                    ionos_resampled_rg[i, :] = f(index_rg_orig)
            
            # Resample in azimuth direction
            if length_orig != length_ion:
                index_az_ml = np.linspace(0, length_ion-1, num=length_ion, endpoint=True)
                index_az_orig = np.linspace(0, length_orig-1, num=length_orig, endpoint=True) * (length_ion-1)/(length_orig-1) if length_orig > 1 else np.array([0])
                ionos_final = np.zeros((length_orig, width_orig), dtype=np.float32)
                for j in range(width_orig):
                    f = interp1d(index_az_ml, ionos_resampled_rg[:, j], kind='cubic', fill_value="extrapolate", bounds_error=False)
                    ionos_final[:, j] = f(index_az_orig)
            else:
                ionos_final = ionos_resampled_rg
            
            # Save resampled dispersive phase
            ionos_final.astype(np.float32).tofile(outDispersive + ".filt")
            write_xml(outDispersive + ".filt", width_orig, length_orig, 1, "FLOAT", "BIL")
            
            # Resample non-dispersive phase
            nonDisp_ml = np.fromfile(outNonDispersive + '.filt', dtype=np.float32).reshape(length_ion, width_ion)
            
            nonDisp_resampled_rg = np.zeros((length_ion, width_orig), dtype=np.float32)
            for i in range(length_ion):
                if width_orig == width_ion:
                    nonDisp_resampled_rg[i, :] = nonDisp_ml[i, :]
                else:
                    f = interp1d(index_rg_ml, nonDisp_ml[i, :], kind='cubic', fill_value="extrapolate", bounds_error=False)
                    nonDisp_resampled_rg[i, :] = f(index_rg_orig)
            
            if length_orig != length_ion:
                nonDisp_final = np.zeros((length_orig, width_orig), dtype=np.float32)
                for j in range(width_orig):
                    f = interp1d(index_az_ml, nonDisp_resampled_rg[:, j], kind='cubic', fill_value="extrapolate", bounds_error=False)
                    nonDisp_final[:, j] = f(index_az_orig)
            else:
                nonDisp_final = nonDisp_resampled_rg
            
            # Save resampled non-dispersive phase
            nonDisp_final.astype(np.float32).tofile(outNonDispersive + ".filt")
            write_xml(outNonDispersive + ".filt", width_orig, length_orig, 1, "FLOAT", "BIL")
            
            logger.info('Ionospheric phase resampled from {}x{} to {}x{} (original interferogram resolution)'.format(
                width_ion, length_ion, width_orig, length_orig))
            
            del ionos_ml, ionos_resampled_rg, ionos_final, nonDisp_ml, nonDisp_resampled_rg, nonDisp_final
        else:
            # Construct expected pattern for warning message
            expected_pattern = os.path.join(ifgDirname, baseName + '.int')
            logger.warning('Original interferogram file not found, cannot resample. Expected file pattern: {}'.format(expected_pattern))
            logger.warning('Ionospheric phase will remain at extra multilooked resolution: {}x{}'.format(length_ion, width_ion))


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

