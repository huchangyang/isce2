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
DEFAULT_IONO_COHERENCE_THRESHOLD = 0.4
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


def apply_alos_style_dual_band_invalid(ion, std, cor_low, cor_high, zero_data=True):
    """
    Alos2Proc ion_filt: if either sub-band coherence is ~0, mark pixel invalid for
    weighting (std=0). Optionally also zero ion (legacy behaviour).
    """
    if cor_low is None or cor_high is None:
        return
    if cor_low.shape != ion.shape or cor_high.shape != ion.shape:
        logger.warning('Coherence array shape mismatch; skipping dual-band invalid mask.')
        return
    invalid = (cor_low <= 1e-6) | (cor_high <= 1e-6)
    idx = np.nonzero(invalid)
    if idx[0].size:
        std[idx] = 0
        if zero_data:
            ion[idx] = 0


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
    parser.add_argument('--full_band_igram', dest='fullBandIgram', type=str, default=None,
            help='full-band wrapped interferogram (.int). Used to define output-mask valid '
                 'pixels (amplitude > 0) and optional legacy full_band_igram output masking.')
    parser.add_argument('--water_mask', dest='waterMask', type=str, default=None,
            help='water body mask in radar coordinates (e.g. geom_reference/waterMask.rdr). '
                 'Applied to compute and output masks. Auto-discovered under geom_reference/ if omitted.')
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
    parser.add_argument('--ion_output_mask_type', dest='ionOutputMaskType', type=str, default='int_valid',
            choices=['int_valid', 'water_and_int', 'water_and_unw', 'full_band_igram', 'compute_mask', 'none'],
            help='mask applied ONLY to final ion output files. '
                 'int_valid (default): sub-band .int amp>0 intersect full-band .int (strict downsample); '
                 'no water mask (avoids boundary artifacts when upsampling to full-band grid). '
                 'water_and_int: alias for int_valid; '
                 'water_and_unw: legacy alias for int_valid; '
                 'full_band_igram: full-band .int amplitude>0 only; '
                 'compute_mask: same as mask.bil; none: no final output masking. '
                 'mask.bil (dispersive_filter_mask_type + unw + water) always controls filtering weights.')
    parser.add_argument('--output_int_amplitude_threshold', dest='outputIntAmplitudeThreshold', type=float,
            default=0.0,
            help='minimum .int amplitude for output_mask valid pixels (default 0). '
                 'Increase slightly (e.g. 1e-4) if edge noise keeps amp>0 in no-data areas.')
    parser.add_argument('--output_int_relative_amplitude_fraction', dest='outputIntRelativeAmpFraction',
            type=float, default=0.0,
            help='if >0, also require amp > fraction * median(positive amps) in each .int (default 0=off). '
                 'Example: 0.02 masks weak edge leakage.')
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
            help='target standard deviation for adaptive filtering (default=None, use 0.005 rad)')
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
    
    # True no-data (e.g. unw==0). Pixels with std==0 but data!=0 may still receive
    # filtered values from neighbours but do not contribute (wgt=0).
    nodata = (data == 0)
    index = np.nonzero(nodata)
    data[index] = 0
    std[index] = 0
    # Compute weight using standard deviation
    wgt = 1.0 / (std**2 + (std==0))
    wgt[nodata] = 0
    wgt[std == 0] = 0
    
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
    n_skip_nodata = int(np.sum(nodata))
    if n_skip_nodata > 0:
        print('skip window search at {} no-data pixels (data==0)'.format(n_skip_nodata))
    for i in range(length):
        if (((i+1)%50) == 0):
            print('processing line %6d of %6d' % (i+1, length), end='\r', flush=True)
        for j in range(width):
            # Do not search for a window at no-data pixels.  FFT convolution can
            # assign non-zero std_filt near valid areas, which would otherwise
            # force the maximum window (very slow, no useful filter result).
            if nodata[i, j]:
                gaussian_index[i, j] = -1
                continue
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


def _load_mask_raster_2d(path):
    """Load a single-band raster as (data, length, width) using ISCE metadata."""
    img = isceobj.createImage()
    img.load(path + '.xml')
    length, width = int(img.length), int(img.width)
    dtype_name = str(getattr(img, 'dataType', 'BYTE')).upper()
    if dtype_name in ('BYTE', 'UINT8'):
        dtype = np.uint8
    elif dtype_name in ('INT8',):
        dtype = np.int8
    elif dtype_name in ('FLOAT', 'FLOAT32'):
        dtype = np.float32
    else:
        dtype = np.int8
    data = np.fromfile(path, dtype=dtype).reshape(length, width)
    return data, length, width


def _water_pixels_from_raster(wbd):
    """
    Return boolean array where True marks water body pixels.

    stripmapStack waterMask.rdr: 0=water, 1=land (StripmapProc convention).
    Legacy ALOS wbd / SWBD geo fill: -1=water.
    """
    vals = set(np.unique(wbd).tolist())
    if -1 in vals:
        return wbd == -1
    if 0 in vals and 1 in vals:
        return wbd == 0
    if 0 in vals:
        return wbd == 0
    logger.warning('Unrecognized water mask values {}; treating 0 as water.'.format(sorted(vals)))
    return wbd == 0


def _resolve_water_mask_path(inps, lowBandIgram=None):
    """
    Locate a water body mask raster (.rdr or legacy .wbd + .xml).

    Prefers explicit --water_mask, then stripmapStack geom_reference/waterMask.rdr,
    then legacy ALOS-style wbd*.wbd next to the sub-band interferograms.
    """
    explicit = getattr(inps, 'waterMask', None)
    if explicit and os.path.exists(explicit + '.xml'):
        return explicit

    search_roots = []
    if lowBandIgram:
        search_roots.append(os.path.dirname(os.path.abspath(lowBandIgram)))
    out_dir = getattr(inps, 'outDir', None)
    if out_dir:
        search_roots.append(os.path.dirname(os.path.abspath(out_dir)))

    number_range_looks_ion = getattr(inps, 'numberRangeLooksIon', None)
    number_azimuth_looks_ion = getattr(inps, 'numberAzimuthLooksIon', None)
    ml2 = None
    if number_range_looks_ion and number_azimuth_looks_ion:
        az_looks = float(getattr(inps, 'azLooks', 1))
        rng_looks = float(getattr(inps, 'rngLooks', 1))
        total_az = int(az_looks * number_azimuth_looks_ion)
        total_rg = int(rng_looks * number_range_looks_ion)
        ml2 = '_{}rlks_{}alks'.format(total_rg, total_az)

    seen = set()
    for start in search_roots:
        cur = start
        for _ in range(6):
            if cur in seen:
                break
            seen.add(cur)
            geom_dir = os.path.join(cur, 'geom_reference')
            if ml2:
                cand_ml = os.path.join(geom_dir, 'waterMask' + ml2 + '.rdr')
                if os.path.exists(cand_ml + '.xml'):
                    return cand_ml
            cand = os.path.join(geom_dir, 'waterMask.rdr')
            if os.path.exists(cand + '.xml'):
                return cand
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    if lowBandIgram:
        ifg_dirname = os.path.dirname(lowBandIgram)
        if ml2:
            wbd_file = os.path.join(ifg_dirname, 'wbd' + ml2 + '.wbd')
            if os.path.exists(wbd_file + '.xml'):
                return wbd_file
        wbd_file = os.path.join(ifg_dirname, 'wbd.wbd')
        if os.path.exists(wbd_file + '.xml'):
            return wbd_file
        parent_dir = os.path.dirname(ifg_dirname)
        wbd_file = os.path.join(parent_dir, 'wbd.wbd')
        if os.path.exists(wbd_file + '.xml'):
            return wbd_file

    return None


def _read_conncomp_2d(unw_path, length, width):
    """Load snaphu/icu connected-component labels for an unwrapped product."""
    cc_path = unw_path + '.conncomp'
    if not os.path.exists(cc_path):
        return None
    cc = np.fromfile(cc_path, dtype=np.uint8)
    if cc.size != length * width:
        logger.warning(
            'Unexpected conncomp size for {}: got {}, expected {}'.format(
                cc_path, cc.size, length * width))
        return None
    return cc.reshape(length, width)


def _compute_unw_valid_mask(lowBandIgram, highBandIgram):
    """
    Valid unwrapped samples for compute/filtering weights (mask.bil).

    Requires phase!=0 and conncomp>0 on both sub-bands when conncomp exists.
    Snaphu often fills conncomp=0 pixels with a constant non-zero phase.
    """
    phase_low, length, width = _read_unw_phase_for_mask(lowBandIgram)
    phase_high, _, _ = _read_unw_phase_for_mask(highBandIgram)
    valid = (phase_low != 0) & (phase_high != 0)

    cc_low = _read_conncomp_2d(lowBandIgram, length, width)
    cc_high = _read_conncomp_2d(highBandIgram, length, width)
    if cc_low is not None and cc_high is not None:
        valid &= (cc_low > 0) & (cc_high > 0)
        logger.info(
            'Compute-mask unw validity: phase!=0 and conncomp>0 on both sub-bands.')
    else:
        logger.info(
            'Compute-mask unw validity: phase!=0 only (conncomp not found).')

    return valid


def _output_unw_valid_mask(lowBandIgram, highBandIgram):
    """Legacy output unw check: phase!=0 on both sub-bands (no conncomp gate)."""
    phase_low, _, _ = _read_unw_phase_for_mask(lowBandIgram)
    phase_high, _, _ = _read_unw_phase_for_mask(highBandIgram)
    return (phase_low != 0) & (phase_high != 0)


def _load_water_mask_bool(inps, lowBandIgram, length, width):
    """Return boolean water-body array resampled to (length, width), or None if unavailable."""
    wbd_file = _resolve_water_mask_path(inps, lowBandIgram=lowBandIgram)
    if not wbd_file:
        return None
    wbd, wbd_length, wbd_width = _load_mask_raster_2d(wbd_file)
    water = _water_pixels_from_raster(wbd)
    if (wbd_length, wbd_width) != (length, width):
        logger.info(
            'Resampling water mask from {}x{} to {}x{}'.format(
                wbd_length, wbd_width, length, width))
        water = _resample_valid_mask(water, length, width)
    logger.info('Water body mask from {}: {} pixels ({:.1f}%)'.format(
        wbd_file, int(np.sum(water)), 100.0 * np.sum(water) / water.size if water.size else 0.0))
    return water


def _build_unw_output_mask(lowBandIgram, highBandIgram):
    """Fallback output mask: sub-band unw phase!=0 on both bands (no water)."""
    img_tmp = isceobj.createImage()
    img_tmp.load(lowBandIgram + '.xml')
    length, width = int(img_tmp.length), int(img_tmp.width)
    valid = _output_unw_valid_mask(lowBandIgram, highBandIgram)
    return valid, length, width


def _output_int_amplitude_threshold(inps):
    return float(getattr(inps, 'outputIntAmplitudeThreshold', 0.0))


def _output_int_relative_amplitude_fraction(inps):
    frac = float(getattr(inps, 'outputIntRelativeAmpFraction', 0.0))
    return frac if frac > 0 else None


def _int_path_from_unw(unw_path):
    """Return collocated sub-band .int path for an unwrapped product, or None."""
    for suffix in ('_snaphu.unw', '_icu.unw', '.unw'):
        if unw_path.endswith(suffix):
            return unw_path[:-len(suffix)] + '.int'
    return None


def _read_int_amp_valid(int_path, amp_threshold=0.0, relative_fraction=None):
    """Return (valid_mask, length, width) from a wrapped .int (amplitude threshold)."""
    img_tmp = isceobj.createImage()
    img_tmp.load(int_path + '.xml')
    length, width = int(img_tmp.length), int(img_tmp.width)
    data = np.memmap(int_path, dtype=np.complex64, mode='r', shape=(length, width))
    amp = np.abs(data).astype(np.float64)
    thr = float(amp_threshold)
    if relative_fraction is not None and relative_fraction > 0:
        positive = amp[amp > 0]
        if positive.size > 0:
            rel_thr = float(np.median(positive)) * relative_fraction
            thr = max(thr, rel_thr)
            logger.info(
                'Output int amp threshold for {}: absolute={:.3g}, relative({:.3g}*median)={:.3g}, '
                'using {:.3g}'.format(
                    os.path.basename(int_path), amp_threshold, relative_fraction, rel_thr, thr))
    valid = (amp > thr) & np.isfinite(amp)
    return valid, length, width


def _resolve_subband_int_valid_mask(lowBandIgram, highBandIgram, amp_threshold=0.0,
                                    relative_fraction=None):
    """Valid where both sub-band .int files have amplitude above threshold."""
    low_int = _int_path_from_unw(lowBandIgram)
    high_int = _int_path_from_unw(highBandIgram)
    if not (low_int and high_int and os.path.exists(low_int + '.xml')
            and os.path.exists(high_int + '.xml')):
        return None, None, None

    valid_low, length, width = _read_int_amp_valid(
        low_int, amp_threshold, relative_fraction=relative_fraction)
    valid_high, length_h, width_h = _read_int_amp_valid(
        high_int, amp_threshold, relative_fraction=relative_fraction)
    if (length, width) != (length_h, width_h):
        logger.warning('Sub-band .int shapes differ; skipping sub-band int for output mask.')
        return None, None, None

    logger.info(
        'Using sub-band .int for output mask: {} and {}'.format(
            os.path.basename(low_int), os.path.basename(high_int)))
    return valid_low & valid_high, length, width


def _build_int_output_mask(inps, lowBandIgram, highBandIgram):
    """
    Output mask: .int amplitude valid area only (no water body mask).

    Prefers collocated sub-band .int (same grid as ion).  Falls back to full-band
    .int with strict downsample (invalid if any full-band pixel in footprint is
    nodata — avoids OR-bleed that kept right-edge no-data visible).
    """
    img_tmp = isceobj.createImage()
    img_tmp.load(lowBandIgram + '.xml')
    length, width = int(img_tmp.length), int(img_tmp.width)

    amp_thr = _output_int_amplitude_threshold(inps)
    rel_frac = _output_int_relative_amplitude_fraction(inps)

    valid, _, _ = _resolve_subband_int_valid_mask(
        lowBandIgram, highBandIgram, amp_threshold=amp_thr, relative_fraction=rel_frac)
    full_valid = None
    full_length = full_width = None

    if valid is None:
        full_valid, full_length, full_width = _resolve_fullband_int_valid_mask(
            inps, amp_threshold=amp_thr, relative_fraction=rel_frac)
        if full_valid is None:
            logger.warning(
                'No sub-band or full-band .int for output mask; falling back to unw phase!=0.')
            return _build_unw_output_mask(lowBandIgram, highBandIgram)
        valid = _resample_mask_to_target(
            full_valid, length, width, strict_downsample=True)
        logger.info(
            'Resampled full-band .int valid mask {}x{} -> {}x{} (strict downsample)'.format(
                full_length, full_width, length, width))
    else:
        if (valid.shape[0], valid.shape[1]) != (length, width):
            logger.warning('Sub-band .int shape != ion grid; ignoring sub-band .int.')
            valid = None
            full_valid, full_length, full_width = _resolve_fullband_int_valid_mask(
                inps, amp_threshold=amp_thr, relative_fraction=rel_frac)
            if full_valid is None:
                return _build_unw_output_mask(lowBandIgram, highBandIgram)
            valid = _resample_mask_to_target(
                full_valid, length, width, strict_downsample=True)
        else:
            # Intersect with full-band when available (tightens overlap-edge mask).
            full_valid, full_length, full_width = _resolve_fullband_int_valid_mask(
                inps, amp_threshold=amp_thr, relative_fraction=rel_frac)
            if full_valid is not None:
                full_on_sub = _resample_mask_to_target(
                    full_valid, length, width, strict_downsample=True)
                n_before = int(np.sum(valid))
                valid &= full_on_sub
                logger.info(
                    'Intersected sub-band .int mask with full-band .int: {} -> {} valid pixels'.format(
                        n_before, int(np.sum(valid))))

    n_valid = int(np.sum(valid))
    logger.info(
        'Output mask (.int valid only): {} valid / {} pixels ({:.1f}%)'.format(
            n_valid, valid.size, 100.0 * n_valid / valid.size if valid.size else 0.0))
    return valid, length, width


def _finalize_compute_mask_with_unw_and_water(inps, mask_file, lowBandIgram, highBandIgram):
    """
    Intersect dispersive_filter mask with sub-band unw!=0 and water body exclusion.
    """
    img_mask = isceobj.createImage()
    img_mask.load(mask_file + '.xml')
    length, width = int(img_mask.length), int(img_mask.width)
    mask = np.fromfile(mask_file, dtype=np.byte).reshape(length, width).astype(bool)
    n_type = int(np.sum(mask))

    unw_valid = _compute_unw_valid_mask(lowBandIgram, highBandIgram)
    n_after_unw = int(np.sum(mask & unw_valid))
    mask &= unw_valid

    water = _load_water_mask_bool(inps, lowBandIgram, length, width)
    n_final = n_after_unw
    if water is not None:
        n_final = int(np.sum(mask & ~water))
        mask &= ~water

    logger.info(
        'Compute mask (type+unw+water): type-valid={}, after unw={}, final={} ({:.1f}%)'.format(
            n_type, n_after_unw, n_final, 100.0 * n_final / mask.size if mask.size else 0.0))

    mask.astype(np.byte).tofile(mask_file)


def getMask(inps, maskFile, lowBandIgram=None, highBandIgram=None):
    '''
    Generate compute mask for ion filtering.

    Final mask = dispersive_filter_mask_type AND compute_unw_valid AND NOT water.

    compute_unw_valid uses conncomp>0 when available (see _compute_unw_valid_mask).
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

    _finalize_compute_mask_with_unw_and_water(inps, maskFile, lowBandIgram, highBandIgram)

    # Verify that compute mask file was created
    if not os.path.exists(maskFile):
        raise RuntimeError('Mask file was not created: {}'.format(maskFile))
    if not os.path.exists(maskFile + '.xml'):
        raise RuntimeError('Mask file XML was not created: {}'.format(maskFile + '.xml'))


def _normalize_output_mask_type(mask_type):
    """Map legacy ion_output_mask_type values to int_valid behavior."""
    if mask_type in ('water_and_int', 'water_and_unw'):
        if mask_type == 'water_and_int':
            logger.info(
                'ion_output_mask_type=water_and_int is an alias for int_valid '
                '(output_mask uses .int amp>0 only; water stays in mask.bil).')
        elif mask_type == 'water_and_unw':
            logger.warning(
                'ion_output_mask_type=water_and_unw is deprecated; using int_valid (.int amp>0 only).')
        return 'int_valid'
    return mask_type


def getOutputMask(inps, outputMaskFile, lowBandIgram=None, highBandIgram=None):
    '''
    Write output_mask.bil according to ion_output_mask_type.

    int_valid (default): sub-band .int amp>0 intersect full-band .int (strict downsample).
    No water mask — water exclusion stays in mask.bil for filtering only.
    '''
    if lowBandIgram is None:
        lowBandIgram = inps.lowBandIgram
    if highBandIgram is None:
        highBandIgram = inps.highBandIgram

    mask_type = _normalize_output_mask_type(
        getattr(inps, 'ionOutputMaskType', 'int_valid'))
    if mask_type != 'int_valid':
        raise RuntimeError(
            'getOutputMask called with ion_output_mask_type={!r}; expected int_valid'.format(
                getattr(inps, 'ionOutputMaskType', 'int_valid')))

    valid, length, width = _build_int_output_mask(inps, lowBandIgram, highBandIgram)
    label = 'int amp>0'

    n_valid = int(np.sum(valid))
    logger.info(
        'Output mask ({}): {} valid / {} pixels ({:.1f}%)'.format(
            label, n_valid, valid.size, 100.0 * n_valid / valid.size if valid.size else 0.0))

    valid.astype(np.byte).tofile(outputMaskFile)
    write_xml(outputMaskFile, width, length, 1, 'BYTE', 'BIL')


def _apply_filtering_masks(ion, std, compute_mask, output_mask):
    """
    Apply separate compute vs output masks before ion filtering.

    - output_mask==0: zero ion and std (true no-data for output)
    - compute_mask==0: zero std only so pixel may receive filtered values but not contribute
    """
    ion[output_mask == 0] = 0
    std[output_mask == 0] = 0
    std[compute_mask == 0] = 0


def _load_compute_and_output_masks(mask_file, output_mask_file, length, width, use_split):
    compute_mask = np.fromfile(mask_file, dtype=np.byte).reshape(length, width)
    if use_split:
        output_mask = np.fromfile(output_mask_file, dtype=np.byte).reshape(length, width)
    else:
        output_mask = compute_mask
    return compute_mask, output_mask


def _prepare_phase_arrays_for_filtering(phase, std, compute_mask, output_mask, use_split,
                                        cor_low, cor_high):
    if use_split:
        _apply_filtering_masks(phase, std, compute_mask, output_mask)
        apply_alos_style_dual_band_invalid(phase, std, cor_low, cor_high, zero_data=False)
    else:
        phase[compute_mask == 0] = 0
        std[compute_mask == 0] = 0
        apply_alos_style_dual_band_invalid(phase, std, cor_low, cor_high, zero_data=True)


def _clip_filtered_output(phase_final, compute_mask, output_mask, use_split):
    if use_split:
        phase_final[output_mask == 0] = 0.0
    else:
        phase_final[compute_mask == 0] = 0.0


def _resolve_output_mask_file(inps, out_dir):
    """Return path to the byte mask file used for final ion output masking."""
    mask_type = _normalize_output_mask_type(
        getattr(inps, 'ionOutputMaskType', 'int_valid'))
    if mask_type in ('none', 'full_band_igram'):
        return None
    if mask_type == 'compute_mask':
        return os.path.join(out_dir, 'mask.bil')
    return os.path.join(out_dir, 'output_mask.bil')


def apply_ion_output_mask_to_outputs(inps, output_files, out_dir):
    """Apply the configured final output mask to ion products."""
    mask_type = _normalize_output_mask_type(
        getattr(inps, 'ionOutputMaskType', 'int_valid'))
    if mask_type == 'none':
        logger.info('ion_output_mask_type=none: skipping final output mask.')
        return
    if mask_type == 'full_band_igram':
        apply_fullband_nodata_mask_to_outputs(inps, output_files)
        return

    mask_file = _resolve_output_mask_file(inps, out_dir)
    if not mask_file or not os.path.exists(mask_file + '.xml'):
        logger.warning('Output mask file not found ({}); skipping final output mask.'.format(mask_file))
        return

    img_mask = isceobj.createImage()
    img_mask.load(mask_file + '.xml')
    mask_length, mask_width = int(img_mask.length), int(img_mask.width)
    valid = np.fromfile(mask_file, dtype=np.byte).reshape(mask_length, mask_width) != 0
    n_invalid = int(np.sum(~valid))
    logger.info(
        'Applying ion output mask from {} (type={}): {} invalid / {} pixels ({:.1f}%)'.format(
            mask_file, mask_type, n_invalid, valid.size,
            100.0 * n_invalid / valid.size if valid.size else 0.0))

    for out_path in output_files:
        if not os.path.exists(out_path):
            logger.warning('Output file not found, skipping output mask: {}'.format(out_path))
            continue
        img_out = isceobj.createImage()
        img_out.load(out_path + '.xml')
        out_length, out_width = int(img_out.length), int(img_out.width)
        out_data = np.fromfile(out_path, dtype=np.float32).reshape(out_length, out_width)

        mask_on_out = _resample_valid_mask(valid, out_length, out_width)
        if (out_length, out_width) != (mask_length, mask_width):
            logger.info('Resampled output mask from {}x{} to {}x{} for {}'.format(
                mask_length, mask_width, out_length, out_width, os.path.basename(out_path)))

        out_data[~mask_on_out] = 0.0
        out_data.astype(np.float32).tofile(out_path)
        logger.info('Applied ion output mask to: {}'.format(out_path))


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


def _read_unw_phase_for_mask(path):
    """Return (phase_array, length, width) from an unwrapped interferogram."""
    img_tmp = isceobj.createImage()
    img_tmp.load(path + '.xml')
    length, width = img_tmp.length, img_tmp.width
    data = np.fromfile(path, dtype=np.float32)
    n_single = length * width
    n_bil = length * 2 * width
    if data.size == n_single:
        return data.reshape(length, width), length, width
    if data.size == n_bil:
        arr = data.reshape(length * 2, width)
        return arr[1:length * 2:2, :], length, width
    raise ValueError(
        'Unexpected binary size for {}: {} floats (expected {} or {})'.format(
            path, data.size, n_single, n_bil))


def _read_int_nodata_mask(int_path, amp_threshold=0.0, relative_fraction=None):
    """Return (valid_mask, length, width) from a wrapped interferogram (.int)."""
    return _read_int_amp_valid(
        int_path, amp_threshold=amp_threshold, relative_fraction=relative_fraction)


def _resolve_fullband_int_valid_mask(inps, amp_threshold=0.0, relative_fraction=None):
    """Build valid-pixel mask from full-band .int, or (None, None, None)."""
    if getattr(inps, 'fullBandIgram', None) and os.path.exists(inps.fullBandIgram + '.xml'):
        logger.info('Using configured full-band interferogram for int mask: {}'.format(
            inps.fullBandIgram))
        return _read_int_amp_valid(
            inps.fullBandIgram, amp_threshold, relative_fraction=relative_fraction)

    pair_dir, pair_name = _find_fullband_pair_dir(inps.lowBandIgram, inps.outDir)
    int_path = _find_fullband_int(pair_dir, pair_name)
    if int_path is not None:
        logger.info('Using full-band interferogram for int mask: {}'.format(int_path))
        return _read_int_amp_valid(int_path, amp_threshold, relative_fraction=relative_fraction)
    return None, None, None


def _resolve_fullband_nodata_mask_with_threshold(inps, amp_threshold=0.0, relative_fraction=None):
    """Build a boolean valid-pixel mask from the original full-band interferogram."""
    valid, length, width = _resolve_fullband_int_valid_mask(
        inps, amp_threshold=amp_threshold, relative_fraction=relative_fraction)
    if valid is None:
        pair_dir, pair_name = _find_fullband_pair_dir(inps.lowBandIgram, inps.outDir)
        unw_method = getattr(inps, 'lowBandIgramUnwMethod', 'snaphu')
        unw_path = _find_fullband_unw(pair_dir, pair_name, unw_method=unw_method)
        if unw_path is not None:
            logger.warning(
                'Full-band .int not found in {}; falling back to full-band unw nodata mask: {}'.format(
                    pair_dir, unw_path))
            phase, length, width = _read_unw_phase_for_mask(unw_path)
            return (phase != 0), length, width
        logger.warning(
            'Full-band interferogram not found in {}; skipping full-band nodata mask.'.format(pair_dir))
        return None, None, None
    return valid, length, width


def _find_fullband_pair_dir(low_igram_path, out_dir):
    """Return (igrams_pair_dir, pair_name) for the full-band interferogram directory."""
    low_dir = os.path.dirname(os.path.abspath(low_igram_path))
    pair_name = os.path.basename(low_dir)
    parent = os.path.dirname(low_dir)
    parent_name = os.path.basename(parent)

    if parent_name in ('LowBand', 'HighBand', 'lowBand', 'highBand'):
        igrams_root = os.path.dirname(parent)
    elif parent_name == 'Igrams':
        igrams_root = parent
    else:
        pair_name = os.path.basename(os.path.abspath(out_dir))
        igrams_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out_dir))), 'Igrams')

    return os.path.join(igrams_root, pair_name), pair_name


def _find_fullband_int(pair_dir, pair_name):
    """Locate the full-band wrapped interferogram (.int) in Igrams/{pair}/."""
    import glob

    preferred = [
        os.path.join(pair_dir, 'filt_{}.int'.format(pair_name)),
        os.path.join(pair_dir, '{}.int'.format(pair_name)),
    ]
    for path in preferred:
        if os.path.exists(path + '.xml'):
            return path

    for pattern in ('filt_*.int', '*.int'):
        for path in sorted(glob.glob(os.path.join(pair_dir, pattern))):
            if os.path.exists(path + '.xml'):
                return path
    return None


def _find_fullband_unw(pair_dir, pair_name, unw_method='snaphu'):
    """Locate the full-band unwrapped interferogram (.unw) in Igrams/{pair}/."""
    import glob

    preferred = [
        os.path.join(pair_dir, 'filt_{}_{}.unw'.format(pair_name, unw_method)),
        os.path.join(pair_dir, 'filt_{}_snaphu.unw'.format(pair_name)),
        os.path.join(pair_dir, '{}_{}.unw'.format(pair_name, unw_method)),
    ]
    for path in preferred:
        if os.path.exists(path + '.xml'):
            return path

    for path in sorted(glob.glob(os.path.join(pair_dir, '*.unw'))):
        if os.path.exists(path + '.xml'):
            return path
    return None


def _resolve_fullband_nodata_mask(inps):
    """Build a boolean valid-pixel mask from the original full-band interferogram.

    Returns (valid_mask, length, width) or (None, None, None) if unavailable.
    """
    return _resolve_fullband_nodata_mask_with_threshold(
        inps,
        amp_threshold=_output_int_amplitude_threshold(inps),
        relative_fraction=_output_int_relative_amplitude_fraction(inps))


def _resample_valid_mask_lenient(valid_mask, target_length, target_width):
    """Downsample: target pixel valid if ANY source pixel in its footprint is valid."""
    src_length, src_width = valid_mask.shape
    if (src_length, src_width) == (target_length, target_width):
        return valid_mask.astype(bool)

    if target_length >= src_length and target_width >= src_width:
        zoom_r = target_length / src_length
        zoom_c = target_width / src_width
        return ndimage.zoom(valid_mask.astype(np.float32), (zoom_r, zoom_c), order=0) > 0.5

    out = np.zeros((target_length, target_width), dtype=bool)
    for i in range(target_length):
        r0 = int(round(i * src_length / target_length))
        r1 = int(round((i + 1) * src_length / target_length))
        r1 = max(r1, r0 + 1)
        for j in range(target_width):
            c0 = int(round(j * src_width / target_width))
            c1 = int(round((j + 1) * src_width / target_width))
            c1 = max(c1, c0 + 1)
            out[i, j] = np.any(valid_mask[r0:r1, c0:c1])
    return out


def _resample_valid_mask_strict(valid_mask, target_length, target_width):
    """Downsample: target pixel valid only if ALL source pixels in footprint are valid."""
    src_length, src_width = valid_mask.shape
    if (src_length, src_width) == (target_length, target_width):
        return valid_mask.astype(bool)

    if target_length >= src_length and target_width >= src_width:
        zoom_r = target_length / src_length
        zoom_c = target_width / src_width
        return ndimage.zoom(valid_mask.astype(np.float32), (zoom_r, zoom_c), order=0) > 0.5

    out = np.zeros((target_length, target_width), dtype=bool)
    for i in range(target_length):
        r0 = int(round(i * src_length / target_length))
        r1 = int(round((i + 1) * src_length / target_length))
        r1 = max(r1, r0 + 1)
        for j in range(target_width):
            c0 = int(round(j * src_width / target_width))
            c1 = int(round((j + 1) * src_width / target_width))
            c1 = max(c1, c0 + 1)
            out[i, j] = np.all(valid_mask[r0:r1, c0:c1])
    return out


def _resample_valid_mask(valid_mask, target_length, target_width):
    """Lenient downsample (legacy helper for water mask upsample paths)."""
    return _resample_valid_mask_lenient(valid_mask, target_length, target_width)


def _resample_mask_to_target(valid_mask, target_length, target_width, strict_downsample=True):
    if strict_downsample:
        return _resample_valid_mask_strict(valid_mask, target_length, target_width)
    return _resample_valid_mask_lenient(valid_mask, target_length, target_width)


def apply_fullband_nodata_mask_to_outputs(inps, output_files):
    """Zero ion output pixels where the original full-band interferogram has no data."""
    valid_mask, mask_length, mask_width = _resolve_fullband_nodata_mask(inps)
    if valid_mask is None:
        return

    n_invalid = int(np.sum(~valid_mask))
    logger.info('Output nodata mask from full-band interferogram: {} invalid / {} pixels ({:.1f}%)'.format(
        n_invalid, valid_mask.size, 100.0 * n_invalid / valid_mask.size))

    for out_path in output_files:
        if not os.path.exists(out_path):
            logger.warning('Output file not found, skipping nodata mask: {}'.format(out_path))
            continue
        img_out = isceobj.createImage()
        img_out.load(out_path + '.xml')
        out_length, out_width = img_out.length, img_out.width
        out_data = np.fromfile(out_path, dtype=np.float32).reshape(out_length, out_width)

        mask_on_out = _resample_mask_to_target(
            valid_mask, out_length, out_width, strict_downsample=True)
        if (out_length, out_width) != (mask_length, mask_width):
            logger.info('Resampled full-band nodata mask from {}x{} to {}x{} for {}'.format(
                mask_length, mask_width, out_length, out_width, os.path.basename(out_path)))

        out_data[~mask_on_out] = 0.0
        out_data.astype(np.float32).tofile(out_path)
        logger.info('Applied full-band nodata mask to: {}'.format(out_path))


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
    outputMaskFile = os.path.join(inps.outDir, "output_mask.bil")
    use_split_filter_masks = _normalize_output_mask_type(
        getattr(inps, 'ionOutputMaskType', 'int_valid')) == 'int_valid'

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
    if use_split_filter_masks:
        getOutputMask(inps, outputMaskFile, lowBandIgram=lowBandIgramForIono, highBandIgram=highBandIgramForIono)
        logger.info(
            'Split masks: mask.bil = filter_type + compute_unw(conncomp) + water; '
            'output_mask.bil = .int amp>0 only, no water (ion_output_mask_type={}).'.format(
                getattr(inps, 'ionOutputMaskType', 'int_valid')))

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

        size_max = getattr(inps, 'filteringWinsizeMaxIon', 501)
        size_min = getattr(inps, 'filteringWinsizeMinIon', 51)
        size_secondary = getattr(inps, 'filteringWinsizeSecondaryIon', 5)
        fitAdaptive = getattr(inps, 'fitAdaptiveIon', True)
        filtSecondary = getattr(inps, 'filtSecondaryIon', True)
        fitIon = getattr(inps, 'fitIon', True)
        filtIon = getattr(inps, 'filtIon', True)
        corThresholdFit = getattr(inps, 'fitIonCoherenceThreshold', 0.25)
        std_out0 = getattr(inps, 'filterStdIon', None)
        if std_out0 is None:
            std_out0 = 0.005

        if (not fitIon) and (not filtIon):
            raise Exception('either fit_ion or filt_ion should be True when doing ionospheric correction')

        if size_min > size_max:
            size_max = size_min
        if size_secondary % 2 != 1:
            size_secondary += 1
            logger.info('Window size of secondary filtering should be odd, changed to {}'.format(size_secondary))
        
        # Read data and std - need to get dimensions first
        img = isceobj.createImage()
        img.load(outDispersive + '.xml')
        width = img.width
        length = img.length
        
        ionos = np.fromfile(outDispersive, dtype=np.float32).reshape(length, width)
        std = np.fromfile(sigmaDispersive, dtype=np.float32).reshape(length, width)
        compute_mask, output_mask = _load_compute_and_output_masks(
            maskFile, outputMaskFile, length, width, use_split_filter_masks)

        cor_low_ion = read_coherence_2d(inps.lowBandCoherence, length, width)
        cor_high_ion = read_coherence_2d(inps.highBandCoherence, length, width)
        g2d = None
        _prepare_phase_arrays_for_filtering(
            ionos, std, compute_mask, output_mask, use_split_filter_masks,
            cor_low_ion, cor_high_ion)
        
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

        # Re-apply output/compute mask to prevent filter/polynomial spillover
        _clip_filtered_output(ionos_final, compute_mask, output_mask, use_split_filter_masks)

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
        _prepare_phase_arrays_for_filtering(
            nonDisp, std_nonDisp, compute_mask, output_mask, use_split_filter_masks,
            cor_low_ion, cor_high_ion)
        
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

        _clip_filtered_output(nonDisp_final, compute_mask, output_mask, use_split_filter_masks)

        nonDisp_final.astype(np.float32).tofile(outNonDispersive + ".filt")
        write_xml(outNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtNonDisp and std_nonDisp_filt is not None:
            std_nonDisp_filt.astype(np.float32).tofile(sigmaNonDispersive + ".filt")
            write_xml(sigmaNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        
        del ionos, std, compute_mask, output_mask, nonDisp, std_nonDisp
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
        compute_mask, output_mask = _load_compute_and_output_masks(
            maskFile, outputMaskFile, length, width, use_split_filter_masks)
        cor_low_ion = read_coherence_2d(inps.lowBandCoherence, length, width)
        cor_high_ion = read_coherence_2d(inps.highBandCoherence, length, width)
        _prepare_phase_arrays_for_filtering(
            ionos, std, compute_mask, output_mask, use_split_filter_masks,
            cor_low_ion, cor_high_ion)
        
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

        # Re-apply output/compute mask to prevent filter/polynomial spillover
        _clip_filtered_output(ionos_final, compute_mask, output_mask, use_split_filter_masks)

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
        _prepare_phase_arrays_for_filtering(
            nonDisp, std_nonDisp, compute_mask, output_mask, use_split_filter_masks,
            cor_low_ion, cor_high_ion)
        
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

        _clip_filtered_output(nonDisp_final, compute_mask, output_mask, use_split_filter_masks)

        nonDisp_final.astype(np.float32).tofile(outNonDispersive + ".filt")
        write_xml(outNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtNonDisp and std_nonDisp_filt is not None:
            std_nonDisp_filt.astype(np.float32).tofile(sigmaNonDispersive + ".filt")
            write_xml(sigmaNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        
        del ionos, std, compute_mask, output_mask, nonDisp, std_nonDisp
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
    

    # Final step: mask ion OUTPUT files only, using no-data regions from the
    # original full-band wrapped interferogram (.int amplitude == 0).
    # This does not change masks used during ion estimation / filtering.
    apply_ion_output_mask_to_outputs(
        inps,
        [outDispersive + '.filt', outNonDispersive + '.filt'],
        inps.outDir)


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

