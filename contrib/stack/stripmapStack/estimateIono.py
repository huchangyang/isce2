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

    parser.add_argument('--dispersive_filter_mask_type', dest='dispersive_filter_mask_type', type=str, default='connected_components',
            help='mask type for iterative low-pass filtering: connected_components or coherence')

    parser.add_argument('--dispersive_filter_coherence_threshold', dest='dispersive_filter_coherence_threshold', type=float, default=0.5,
            help='coherence threshold when mask type for iterative low-pass filtering is coherence')

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
    parser.add_argument('--filtering_winsize_max_ion', dest='filteringWinsizeMaxIon', type=int, default=301,
            help='maximum window size for adaptive Gaussian filtering (default=301)')
    parser.add_argument('--filtering_winsize_min_ion', dest='filteringWinsizeMinIon', type=int, default=31,
            help='minimum window size for adaptive Gaussian filtering (default=31)')
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

def ion_std(fl, fu, numberOfLooks, cor):
    '''
    Compute standard deviation of ionospheric phase (same as Alos2Proc/StripmapProc)
    
    fl:  lower band center frequency
    fu:  upper band center frequency
    numberOfLooks: number of looks
    cor: coherence, must be numpy array
    
    Returns:
    std: standard deviation of ionospheric phase
    '''
    f0 = (fl + fu) / 2.0
    interferogramVar = (1.0 - cor**2) / (2.0 * numberOfLooks * cor**2 + (cor==0))
    std = fl*fu/f0/(fu**2-fl**2)*np.sqrt(fu**2*interferogramVar+fl**2*interferogramVar)
    std[np.nonzero(cor==0)] = 0
    return std

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
    # Save original valid pixel mask to distinguish truly masked pixels from boundary effects
    original_valid_mask = (data != 0) * (std != 0)
    
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
        # Calculate effective weight coverage
        # Use a lower threshold for boundary pixels to avoid marking valid pixels as invalid
        # due to boundary effects (zero-padding in fftconvolve)
        weight_coverage = ss.fftconvolve(wgt!=0, gaussian_filters[i]!=0, mode='same')
        # For boundary pixels (where original data is valid), use a more lenient threshold
        # This prevents valid boundary pixels from being incorrectly marked as invalid
        # Use adaptive threshold: lower for originally valid pixels, especially near boundaries
        size_half = int((size - 1) / 2)
        # Create boundary mask: pixels within size_half of image edges
        boundary_mask = np.zeros((length, width), dtype=bool)
        boundary_mask[:size_half, :] = True  # Top edge
        boundary_mask[-size_half:, :] = True  # Bottom edge
        boundary_mask[:, :size_half] = True  # Left edge
        boundary_mask[:, -size_half:] = True  # Right edge
        # Use even lower threshold for valid pixels near boundaries
        threshold = np.where(original_valid_mask & boundary_mask, 0.2, 
                           np.where(original_valid_mask, 0.3, 0.5))
        index = np.nonzero(weight_coverage < threshold)
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

def adjust_phase_polynomial(lowBandIgram, highBandIgram, outputDir, lowBandCoherence=None, highBandCoherence=None):
    '''
    Adjust phase using polynomial fitting (similar to ALOS processing)
    This function adjusts the upper band phase to remove relative phase unwrapping errors
    using polynomial fitting, similar to computeIonosphere in runIonFilt.py
    
    Returns: adjusted high band interferogram file path
    '''
    logger.info('Adjusting phase using polynomial fitting (ALOS-style)')
    
    # Read unwrapped interferograms
    img_low = isceobj.createImage()
    img_low.load(lowBandIgram + '.xml')
    width = img_low.width
    length = img_low.length
    
    # Read amplitude (band 0) and phase (band 1) data
    # Note: Invalid regions (originally 0 in wrapped phase) may be unwrapped to 0, ±2π, etc.
    # So we use amplitude to identify valid pixels (amplitude > 0)
    lowerAmp = np.fromfile(lowBandIgram, dtype=np.float32).reshape(length*2, width)[0:length*2:2, :]
    lowerUnw = np.fromfile(lowBandIgram, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
    upperAmp = np.fromfile(highBandIgram, dtype=np.float32).reshape(length*2, width)[0:length*2:2, :]
    upperUnw = np.fromfile(highBandIgram, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
    
    # Create valid pixel mask based on amplitude (more reliable than phase)
    # Invalid regions typically have zero or very small amplitude
    valid_mask = (lowerAmp > 0) * (upperAmp > 0)
    
    # Prepare weight using coherence if available
    if lowBandCoherence and highBandCoherence and os.path.exists(lowBandCoherence + '.xml') and os.path.exists(highBandCoherence + '.xml'):
        try:
            # Check coherence file format (single band or BIL format with 2 bands)
            img_cor_low = isceobj.createImage()
            img_cor_low.load(lowBandCoherence + '.xml')
            cor_low_bands = img_cor_low.bands
            
            img_cor_high = isceobj.createImage()
            img_cor_high.load(highBandCoherence + '.xml')
            cor_high_bands = img_cor_high.bands
            
            # Read coherence data based on format
            if cor_low_bands == 2:  # BIL format (amplitude + coherence)
                cor_low = np.fromfile(lowBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
            else:  # Single band format
                cor_low = np.fromfile(lowBandCoherence, dtype=np.float32).reshape(length, width)
            
            if cor_high_bands == 2:  # BIL format (amplitude + coherence)
                cor_high = np.fromfile(highBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
            else:  # Single band format
                cor_high = np.fromfile(highBandCoherence, dtype=np.float32).reshape(length, width)
            
            # Check dimensions match
            if cor_low.shape != (length, width) or cor_high.shape != (length, width):
                raise ValueError('Coherence dimensions ({}, {}) do not match interferogram dimensions ({}, {})'.format(
                    cor_low.shape, cor_high.shape, (length, width)))
            
            # Use average coherence as weight, with high power (similar to ALOS corOrderAdj=20)
            cor = (cor_low + cor_high) / 2.0
            cor[np.nonzero(cor<0)] = 0.0
            cor[np.nonzero(cor>1)] = 0.0
            wgt = cor**20  # Similar to corOrderAdj=20 in ALOS
            wgt[np.nonzero(~valid_mask)] = 0  # Mask out invalid pixels using amplitude
        except Exception as e:
            logger.warning('Could not read coherence files for phase adjustment: {}. Using binary mask.'.format(e))
            # Use binary mask if coherence reading fails
            wgt = np.ones((length, width), dtype=np.float32)
            wgt[np.nonzero(~valid_mask)] = 0  # Mask out invalid pixels using amplitude
    else:
        # Use binary mask if coherence not available
        wgt = np.ones((length, width), dtype=np.float32)
        wgt[np.nonzero(~valid_mask)] = 0  # Mask out invalid pixels using amplitude
    
    # Compute phase difference
    phase_diff = lowerUnw - upperUnw
    
    # Fit polynomial surface to phase difference (order 2, similar to ALOS)
    diff_fit, coeff = polyfit_2d(phase_diff, wgt, 2)
    
    # Adjust upper band phase (use valid_mask based on amplitude, not phase)
    flag2 = valid_mask
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
    
    # Read original file structure (amplitude + phase)
    # BIL format: data is stored as (length*2, width) where rows are interleaved
    # Row 0, 2, 4, ... = band 0 (amplitude)
    # Row 1, 3, 5, ... = band 1 (phase)
    original_data = np.fromfile(highBandIgram, dtype=np.float32).reshape(length*2, width)
    original_data[1:length*2:2, :] = upperUnw_adjusted
    
    # Save adjusted file
    original_data.astype(np.float32).tofile(highBandIgramAdjusted)
    
    # Write XML: for BIL format with 2 bands, length should be the original image length (not length*2)
    # The XML length parameter refers to the number of image rows, not the number of data rows in the file
    write_xml(highBandIgramAdjusted, width, length, 2, "FLOAT", "BIL")
    
    logger.info('Adjusted high band interferogram saved to: {}'.format(highBandIgramAdjusted))
    
    return highBandIgramAdjusted


def check_consistency(lowBandIgram, highBandIgram, outputDir):
    """
    Check consistency between low and high band unwrapped interferograms
    by computing the number of 2π jumps between them.
    
    Returns the path to the jumps file (single band float format).
    """
    jumpFile = os.path.join(outputDir, "jumps.bil")
    
    # Get image dimensions from low band interferogram
    img_low = isceobj.createImage()
    img_low.load(lowBandIgram + '.xml')
    width = img_low.width
    length = img_low.length
    
    # Read phase data from both interferograms
    # Both are BIL format: band 0 = amplitude, band 1 = phase
    low_data = np.fromfile(lowBandIgram, dtype=np.float32).reshape(length*2, width)
    low_phase = low_data[1:length*2:2, :]
    
    high_data = np.fromfile(highBandIgram, dtype=np.float32).reshape(length*2, width)
    high_phase = high_data[1:length*2:2, :]
    
    # Compute jumps: round((phase_low - phase_high) / (2π))
    phase_diff = low_phase - high_phase
    jumps = np.round(phase_diff / (2.0 * np.pi))
    
    # Save as BIL format (2 bands) for compatibility with imageMath.py
    # Band 0: jumps (same as band 1)
    # Band 1: jumps
    # This ensures imageMath.py can read it correctly when used with --c option
    # BIL format: data is stored as (length*2, width) where rows are interleaved
    jumps_bil = np.zeros((length*2, width), dtype=np.float32)
    jumps_bil[0:length*2:2, :] = jumps  # Band 0
    jumps_bil[1:length*2:2, :] = jumps  # Band 1
    
    jumps_bil.astype(np.float32).tofile(jumpFile)
    # Write XML: for BIL format with 2 bands, length should be the original image length (not length*2)
    write_xml(jumpFile, width, length, 2, "FLOAT", "BIL")
    
    logger.info('Jumps file saved to: {}'.format(jumpFile))
    logger.info('  Jumps range: [{:.0f}, {:.0f}]'.format(np.nanmin(jumps), np.nanmax(jumps)))
    logger.info('  Non-zero jumps: {} / {} pixels ({:.2f}%)'.format(
        np.sum(jumps != 0), jumps.size, 100.0 * np.sum(jumps != 0) / jumps.size))
    
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
        # Use amplitude (a_0, b_0) to mask out invalid pixels in the calculation
        # This ensures that invalid regions (originally 0 in wrapped phase, unwrapped to 0, ±2π, etc.)
        # do not participate in the ionosphere estimation
        # Note: c_0 and g_0 are used to access band 0 of jumps file (BIL format, 2 bands)
        cmd = 'imageMath.py -e="((a_0>0)*(b_0>0))*({0}*((a_1-2*PI*c_0)*{1}-(b_1+(2.0*PI*g_0)-2*PI*(c_0+f))*{2}))" --a={3} --b={4} --c={5} --f={6} --g={7} -o {8} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, m , d, jumpFile, outDispersive)
        print(cmd)
        os.system(cmd)

        coefn = f0/(fH**2-fL**2)
        # Use amplitude to mask out invalid pixels
        # Note: c_0 and g_0 are used to access band 0 of jumps file (BIL format, 2 bands)
        cmd = 'imageMath.py -e="((a_0>0)*(b_0>0))*({0}*((a_1+(2.0*PI*g_0)-2*PI*c_0)*{1}-(b_1-2*PI*(c_0+f))*{2}))" --a={3} --b={4} --c={5} --f={6} --g={7} -o {8} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, m , d, jumpFile, outNonDispersive)
        print(cmd)
        os.system(cmd)

    else:
        
        coef = (fL*fH)/(f0*(fH**2 - fL**2))
        # Use amplitude (a_0, b_0) to mask out invalid pixels in the calculation
        # This ensures that invalid regions do not participate in the ionosphere estimation
        # Note: c_0 is used to access band 0 of jumps file (BIL format, 2 bands)
        cmd = 'imageMath.py -e="((a_0>0)*(b_0>0))*({0}*(a_1*{1}-(b_1+2.0*PI*c_0)*{2}))" --a={3} --b={4} --c={5}  -o {6} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, jumpFile, outDispersive)

        print(cmd)
        os.system(cmd)

        coefn = f0/(fH**2-fL**2)
        # Use amplitude to mask out invalid pixels
        # Note: c_0 is used to access band 0 of jumps file (BIL format, 2 bands)
        cmd = 'imageMath.py -e="((a_0>0)*(b_0>0))*({0}*((a_1+2.0*PI*c_0)*{1}-(b_1)*{2}))" --a={3} --b={4} --c={5} -o {6} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, jumpFile, outNonDispersive)
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
    
    cmd = 'imageMath.py -e="sqrt(1-a**2)/a/sqrt(2.0*{0})" --a={1} -o {2} -t float -s BIL'.format(N, lowBandCoherence, Sig_phi_L)
    print(cmd)
    os.system(cmd)
    #Sig_phi_L = np.sqrt(1-cL**2)/cL/np.sqrt(2.*N)

    #cH = read(inps.highBandCoherence,bands=[1])
    #cH = cH[0,:,:]
    #cH[cH==0.0]=0.001

    cmd = 'imageMath.py -e="sqrt(1-a**2)/a/sqrt(2.0*{0})" --a={1} -o {2} -t float -s BIL'.format(N, highBandCoherence, Sig_phi_H)
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

    if inps.dispersive_filter_mask_type == "coherence":
        print ('generating a mask based on coherence files of sub-band interferograms with a threshold of {0}'.format(inps.dispersive_filter_coherence_threshold))
        cmd = 'imageMath.py -e="(a>{0})*(b>{0})" --a={1} --b={2} -t byte -s BIL -o {3}'.format(inps.dispersive_filter_coherence_threshold, lowBandCor, highBandCor, maskFile)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError('Failed to generate mask file using coherence files. Command: {}'.format(cmd))
    elif (inps.dispersive_filter_mask_type == "connected_components") and ((os.path.exists(lowBandIgram + '.conncomp')) and (os.path.exists(highBandIgram + '.conncomp'))):
       # If connected components from snaphu exists, let's get a mask based on that. 
       # Regions of zero are masked out. Let's assume that islands have been connected. 
        print ('generating a mask based on .conncomp files')
        cmd = 'imageMath.py -e="(a>0)*(b>0)" --a={0} --b={1} -t byte -s BIL -o {2}'.format(lowBandIgram + '.conncomp', highBandIgram + '.conncomp', maskFile)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError('Failed to generate mask file using connected components. Command: {}'.format(cmd))
    else:
        # Use amplitude to identify valid pixels
        # Note: Due to phase wrapping, invalid regions (originally 0 in wrapped phase)
        # may be unwrapped to 0, ±2π, ±4π, etc. However, invalid regions typically
        # have zero or very small amplitude. So we use amplitude (band 0) to identify valid pixels.
        print ('generating a mask based on unwrapped file amplitudes.')
        print ('  Note: Invalid regions (originally 0 in wrapped phase) may be unwrapped to 0, ±2π, etc.')
        print ('  Using amplitude (band 0) to identify valid pixels (amplitude > 0).')
        # Check amplitude (band 0) > 0 for both interferograms
        # This is more reliable than checking phase != 0, since unwrapped phase
        # from invalid regions can be 0, ±2π, ±4π, etc.
        cmd = 'imageMath.py -e="(a_0>0)*(b_0>0)" --a={0} --b={1} -t byte -s BIL -o {2}'.format(lowBandIgram , highBandIgram , maskFile)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError('Failed to generate mask file using unwrapped file amplitudes. Command: {}'.format(cmd))
    
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

    # Note: g_0 is used to access band 0 of jumps file (BIL format, 2 bands)
    cmd = 'imageMath.py -e="round(((a_1+(2.0*PI*g_0)) - (b_1) - (2.0*{0}/3.0/{1})*c + (2.0*{0}/3.0/{1})*f )/2.0/PI)" --a={2} --b={3} --c={4} --f={5} --g={6}  -o {7} -t float32 -s BIL'.format(B, f0, highBandIgram, lowBandIgram, nonDispFile, dispFile, jumpsFile, dFile)

    print(cmd)

    os.system(cmd)
    #d = (phH - phL - (2.*B/3./f0)*ph_nondis + (2.*B/3./f0)*ph_iono )/2./PI
    #d = np.round(d)

    #cmd = 'imageMath.py -e="round(((a_1 - {6}) + (b_1-{7}) - 2.0*c - 2.0*f )/4.0/PI - g/2)" --a={0} --b={1} --c={2} --f={3} --g={4} -o {5} -t float32 -s BIL'.format(lowBandIgram, highBandIgram, nonDispFile, dispFile, dFile, mFile, refL, refH)

    # Note: k_0 is used to access band 0 of jumps file, g_0 is used to access band 0 of dJumps file (both BIL format, 2 bands)
    cmd = 'imageMath.py -e="round(((a_1 ) + (b_1+(2.0*PI*k_0)) - 2.0*c - 2.0*f )/4.0/PI - g_0/2)" --a={0} --b={1} --c={2} --f={3} --g={4} --k={5} -o {6} -t float32 -s BIL'.format(lowBandIgram, highBandIgram, nonDispFile, dispFile, dFile, jumpsFile, mFile)

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


def main(iargs=None):


    inps = cmdLineParse(iargs)

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
    
    # Log frequency information for verification
    logger.info('Band frequency information:')
    logger.info('  Center wavelength (wvl0): {:.6f} m'.format(wvl))
    logger.info('  Lower band wavelength (wvlL): {:.6f} m'.format(wvlL))
    logger.info('  Upper band wavelength (wvlH): {:.6f} m'.format(wvlH))
    logger.info('  Center frequency (f0): {:.2f} Hz ({:.6f} GHz)'.format(f0, f0/1e9))
    logger.info('  Lower band frequency (fL): {:.2f} Hz ({:.6f} GHz)'.format(fL, fL/1e9))
    logger.info('  Upper band frequency (fH): {:.2f} Hz ({:.6f} GHz)'.format(fH, fH/1e9))
    logger.info('  Frequency difference (fH - fL): {:.2f} Hz ({:.6f} GHz)'.format(fH - fL, (fH - fL)/1e9))
    logger.info('  Total bandwidth (B): {:.2e} Hz ({:.6f} GHz)'.format(B, B/1e9))

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
                highBandCoherence=inps.highBandCoherence
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
    jumpsFile = check_consistency(lowBandIgramForIono, highBandIgramForIonoAdjusted, inps.outDir)

    #########################################################
    # generating a mask which will help filtering the estimated dispersive and non-dispersive phase
    # Generate mask BEFORE estimating ionosphere to mask out zero-value pixels at edges
    # Use multilooked interferograms for mask generation if they were used for ionosphere estimation
    getMask(inps, maskFile, lowBandIgram=lowBandIgramForIono, highBandIgram=highBandIgramForIono)
    
    #########################################################
    # estimating the dispersive and non-dispersive components
    # Use adjusted high band interferogram if available
    dispersive_nonDispersive(lowBandIgramForIono, highBandIgramForIonoAdjusted, f0, fL, fH, outDispersive, outNonDispersive, jumpsFile)
    
    # Apply mask to zero out invalid pixels (edges with zero values) in the initial estimates
    # This prevents zero-value pixels from affecting the initial ionosphere estimation
    logger.info('Applying mask to initial ionosphere estimates to mask out zero-value pixels')
    img = isceobj.createImage()
    img.load(outDispersive + '.xml')
    width = img.width
    length = img.length
    
    mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
    
    # Apply mask to dispersive phase
    dispersive = np.fromfile(outDispersive, dtype=np.float32).reshape(length, width)
    dispersive[np.nonzero(mask==0)] = 0.0
    dispersive.astype(np.float32).tofile(outDispersive)
    
    # Apply mask to non-dispersive phase
    nonDispersive = np.fromfile(outNonDispersive, dtype=np.float32).reshape(length, width)
    nonDispersive[np.nonzero(mask==0)] = 0.0
    nonDispersive.astype(np.float32).tofile(outNonDispersive)
    
    # Calculate number of looks and standard deviation (same as StripmapProc/Alos2Proc)
    # For multilooked subband interferograms, total looks = original looks * additional ionospheric looks
    azLooks = getattr(inps, 'azLooks', 1)
    rgLooks = getattr(inps, 'rngLooks', 1)
    totalLooks = azLooks * rgLooks
    if useMultilookedUnw and numberRangeLooksIon and numberAzimuthLooksIon:
        totalLooks = totalLooks * numberRangeLooksIon * numberAzimuthLooksIon
    logger.info('Using number of looks: {:.2f} (azLooks={:.0f}, rgLooks={:.0f}, azLooksIon={:.0f}, rgLooksIon={:.0f})'.format(
        totalLooks, azLooks, rgLooks, 
        numberAzimuthLooksIon if useMultilookedUnw and numberAzimuthLooksIon else 1,
        numberRangeLooksIon if useMultilookedUnw and numberRangeLooksIon else 1))
    
    # Compute standard deviation of ionospheric phase (same as Alos2Proc/StripmapProc)
    # Read coherence files for std calculation
    # Get dimensions from dispersive phase file
    img = isceobj.createImage()
    img.load(outDispersive + '.xml')
    width = img.width
    length = img.length
    
    # Read coherence files
    cor = None
    if inps.lowBandCoherence and inps.highBandCoherence and os.path.exists(inps.lowBandCoherence + '.xml') and os.path.exists(inps.highBandCoherence + '.xml'):
        try:
            # Check coherence file format (single band or BIL format with 2 bands)
            img_cor_low = isceobj.createImage()
            img_cor_low.load(inps.lowBandCoherence + '.xml')
            cor_low_bands = img_cor_low.bands
            
            img_cor_high = isceobj.createImage()
            img_cor_high.load(inps.highBandCoherence + '.xml')
            cor_high_bands = img_cor_high.bands
            
            # Read coherence data based on format
            if cor_low_bands == 2:  # BIL format (amplitude + coherence)
                cor_low = np.fromfile(inps.lowBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
            else:  # Single band format
                cor_low = np.fromfile(inps.lowBandCoherence, dtype=np.float32).reshape(length, width)
            
            if cor_high_bands == 2:  # BIL format (amplitude + coherence)
                cor_high = np.fromfile(inps.highBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
            else:  # Single band format
                cor_high = np.fromfile(inps.highBandCoherence, dtype=np.float32).reshape(length, width)
            
            # Check dimensions match
            if cor_low.shape == (length, width) and cor_high.shape == (length, width):
                cor = (cor_low + cor_high) / 2.0
                cor[np.nonzero(cor<0)] = 0.0
                cor[np.nonzero(cor>1)] = 0.0
                # Apply mask to coherence to ensure invalid regions are excluded
                mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
                cor[np.nonzero(mask==0)] = 0.0
                logger.info('Successfully read coherence files for std calculation')
            else:
                logger.warning('Coherence dimensions ({}, {}) do not match ionosphere dimensions ({}, {})'.format(
                    cor_low.shape, cor_high.shape, (length, width)))
        except Exception as e:
            logger.warning('Could not read coherence files for std calculation: {}'.format(e))
    
    # Compute std using ion_std function (same as Alos2Proc/StripmapProc)
    # Load mask to apply to std calculation
    mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
    
    if cor is not None:
        std_dispersive = ion_std(fL, fH, totalLooks, cor)
        std_nonDispersive = ion_std(fL, fH, totalLooks, cor)  # Same formula for non-dispersive
        # Apply mask to std to ensure invalid regions have zero std
        std_dispersive[np.nonzero(mask==0)] = 0.0
        std_nonDispersive[np.nonzero(mask==0)] = 0.0
        logger.info('Computed standard deviation of ionospheric phase using ion_std function (Alos2Proc/StripmapProc method)')
    else:
        logger.warning('Could not compute std from coherence, using theoretical variance method as fallback')
        theoretical_variance_fromSubBands(inps, f0, fL, fH, B, sigmaDispersive, sigmaNonDispersive, totalLooks)
        # Read the computed std files
        std_dispersive = np.fromfile(sigmaDispersive, dtype=np.float32).reshape(length, width)
        std_nonDispersive = np.fromfile(sigmaNonDispersive, dtype=np.float32).reshape(length, width)
        # Apply mask to std to ensure invalid regions have zero std
        std_dispersive[np.nonzero(mask==0)] = 0.0
        std_nonDispersive[np.nonzero(mask==0)] = 0.0
    
    # Save std files (same format as Alos2Proc/StripmapProc, using .sig extension)
    std_dispersive.astype(np.float32).tofile(sigmaDispersive)
    write_xml(sigmaDispersive, width, length, 1, "FLOAT", "BIL")
    std_nonDispersive.astype(np.float32).tofile(sigmaNonDispersive)
    write_xml(sigmaNonDispersive, width, length, 1, "FLOAT", "BIL")
    logger.info('Saved standard deviation files: {} and {}'.format(sigmaDispersive, sigmaNonDispersive)) 

    # Use adaptive Gaussian filtering if explicitly requested, otherwise use original iterative filtering
    useAdaptiveFilter = getattr(inps, 'useAdaptiveGaussian', True)
    if useAdaptiveFilter:
        # Use adaptive Gaussian filtering (similar to StripmapProc)
        logger.info('Using adaptive Gaussian filtering for ionospheric phase')
        
        # Read data and std - need to get dimensions first
        img = isceobj.createImage()
        img.load(outDispersive + '.xml')
        width = img.width
        length = img.length
        
        ionos = np.fromfile(outDispersive, dtype=np.float32).reshape(length, width)
        std = np.fromfile(sigmaDispersive, dtype=np.float32).reshape(length, width)
        mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
        
        # Apply mask
        ionos[mask==0] = 0
        std[mask==0] = 0
        
        # Get filtering parameters (defaults match StripmapProc/alosStack.xml)
        size_max = getattr(inps, 'filteringWinsizeMaxIon', 301)
        size_min = getattr(inps, 'filteringWinsizeMinIon', 31)
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
        
        # If std_out0 is None, use a reasonable default (matching StripmapProc)
        if std_out0 is None:
            std_out0 = 0.005  # Default for stripmap (matches ALOS-2 SPT/SM1 modes), can be overridden by user
        
        if size_min > size_max:
            size_max = size_min
        if size_secondary % 2 != 1:
            size_secondary += 1
            logger.info('Window size of secondary filtering should be odd, changed to {}'.format(size_secondary))
        
        # Global polynomial fitting (ALOS-style) before filtering
        ionos_fit = None
        if fitIon:
            logger.info('Applying global polynomial fit to ionospheric phase (ALOS-style)')
            # Prepare weight using standard deviation
            wgt = std**2
            wgt[np.nonzero(std==0)] = 0
            
            # Apply coherence threshold if coherence files are available
            if inps.lowBandCoherence and inps.highBandCoherence:
                try:
                    cor_low = np.fromfile(inps.lowBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :] if os.path.exists(inps.lowBandCoherence + '.xml') else None
                    cor_high = np.fromfile(inps.highBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :] if os.path.exists(inps.highBandCoherence + '.xml') else None
                    if cor_low is not None and cor_high is not None:
                        cor = (cor_low + cor_high) / 2.0
                        cor[np.nonzero(cor<0)] = 0.0
                        cor[np.nonzero(cor>1)] = 0.0
                        wgt[np.nonzero(cor<corThresholdFit)] = 0
                except:
                    pass
            
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
                ionos.copy(), std.copy(), size_min, size_max, std_out0, fit=fitAdaptive)
        
            # Apply secondary filtering if requested
            if filtSecondary:
                logger.info('Applying secondary filtering with window size {}'.format(size_secondary))
                import scipy.signal as ss
                # Create Gaussian kernel for secondary filtering
                hsize = (size_secondary - 1) / 2
                x = np.arange(-hsize, hsize + 1)
                f = np.exp(-x**2/(2.0*(size_secondary/2.0)**2)) / ((size_secondary/2.0) * np.sqrt(2.0*np.pi))
                g2d = np.outer(f, f)
                g2d = g2d / np.sum(g2d)
            # Apply secondary filtering
            # Use mask to prevent masked regions from being filled
            mask_filt = (ionos_filt!=0).astype(np.float32)
            scale = ss.fftconvolve(mask_filt, g2d, mode='same')
            ionos_filt_filtered = ss.fftconvolve(ionos_filt, g2d, mode='same') / (scale + (scale==0))
            # Only update pixels that were valid before filtering (preserve mask)
            ionos_filt = mask_filt * ionos_filt_filtered
        
        # Combine fit and filt results (ALOS-style)
        if fitIon and filtIon:
            ionos_final = ionos_filt + ionos_fit * (ionos_filt!=0)
        elif fitIon and not filtIon:
            ionos_final = ionos_fit
        elif not fitIon and filtIon:
            ionos_final = ionos_filt
        else:
            ionos_final = ionos
        
        # Apply mask to final result to ensure masked regions remain zero
        # This is critical because filtering can "leak" values from boundary regions
        mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
        ionos_final[np.nonzero(mask==0)] = 0.0
        
        # Save filtered results
        ionos_final.astype(np.float32).tofile(outDispersive + ".filt")
        write_xml(outDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtIon and std_filt is not None:
            std_filt[np.nonzero(mask==0)] = 0.0  # Also mask std
            std_filt.astype(np.float32).tofile(sigmaDispersive + ".filt")
            write_xml(sigmaDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
            # Save window size file (same as Alos2Proc/StripmapProc)
            if window_size is not None:
                windowSizeDispersive = outDispersive + ".filt.win"
                windowSizeDispersive = os.path.abspath(windowSizeDispersive)
                print('Saving dispersive window size file: {}'.format(windowSizeDispersive))
                logger.info('Saving dispersive window size file: {}'.format(windowSizeDispersive))
                window_size.astype(np.float32).tofile(windowSizeDispersive)
                write_xml(windowSizeDispersive, width, length, 1, "FLOAT", "BIL")
                print('Saved dispersive window size file: {}'.format(windowSizeDispersive))
                logger.info('Saved dispersive window size file: {}'.format(windowSizeDispersive))
            else:
                print('WARNING: Window size is None for dispersive phase!')
                logger.warning('Window size is None for dispersive phase (filtIon={}, std_filt is not None={})'.format(
                    filtIon, std_filt is not None))
        
        # Filter non-dispersive phase
        nonDisp = np.fromfile(outNonDispersive, dtype=np.float32).reshape(length, width)
        std_nonDisp = np.fromfile(sigmaNonDispersive, dtype=np.float32).reshape(length, width)
        nonDisp[mask==0] = 0
        std_nonDisp[mask==0] = 0
        
        # Global polynomial fitting for non-dispersive phase
        nonDisp_fit = None
        if fitIon:
            wgt = std_nonDisp**2
            wgt[np.nonzero(std_nonDisp==0)] = 0
            if inps.lowBandCoherence and inps.highBandCoherence:
                try:
                    cor_low = np.fromfile(inps.lowBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :] if os.path.exists(inps.lowBandCoherence + '.xml') else None
                    cor_high = np.fromfile(inps.highBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :] if os.path.exists(inps.highBandCoherence + '.xml') else None
                    if cor_low is not None and cor_high is not None:
                        cor = (cor_low + cor_high) / 2.0
                        cor[np.nonzero(cor<0)] = 0.0
                        cor[np.nonzero(cor>1)] = 0.0
                        wgt[np.nonzero(cor<corThresholdFit)] = 0
                except:
                    pass
            index = np.nonzero(wgt!=0)
            if len(index[0]) > 0:
                wgt[index] = 1.0/(wgt[index])
                nonDisp_fit, _ = polyfit_2d(nonDisp.copy(), wgt, 2)
                nonDisp = nonDisp - nonDisp_fit * (nonDisp!=0)
        
        nonDisp_filt = None
        std_nonDisp_filt = None
        window_size_nonDisp = None
        if filtIon:
            nonDisp_filt, std_nonDisp_filt, window_size_nonDisp = adaptive_gaussian(
                nonDisp.copy(), std_nonDisp.copy(), size_min, size_max, std_out0, fit=fitAdaptive)
            
            # Apply secondary filtering to non-dispersive phase if requested
            if filtSecondary:
                # Create Gaussian kernel if not already created (from dispersive phase filtering)
                if 'g2d' not in locals() or g2d is None:
                    hsize = (size_secondary - 1) / 2
                    x = np.arange(-hsize, hsize + 1)
                    f = np.exp(-x**2/(2.0*(size_secondary/2.0)**2)) / ((size_secondary/2.0) * np.sqrt(2.0*np.pi))
                    g2d = np.outer(f, f)
                    g2d = g2d / np.sum(g2d)
                # Use mask to prevent masked regions from being filled
                mask_filt = (nonDisp_filt!=0).astype(np.float32)
                scale = ss.fftconvolve(mask_filt, g2d, mode='same')
                nonDisp_filt_filtered = ss.fftconvolve(nonDisp_filt, g2d, mode='same') / (scale + (scale==0))
                # Only update pixels that were valid before filtering (preserve mask)
                nonDisp_filt = mask_filt * nonDisp_filt_filtered
        
        # Combine fit and filt results for non-dispersive phase
        if fitIon and filtIon:
            nonDisp_final = nonDisp_filt + nonDisp_fit * (nonDisp_filt!=0)
        elif fitIon and not filtIon:
            nonDisp_final = nonDisp_fit
        elif not fitIon and filtIon:
            nonDisp_final = nonDisp_filt
        else:
            nonDisp_final = nonDisp
        
        nonDisp_final.astype(np.float32).tofile(outNonDispersive + ".filt")
        write_xml(outNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtIon and std_nonDisp_filt is not None:
            std_nonDisp_filt.astype(np.float32).tofile(sigmaNonDispersive + ".filt")
            write_xml(sigmaNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
            # Save window size file (same as Alos2Proc/StripmapProc)
            if window_size_nonDisp is not None:
                windowSizeNonDispersive = outNonDispersive + ".filt.win"
                windowSizeNonDispersive = os.path.abspath(windowSizeNonDispersive)
                print('Saving non-dispersive window size file: {}'.format(windowSizeNonDispersive))
                logger.info('Saving non-dispersive window size file: {}'.format(windowSizeNonDispersive))
                window_size_nonDisp.astype(np.float32).tofile(windowSizeNonDispersive)
                write_xml(windowSizeNonDispersive, width, length, 1, "FLOAT", "BIL")
                print('Saved non-dispersive window size file: {}'.format(windowSizeNonDispersive))
                logger.info('Saved non-dispersive window size file: {}'.format(windowSizeNonDispersive))
            else:
                print('WARNING: Window size is None for non-dispersive phase!')
                logger.warning('Window size is None for non-dispersive phase (filtIon={}, std_nonDisp_filt is not None={})'.format(
                    filtIon, std_nonDisp_filt is not None))
        
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
    
    # Apply mask to corrected estimates to mask out zero-value pixels
    logger.info('Applying mask to corrected ionosphere estimates to mask out zero-value pixels')
    img = isceobj.createImage()
    img.load(outDispersive + '.xml')
    width = img.width
    length = img.length
    
    mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
    
    # Apply mask to corrected dispersive phase
    dispersive = np.fromfile(outDispersive, dtype=np.float32).reshape(length, width)
    dispersive[np.nonzero(mask==0)] = 0.0
    dispersive.astype(np.float32).tofile(outDispersive)
    
    # Apply mask to corrected non-dispersive phase
    nonDispersive = np.fromfile(outNonDispersive, dtype=np.float32).reshape(length, width)
    nonDispersive[np.nonzero(mask==0)] = 0.0
    nonDispersive.astype(np.float32).tofile(outNonDispersive)

    # Filter the corrected estimates
    if useAdaptiveFilter:
        # Use adaptive Gaussian filtering again
        import scipy.signal as ss
        ionos = np.fromfile(outDispersive, dtype=np.float32).reshape(length, width)
        std = np.fromfile(sigmaDispersive, dtype=np.float32).reshape(length, width)
        mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
        ionos[mask==0] = 0
        std[mask==0] = 0
        
        # Global polynomial fitting for corrected dispersive phase
        ionos_fit = None
        if fitIon:
            wgt = std**2
            wgt[np.nonzero(std==0)] = 0
            if inps.lowBandCoherence and inps.highBandCoherence:
                try:
                    cor_low = np.fromfile(inps.lowBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :] if os.path.exists(inps.lowBandCoherence + '.xml') else None
                    cor_high = np.fromfile(inps.highBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :] if os.path.exists(inps.highBandCoherence + '.xml') else None
                    if cor_low is not None and cor_high is not None:
                        cor = (cor_low + cor_high) / 2.0
                        cor[np.nonzero(cor<0)] = 0.0
                        cor[np.nonzero(cor>1)] = 0.0
                        wgt[np.nonzero(cor<corThresholdFit)] = 0
                except:
                    pass
            index = np.nonzero(wgt!=0)
            if len(index[0]) > 0:
                wgt[index] = 1.0/(wgt[index])
                ionos_fit, _ = polyfit_2d(ionos.copy(), wgt, 2)
                ionos = ionos - ionos_fit * (ionos!=0)
        
        ionos_filt = None
        std_filt = None
        window_size_corrected = None
        g2d = None
        if filtIon:
            ionos_filt, std_filt, window_size_corrected = adaptive_gaussian(
                ionos.copy(), std.copy(), size_min, size_max, std_out0, fit=fitAdaptive)
            
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
        
        # Apply mask to filtered results to ensure masked regions remain zero
        # This is critical because filtering can fill masked regions with non-zero values
        # from neighboring valid pixels, especially at boundaries
        mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
        ionos_final[np.nonzero(mask==0)] = 0.0
        if filtIon and std_filt is not None:
            std_filt[np.nonzero(mask==0)] = 0.0
        
        ionos_final.astype(np.float32).tofile(outDispersive + ".filt")
        write_xml(outDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtIon and std_filt is not None:
            std_filt.astype(np.float32).tofile(sigmaDispersive + ".filt")
            write_xml(sigmaDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
            # Save window size file for corrected dispersive phase (same as Alos2Proc/StripmapProc)
            if window_size_corrected is not None:
                windowSizeDispersive = outDispersive + ".filt.win"
                windowSizeDispersive = os.path.abspath(windowSizeDispersive)
                print('Saving corrected dispersive window size file: {}'.format(windowSizeDispersive))
                logger.info('Saving corrected dispersive window size file: {}'.format(windowSizeDispersive))
                window_size_corrected.astype(np.float32).tofile(windowSizeDispersive)
                write_xml(windowSizeDispersive, width, length, 1, "FLOAT", "BIL")
                print('Saved corrected dispersive window size file: {}'.format(windowSizeDispersive))
                logger.info('Saved corrected dispersive window size file: {}'.format(windowSizeDispersive))
            else:
                print('WARNING: Window size is None for corrected dispersive phase!')
                logger.warning('Window size is None for corrected dispersive phase (filtIon={}, std_filt is not None={})'.format(
                    filtIon, std_filt is not None))
        
        nonDisp = np.fromfile(outNonDispersive, dtype=np.float32).reshape(length, width)
        std_nonDisp = np.fromfile(sigmaNonDispersive, dtype=np.float32).reshape(length, width)
        nonDisp[mask==0] = 0
        std_nonDisp[mask==0] = 0
        
        # Global polynomial fitting for corrected non-dispersive phase
        nonDisp_fit = None
        if fitIon:
            wgt = std_nonDisp**2
            wgt[np.nonzero(std_nonDisp==0)] = 0
            if inps.lowBandCoherence and inps.highBandCoherence:
                try:
                    cor_low = np.fromfile(inps.lowBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :] if os.path.exists(inps.lowBandCoherence + '.xml') else None
                    cor_high = np.fromfile(inps.highBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :] if os.path.exists(inps.highBandCoherence + '.xml') else None
                    if cor_low is not None and cor_high is not None:
                        cor = (cor_low + cor_high) / 2.0
                        cor[np.nonzero(cor<0)] = 0.0
                        cor[np.nonzero(cor>1)] = 0.0
                        wgt[np.nonzero(cor<corThresholdFit)] = 0
                except:
                    pass
            index = np.nonzero(wgt!=0)
            if len(index[0]) > 0:
                wgt[index] = 1.0/(wgt[index])
                nonDisp_fit, _ = polyfit_2d(nonDisp.copy(), wgt, 2)
                nonDisp = nonDisp - nonDisp_fit * (nonDisp!=0)
        
        nonDisp_filt = None
        std_nonDisp_filt = None
        window_size_nonDisp_corrected = None
        if filtIon:
            nonDisp_filt, std_nonDisp_filt, window_size_nonDisp_corrected = adaptive_gaussian(
                nonDisp.copy(), std_nonDisp.copy(), size_min, size_max, std_out0, fit=fitAdaptive)
            
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
        if fitIon and filtIon:
            nonDisp_final = nonDisp_filt + nonDisp_fit * (nonDisp_filt!=0)
        elif fitIon and not filtIon:
            nonDisp_final = nonDisp_fit
        elif not fitIon and filtIon:
            nonDisp_final = nonDisp_filt
        else:
            nonDisp_final = nonDisp
        
        nonDisp_final.astype(np.float32).tofile(outNonDispersive + ".filt")
        write_xml(outNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtIon and std_nonDisp_filt is not None:
            std_nonDisp_filt.astype(np.float32).tofile(sigmaNonDispersive + ".filt")
            write_xml(sigmaNonDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
            # Save window size file for corrected non-dispersive phase (same as Alos2Proc/StripmapProc)
            if window_size_nonDisp_corrected is not None:
                windowSizeNonDispersive = outNonDispersive + ".filt.win"
                windowSizeNonDispersive = os.path.abspath(windowSizeNonDispersive)
                print('Saving corrected non-dispersive window size file: {}'.format(windowSizeNonDispersive))
                logger.info('Saving corrected non-dispersive window size file: {}'.format(windowSizeNonDispersive))
                window_size_nonDisp_corrected.astype(np.float32).tofile(windowSizeNonDispersive)
                write_xml(windowSizeNonDispersive, width, length, 1, "FLOAT", "BIL")
                print('Saved corrected non-dispersive window size file: {}'.format(windowSizeNonDispersive))
                logger.info('Saved corrected non-dispersive window size file: {}'.format(windowSizeNonDispersive))
            else:
                print('WARNING: Window size is None for corrected non-dispersive phase!')
                logger.warning('Window size is None for corrected non-dispersive phase (filtIon={}, std_nonDisp_filt is not None={})'.format(
                    filtIon, std_nonDisp_filt is not None))
        
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
    


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()

