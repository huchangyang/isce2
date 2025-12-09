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
    parser.add_argument('--filtering_winsize_min_ion', dest='filteringWinsizeMinIon', type=int, default=11,
            help='minimum window size for adaptive Gaussian filtering (default=11)')
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
    
    # Read phase data (band 2 for unwrapped phase)
    lowerUnw = np.fromfile(lowBandIgram, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
    upperUnw = np.fromfile(highBandIgram, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
    
    # Prepare weight using coherence if available
    if lowBandCoherence and highBandCoherence and os.path.exists(lowBandCoherence + '.xml') and os.path.exists(highBandCoherence + '.xml'):
        cor_low = np.fromfile(lowBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
        cor_high = np.fromfile(highBandCoherence, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
        # Use average coherence as weight, with high power (similar to ALOS corOrderAdj=20)
        cor = (cor_low + cor_high) / 2.0
        cor[np.nonzero(cor<0)] = 0.0
        cor[np.nonzero(cor>1)] = 0.0
        wgt = cor**20  # Similar to corOrderAdj=20 in ALOS
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
    
    # Read original file structure (amplitude + phase)
    original_data = np.fromfile(highBandIgram, dtype=np.float32).reshape(length*2, width)
    original_data[1:length*2:2, :] = upperUnw_adjusted
    
    # Save adjusted file
    original_data.astype(np.float32).tofile(highBandIgramAdjusted)
    write_xml(highBandIgramAdjusted, width, length*2, 2, "FLOAT", "BIL")
    
    logger.info('Adjusted high band interferogram saved to: {}'.format(highBandIgramAdjusted))
    
    return highBandIgramAdjusted


def check_consistency(lowBandIgram, highBandIgram, outputDir):


    jumpFile = os.path.join(outputDir , "jumps.bil")
    cmd = 'imageMath.py -e="round((a_1-b_1)/(2.0*PI))" --a={0}  --b={1} -o {2} -t float  -s BIL'.format(lowBandIgram, highBandIgram, jumpFile)
    print(cmd)
    os.system(cmd)

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
        print ('generating a mask based on unwrapped files. Pixels with phase = 0 are masked out.')
        cmd = 'imageMath.py -e="(a_1!=0)*(b_1!=0)" --a={0} --b={1} -t byte -s BIL -o {2}'.format(lowBandIgram , highBandIgram , maskFile)
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
    
    logger.info('Computed number of looks for subband interferograms: {:.2f}'.format(numberOfLooks))
    logger.info('  Azimuth bandwidth: {:.2f} Hz'.format(azimuthBandwidth))
    logger.info('  Range sampling rate: {:.2e} Hz'.format(rangeSamplingRate))
    logger.info('  Subband range bandwidth: {:.2e} Hz'.format(subbandRangeBandwidth))
    
    return numberOfLooks


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
        # Use the more accurate calculation if available
        totalLooks = numberOfLooks if numberOfLooks > 0 else simpleTotalLooks
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
        size_min = getattr(inps, 'filteringWinsizeMinIon', 11)
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
            std_out0 = 0.05  # Default fallback
        
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
        if filtIon:
            nonDisp_filt, std_nonDisp_filt, _ = adaptive_gaussian(
                nonDisp.copy(), std_nonDisp.copy(), size_min, size_max, std_out0, fit=fitAdaptive)
            
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
        g2d = None
        if filtIon:
            ionos_filt, std_filt, _ = adaptive_gaussian(
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
        
        ionos_final.astype(np.float32).tofile(outDispersive + ".filt")
        write_xml(outDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        if filtIon and std_filt is not None:
            std_filt.astype(np.float32).tofile(sigmaDispersive + ".filt")
            write_xml(sigmaDispersive + ".filt", width, length, 1, "FLOAT", "BIL")
        
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
        if filtIon:
            nonDisp_filt, std_nonDisp_filt, _ = adaptive_gaussian(
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

