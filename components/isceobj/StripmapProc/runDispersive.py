#
# Author: Heresh Fattahi, Cunren Liang
#
#
import logging
import os
from osgeo import gdal
import isceobj
from isceobj.Constants import SPEED_OF_LIGHT
import numpy as np




logger = logging.getLogger('isce.insar.runDispersive')

def getValue(dataFile, band, y_ref, x_ref):
    ds = gdal.Open(dataFile, gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    b = ds.GetRasterBand(band)
    ref = b.ReadAsArray(x_ref,y_ref,1,1)
    
    ds = None
    return ref[0][0]


def computeIonosphereFromComplexInt(lowerInt, upperInt, cor, fl, fu, adjFlag=1):
    '''
    Compute ionosphere from complex interferograms (multilooked)
    Similar to Alos2Proc computeIonosphere but works with complex interferograms
    
    lowerInt:  lower band complex interferogram (numpy array)
    upperInt:  upper band complex interferogram (numpy array)
    cor:       coherence (numpy array)
    fl:        lower band center frequency
    fu:        upper band center frequency
    adjFlag:   method for removing relative phase unwrapping errors
                0: mean value
                1: polynomial
    '''
    # Extract phase from complex interferograms
    lowerPhase = np.angle(lowerInt)
    upperPhase = np.angle(upperInt)
    
    # Compute phase difference from complex product (more accurate)
    phaseDiffComplex = lowerInt * np.conj(upperInt)
    phaseDiff = np.angle(phaseDiffComplex)
    
    # Weight by coherence
    corOrderAdj = 20
    wgt = cor**corOrderAdj
    wgt[np.nonzero(cor<0.97)] = 0  # Only use high coherence pixels
    
    # Unwrap phase difference using polynomial fitting
    if np.sum(wgt>0) > 100:  # Need enough valid pixels
        diff_fit, coeff = polyfit_2d(phaseDiff, wgt, 2)
        unwd = np.round((phaseDiff - diff_fit) / (2.0*np.pi))
        upperPhaseUnw = upperPhase + unwd * (2.0*np.pi)
    else:
        # Fall back to mean adjustment
        flag = (np.abs(lowerInt)!=0)*(wgt!=0)
        index = np.nonzero(flag!=0)
        if len(index[0]) > 0:
            mv = np.mean((phaseDiff)[index], dtype=np.float64)
            unwd = np.round((phaseDiff - mv) / (2.0*np.pi))
            upperPhaseUnw = upperPhase + unwd * (2.0*np.pi)
        else:
            upperPhaseUnw = upperPhase
    
    # Now use computeIonosphere logic
    lowerUnw = lowerPhase.copy()
    upperUnw = upperPhaseUnw.copy()
    
    # Adjust phase using polynomial or mean
    if adjFlag == 0:
        flag = (lowerUnw!=0)*(wgt!=0)
        index = np.nonzero(flag!=0)
        mv = np.mean((lowerUnw - upperUnw)[index], dtype=np.float64)
        diff = mv
    else:
        diff, coeff = polyfit_2d(lowerUnw - upperUnw, wgt, 2)
    
    flag2 = (lowerUnw!=0)
    index2 = np.nonzero(flag2)
    unwd = ((lowerUnw - upperUnw) - diff)[index2] / (2.0*np.pi)
    unw_adj = np.around(unwd) * (2.0*np.pi)
    upperUnw[index2] += unw_adj
    
    # Compute ionosphere
    f0 = (fl + fu) / 2.0
    ionos = fl * fu * (lowerUnw * fu - upperUnw * fl) / f0 / (fu**2 - fl**2)
    
    return ionos

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

def dispersive_nonDispersive(lowBandIgram, highBandIgram, f0, fL, fH, outDispersive, outNonDispersive, y_ref=None, x_ref=None, m=None , d=None):
    
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
        cmd = 'imageMath.py -e="{0}*((a_1-2*PI*c)*{1}-(b_1+(2.0*PI)-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} -o {7} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, m , d, outDispersive)
        print(cmd)
        os.system(cmd)

        coefn = f0/(fH**2-fL**2)
        cmd = 'imageMath.py -e="{0}*((a_1+(2.0*PI)-2*PI*c)*{1}-(b_1-2*PI*(c+f))*{2})" --a={3} --b={4} --c={5} --f={6} -o {7} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, m , d, outNonDispersive)
        print(cmd)
        os.system(cmd)

    else:
        
        coef = (fL*fH)/(f0*(fH**2 - fL**2))
        cmd = 'imageMath.py -e="{0}*(a_1*{1}-(b_1+2.0*PI)*{2})" --a={3} --b={4} -o {5} -t float32 -s BIL'.format(coef,fH, fL, lowBandIgram, highBandIgram, outDispersive)

        print(cmd)
        os.system(cmd)

        coefn = f0/(fH**2-fL**2)
        cmd = 'imageMath.py -e="{0}*((a_1+2.0*PI)*{1}-(b_1)*{2})" --a={3} --b={4} -o {5} -t float32 -s BIL'.format(coefn,fH, fL, highBandIgram, lowBandIgram, outNonDispersive)
        print(cmd)
        os.system(cmd)


    return None

def std_iono_mean_coh(f0,fL,fH,coh_mean,rgLooks,azLooks):
    
    # From Liao et al., Remote Sensing of Environment 2018
    
    # STD sub-band at average coherence value (Eq. 8)
    Nb = (rgLooks*azLooks)/3.0
    coeffA = (np.sqrt(2.0*Nb))**(-1)
    coeffB = np.sqrt(1-coh_mean**2)/coh_mean
    std_subbands = coeffA * coeffB
    
    # STD Ionosphere (Eq. 7)
    coeffC = np.sqrt(1+(fL/fH)**2)
    coeffD = (fH*fL*fH)/(f0*(fH**2-fL**2))
    std_iono = coeffC*coeffD*std_subbands
    
    return std_iono
    
def theoretical_variance_fromSubBands(self, f0, fL, fH, B, Sig_phi_iono, Sig_phi_nonDisp,N):
    
    # Calculating the theoretical variance of the ionospheric phase based on the coherence of the sub-band interferograns 
    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
    lowBandCoherence = os.path.join(ifgDirname , self.insar.coherenceFilename)
    Sig_phi_L = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename + ".sig")

    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)
    highBandCoherence = os.path.join(ifgDirname , self.insar.coherenceFilename)
    Sig_phi_H = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename + ".sig")
    
    cmd = 'imageMath.py -e="sqrt(1-a**2)/a/sqrt(2.0*{0})" --a={1} -o {2} -t float -s BIL'.format(N, lowBandCoherence, Sig_phi_L)
   
    print(cmd)
    os.system(cmd)
    cmd = 'imageMath.py -e="sqrt(1-a**2)/a/sqrt(2.0*{0})" --a={1} -o {2} -t float -s BIL'.format(N, highBandCoherence, Sig_phi_H)
    print(cmd)
    os.system(cmd)

    coef = (fL*fH)/(f0*(fH**2 - fL**2))

    cmd = 'imageMath.py -e="sqrt(({0}**2)*({1}**2)*(a**2) + ({0}**2)*({2}**2)*(b**2))" --a={3} --b={4} -o {5} -t float -s BIL'.format(coef, fL, fH, Sig_phi_L, Sig_phi_H, Sig_phi_iono)
    os.system(cmd)

    coef_non = f0/(fH**2 - fL**2)
    cmd = 'imageMath.py -e="sqrt(({0}**2)*({1}**2)*(a**2) + ({0}**2)*({2}**2)*(b**2))" --a={3} --b={4} -o {5} -t float -s BIL'.format(coef_non, fL, fH, Sig_phi_L, Sig_phi_H, Sig_phi_nonDisp)
    os.system(cmd)

  
    return None #Sig_phi_iono, Sig_phi_nonDisp

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
        # Gaussian kernel - use np.tile instead of deprecated np.matlib.repmat for better performance
        hsize = (size - 1) / 2
        x = np.arange(-hsize, hsize + 1)
        f = np.exp(-x**2/(2.0*(size/2.0)**2)) / ((size/2.0) * np.sqrt(2.0*np.pi))
        # More efficient: use np.outer for 2D Gaussian kernel
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
    # Further reduce print frequency for better performance (print every 100 lines or 5% progress)
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

def lowPassFilter(self,dataFile, sigDataFile, maskFile, Sx, Sy, sig_x, sig_y, iteration=5, theta=0.0):
    ds = gdal.Open(dataFile + '.vrt', gdal.GA_ReadOnly)
    length = ds.RasterYSize
    width = ds.RasterXSize

    dataIn = np.memmap(dataFile, dtype=np.float32, mode='r', shape=(length,width))
    sigData = np.memmap(sigDataFile, dtype=np.float32, mode='r', shape=(length,width))
    mask = np.memmap(maskFile, dtype=np.byte, mode='r', shape=(length,width))

    dataF, sig_dataF = iterativeFilter(self,dataIn[:,:], mask[:,:], sigData[:,:], iteration, Sx, Sy, sig_x, sig_y, theta)

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

def iterativeFilter(self,dataIn, mask, Sig_dataIn, iteration, Sx, Sy, sig_x, sig_y, theta=0.0):
    data = np.zeros(dataIn.shape)
    data[:,:] = dataIn[:,:]
    Sig_data = np.zeros(dataIn.shape)
    Sig_data[:,:] = Sig_dataIn[:,:]

    print ('masking the data')
    data[mask==0]=np.nan
    Sig_data[mask==0]=np.nan
    
    if self.dispersive_filling_method == "smoothed":
       print('Filling the holes with smoothed values')
       dataF = fill_with_smoothed(data,3)
       Sig_data = fill_with_smoothed(Sig_data,3) 
    else:
       print ('Filling the holes with nearest neighbor interpolation')
       dataF = fill(data)
       Sig_data = fill(Sig_data)
       
    print ('Low pass Gaussian filtering the interpolated data')
    dataF, Sig_dataF = Filter(dataF, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0)
    for i in range(iteration):
       print ('iteration: ', i , ' of ',iteration)
       print ('masking the interpolated and filtered data')
       dataF[mask==0]=np.nan
       
       if self.dispersive_filling_method == "smoothed":
          print("Fill the holes with smoothed values")
          dataF = fill_with_smoothed(dataF,3)
       else:
          print('Filling the holes with nearest neighbor interpolation of the filtered data from previous step')
          dataF = fill(dataF)

       print('Replace the valid pixels with original unfiltered data')
       dataF[mask==1]=data[mask==1]
       dataF, Sig_dataF = Filter(dataF, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0)

    return dataF, Sig_dataF

def Filter(data, Sig_data, Sx, Sy, sig_x, sig_y, theta=0.0):
    
    import cv2

    kernel = Gaussian_kernel(Sx, Sy, sig_x, sig_y) #(800, 800, 15.0, 100.0)
    kernel = rotate(kernel , theta)

    data = data/Sig_data**2
    data = cv2.filter2D(data,-1,kernel)
    W1 = cv2.filter2D(1.0/Sig_data**2,-1,kernel)
    W2 = cv2.filter2D(1.0/Sig_data**2,-1,kernel**2)

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

def fill_with_smoothed(off,filterSize):
    
    from astropy.convolution import convolve
    
    off_2filt=np.copy(off)
    kernel = np.ones((filterSize,filterSize),np.float32)/(filterSize*filterSize)
    loop = 0
    cnt2=1
    
    while (cnt2!=0 & loop<100):
       loop += 1
       idx2= np.isnan(off_2filt)
       cnt2 = np.sum(np.count_nonzero(np.isnan(off_2filt)))
       print(cnt2)
       if cnt2 != 0:
          off_filt= convolve(off_2filt,kernel,boundary='extend',nan_treatment='interpolate')
          off_2filt[idx2]=off_filt[idx2]
          idx3 = np.where(off_filt == 0)
          off_2filt[idx3]=np.nan
          off_filt=None
          
    return off_2filt
    


def fill(data, invalid=None):
    
    from scipy import ndimage
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


def getMask(self, maskFile,std_iono, lowBandIgram=None, highBandIgram=None):
    
    from scipy.ndimage import median_filter
    
    # Get ionospheric looks parameters (needed for ml2 calculation and water mask processing)
    numberRangeLooksIon = getattr(self, 'numberRangeLooksIon', None)
    numberAzimuthLooksIon = getattr(self, 'numberAzimuthLooksIon', None)
    
    # If not found in self, try to get from self.insar (for direct StripmapProc usage)
    if numberRangeLooksIon is None:
        numberRangeLooksIon = getattr(self.insar, 'numberRangeLooksIon', None)
    if numberAzimuthLooksIon is None:
        numberAzimuthLooksIon = getattr(self.insar, 'numberAzimuthLooksIon', None)
    
    if numberRangeLooksIon is None:
        numberRangeLooksIon = getattr(self, 'numberRangeLooks', 1)
    if numberAzimuthLooksIon is None:
        numberAzimuthLooksIon = getattr(self, 'numberAzimuthLooks', 1)
    
    # Determine looks used for multilooking
    referenceFrame = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    azLooks, rgLooks = self.insar.numberOfLooks(referenceFrame, self.posting,
                                                self.numberAzimuthLooks, self.numberRangeLooks)
    
    # Total looks including ionospheric looks
    totalAzLooks = azLooks * numberAzimuthLooksIon
    totalRgLooks = rgLooks * numberRangeLooksIon
    
    ml2 = '_{}rlks_{}alks'.format(totalRgLooks, totalAzLooks)
    
    # If lowBandIgram and highBandIgram are provided, use them directly
    # Otherwise, detect multilooked unwrapped interferograms if available
    if lowBandIgram is None or highBandIgram is None:
        # Look for multilooked unwrapped interferograms first
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
        lowBandIgramUnw = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename.replace('.flat', ml2 + '.unw'))
        if not os.path.exists(lowBandIgramUnw):
            # Try without filt_ prefix
            lowBandIgramUnw = os.path.join(ifgDirname, self.insar.ifgFilename.replace('.flat', ml2 + '.unw'))
        if not os.path.exists(lowBandIgramUnw):
            # Fall back to regular .unw files
            lowBandIgram = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename)
            if '.flat' in lowBandIgram:
                lowBandIgram = lowBandIgram.replace('.flat', '.unw')
            elif '.int' in lowBandIgram:
                lowBandIgram = lowBandIgram.replace('.int', '.unw')
            else:
                lowBandIgram += '.unw'
        else:
            lowBandIgram = lowBandIgramUnw

    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)
    highBandIgramUnw = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename.replace('.flat', ml2 + '.unw'))
    if not os.path.exists(highBandIgramUnw):
        # Try without filt_ prefix
        highBandIgramUnw = os.path.join(ifgDirname, self.insar.ifgFilename.replace('.flat', ml2 + '.unw'))
    if not os.path.exists(highBandIgramUnw):
        # Fall back to regular .unw files
        highBandIgram = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename)
        if '.flat' in highBandIgram:
            highBandIgram = highBandIgram.replace('.flat', '.unw')
        elif '.int' in highBandIgram:
            highBandIgram = highBandIgram.replace('.int', '.unw')
        else:
            highBandIgram += '.unw'
    else:
        highBandIgram = highBandIgramUnw
    
    # Get coherence file paths (these should match the unwrapped interferogram dimensions)
    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
    lowBandCor = os.path.join(ifgDirname, self.insar.coherenceFilename)
    
    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)
    highBandCor = os.path.join(ifgDirname, self.insar.coherenceFilename)

    if (self.dispersive_filter_mask_type == "coherence") and (not self.dispersive_filter_mask_type == "median_filter"):
        print ('generating a mask based on coherence files of sub-band interferograms with a threshold of {0}'.format(self.dispersive_filter_coherence_threshold))
        cmd = 'imageMath.py -e="(a>{0})*(b>{0})" --a={1} --b={2} -t byte -s BIL -o {3}'.format(self.dispersive_filter_coherence_threshold, lowBandCor, highBandCor, maskFile)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError('Failed to generate mask file using coherence files. Command: {}'.format(cmd))
    elif (self.dispersive_filter_mask_type == "connected_components") and ((os.path.exists(lowBandIgram + '.conncomp')) and (os.path.exists(highBandIgram + '.conncomp'))):
       # If connected components from snaphu exists, let's get a mask based on that. 
       # Regions of zero are masked out. Let's assume that islands have been connected. 
        print ('generating a mask based on .conncomp files')
        cmd = 'imageMath.py -e="(a>0)*(b>0)" --a={0} --b={1} -t byte -s BIL -o {2}'.format(lowBandIgram + '.conncomp', highBandIgram + '.conncomp', maskFile)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError('Failed to generate mask file using connected components. Command: {}'.format(cmd))

    elif self.dispersive_filter_mask_type == "median_filter":
        print('Generating mask based on median filtering of the raw dispersive component')
	
	# Open raw dispersive component (non-filtered, no unwrapping-error corrected)
        dispFilename = os.path.join(self.insar.ionosphereDirname,self.insar.dispersiveFilename)
        sigFilename = os.path.join(self.insar.ionosphereDirname,self.insar.dispersiveFilename+'.sig')
	
        ds = gdal.Open(dispFilename+'.vrt',gdal.GA_ReadOnly)
        disp = ds.GetRasterBand(1).ReadAsArray()
        ds=None

        mask = (np.abs(disp-median_filter(disp,15))<3*std_iono) 
        
        mask = mask.astype(np.float32)
        mask.tofile(maskFile)
        dims=np.shape(mask)
        write_xml(maskFile,dims[1],dims[0],1,"FLOAT","BIL")

    else:
        print ('generating a mask based on unwrapped files. Pixels with phase = 0 are masked out.')
        cmd = 'imageMath.py -e="(a_1!=0)*(b_1!=0)" --a={0} --b={1} -t byte -s BIL -o {2}'.format(lowBandIgram , highBandIgram , maskFile)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError('Failed to generate mask file using unwrapped files. Command: {}'.format(cmd))
    
    # Apply water body mask if available (matching Alos2Proc behavior)
    # Check for water body file in the geometry directory
    geometryDir = getattr(self.insar, 'geometryDirname', None)
    if geometryDir is None:
        # Try alternative geometry directory name
        geometryDir = getattr(self.insar, 'geometryDir', None)
    if geometryDir is None:
        # Fallback: try to construct geometry directory path
        # Usually geometry directory is at the same level as interferogram directory
        ifgDirname = os.path.dirname(self.insar.ifgDirname) if hasattr(self.insar, 'ifgDirname') else None
        if ifgDirname:
            geometryDir = os.path.join(ifgDirname, 'geometry')
        else:
            geometryDir = 'geometry'  # Last resort: current directory
    
    # Get ionospheric looks parameters to determine if we need multilooking
    numberRangeLooksIon = getattr(self, 'numberRangeLooksIon', None)
    numberAzimuthLooksIon = getattr(self, 'numberAzimuthLooksIon', None)
    if numberRangeLooksIon is None:
        numberRangeLooksIon = getattr(self.insar, 'numberRangeLooksIon', None)
    if numberAzimuthLooksIon is None:
        numberAzimuthLooksIon = getattr(self.insar, 'numberAzimuthLooksIon', None)
    
    # Determine total looks if ionospheric looks are specified
    if numberRangeLooksIon and numberAzimuthLooksIon and (numberRangeLooksIon > 1 or numberAzimuthLooksIon > 1):
        referenceFrame = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
        azLooks, rgLooks = self.insar.numberOfLooks(referenceFrame, self.posting,
                                                    self.numberAzimuthLooks, self.numberRangeLooks)
        totalAzLooks = azLooks * numberAzimuthLooksIon
        totalRgLooks = rgLooks * numberRangeLooksIon
        ml2 = '_{}rlks_{}alks'.format(totalRgLooks, totalAzLooks)
        
        # First try to find waterMask with multilook suffix
        wbdFile = os.path.join(geometryDir, 'waterMask' + ml2 + '.rdr')
        if not os.path.exists(wbdFile + '.xml'):
            # If multilooked version doesn't exist, try original resolution
            wbdFileOrig = os.path.join(geometryDir, 'waterMask.rdr')
            if os.path.exists(wbdFileOrig + '.xml'):
                logger.info('Found original resolution waterMask.rdr, applying additional multilooking')
                # Need to multilook the water mask
                from mroipac.looks.Looks import Looks
                
                # Load original water mask
                img_wbd_orig = isceobj.createImage()
                img_wbd_orig.load(wbdFileOrig + '.xml')
                img_wbd_orig.filename = wbdFileOrig
                width_orig = img_wbd_orig.width
                length_orig = img_wbd_orig.length
                
                # Create Looks object for multilooking
                looksObj = Looks()
                looksObj.setDownLooks(numberRangeLooksIon)
                looksObj.setAcrossLooks(numberAzimuthLooksIon)
                looksObj.setInputImage(img_wbd_orig)
                looksObj.setOutputFilename(wbdFile)
                looksObj.looks()
                
                logger.info('Multilooked waterMask from {}x{} to match interferogram resolution'.format(
                    width_orig, length_orig))
            else:
                wbdFile = None
        else:
            logger.info('Found multilooked waterMask: {}'.format(wbdFile))
    else:
        # No additional multilooking needed, use original resolution
        wbdFile = os.path.join(geometryDir, 'waterMask.rdr')
        if not os.path.exists(wbdFile + '.xml'):
            wbdFile = None
    
    # Apply water body mask if found
    if wbdFile and os.path.exists(wbdFile + '.xml'):
        logger.info('Applying water body mask from: {}'.format(wbdFile))
        # Load mask and water body files
        img_mask = isceobj.createImage()
        img_mask.load(maskFile + '.xml')
        width = img_mask.width
        length = img_mask.length
        
        # Load water body mask and check dimensions
        img_wbd = isceobj.createImage()
        img_wbd.load(wbdFile + '.xml')
        width_wbd = img_wbd.width
        length_wbd = img_wbd.length
        
        if width != width_wbd or length != length_wbd:
            logger.warning('Water mask dimensions ({}, {}) do not match mask dimensions ({}, {}). Skipping water body mask.'.format(
                width_wbd, length_wbd, width, length))
        else:
            mask = np.fromfile(maskFile, dtype=np.byte).reshape(length, width)
            wbd = np.fromfile(wbdFile, dtype=np.int8).reshape(length, width)
            
            # Mask out water body regions (wbd==0 means water, wbd==1 means land)
            mask[np.nonzero(wbd==0)] = 0
            
            # Save updated mask
            mask.astype(np.byte).tofile(maskFile)
            logger.info('Water body mask applied: {} pixels masked out'.format(np.sum(wbd==0)))
    
    # Verify that mask file was created
    if not os.path.exists(maskFile):
        raise RuntimeError('Mask file was not created: {}'.format(maskFile))
    if not os.path.exists(maskFile + '.xml'):
        raise RuntimeError('Mask file XML was not created: {}'.format(maskFile + '.xml'))

def unwrapp_error_correction(f0, B, dispFile, nonDispFile,lowBandIgram, highBandIgram, y_ref=None, x_ref=None):

    dFile = os.path.join(os.path.dirname(dispFile) , "dJumps.bil")
    mFile = os.path.join(os.path.dirname(dispFile) , "mJumps.bil")

    if y_ref and x_ref:
        refL = getValue(lowBandIgram, 2, y_ref, x_ref)
        refH = getValue(highBandIgram, 2, y_ref, x_ref)

    else:
        refL = 0.0
        refH = 0.0

    cmd = 'imageMath.py -e="round(((a_1+(2.0*PI)) - (b_1) - (2.0*{0}/3.0/{1})*c + (2.0*{0}/3.0/{1})*f )/2.0/PI)" --a={2} --b={3} --c={4} --f={5}  -o {6} -t float32 -s BIL'.format(B, f0, highBandIgram, lowBandIgram, nonDispFile, dispFile, dFile)
    print(cmd)
    os.system(cmd)
    
    cmd = 'imageMath.py -e="round(((a_1 ) + (b_1+(2.0*PI)) - 2.0*c - 2.0*f )/4.0/PI - g/2)" --a={0} --b={1} --c={2} --f={3} --g={4} -o {5} -t float32 -s BIL'.format(lowBandIgram, highBandIgram, nonDispFile, dispFile, dFile, mFile)
    print(cmd)
    os.system(cmd)

    return mFile , dFile


def runDispersive(self):

    if not self.doDispersive:
        print('Estimating dispersive phase not requested ... skipping')
        return

    # Use multilooked and unwrapped interferograms (.unw) for ionosphere estimation
    # Get ionospheric looks if available, otherwise use regular looks
    # In runDispersive, self is the Insar instance, so get parameters from self first
    numberRangeLooksIon = getattr(self, 'numberRangeLooksIon', None)
    numberAzimuthLooksIon = getattr(self, 'numberAzimuthLooksIon', None)
    
    # If not found in self, try to get from self.insar (for direct StripmapProc usage)
    if numberRangeLooksIon is None:
        numberRangeLooksIon = getattr(self.insar, 'numberRangeLooksIon', None)
    if numberAzimuthLooksIon is None:
        numberAzimuthLooksIon = getattr(self.insar, 'numberAzimuthLooksIon', None)
    
    if numberRangeLooksIon is None:
        numberRangeLooksIon = getattr(self, 'numberRangeLooks', 1)
    if numberAzimuthLooksIon is None:
        numberAzimuthLooksIon = getattr(self, 'numberAzimuthLooks', 1)
    
    # Determine looks used for multilooking
    referenceFrame = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    azLooks, rgLooks = self.insar.numberOfLooks(referenceFrame, self.posting,
                                                self.numberAzimuthLooks, self.numberRangeLooks)
    
    # Total looks including ionospheric looks
    totalAzLooks = azLooks * numberAzimuthLooksIon
    totalRgLooks = rgLooks * numberRangeLooksIon
    
    ml2 = '_{}rlks_{}alks'.format(totalRgLooks, totalAzLooks)
    
    # Look for multilooked and unwrapped interferograms first
    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
    lowBandIgramUnw = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename.replace('.flat', ml2 + '.unw'))
    if not os.path.exists(lowBandIgramUnw):
        # Try without filt_ prefix
        lowBandIgramUnw = os.path.join(ifgDirname, self.insar.ifgFilename.replace('.flat', ml2 + '.unw'))
    if not os.path.exists(lowBandIgramUnw):
        # Fall back to original method with regular .unw files
        logger.info('Multilooked unwrapped interferogram not found, using regular unwrapped phase files')
        lowBandIgram = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename)
        if '.flat' in lowBandIgram:
            lowBandIgram = lowBandIgram.replace('.flat', '.unw')
        elif '.int' in lowBandIgram:
            lowBandIgram = lowBandIgram.replace('.int', '.unw')
        else:
            lowBandIgram += '.unw'
        useMultilookedUnw = False
    else:
        lowBandIgram = lowBandIgramUnw
        useMultilookedUnw = True

    ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)
    highBandIgramUnw = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename.replace('.flat', ml2 + '.unw'))
    if not os.path.exists(highBandIgramUnw):
        # Try without filt_ prefix
        highBandIgramUnw = os.path.join(ifgDirname, self.insar.ifgFilename.replace('.flat', ml2 + '.unw'))
    if not os.path.exists(highBandIgramUnw):
        # Fall back to original method with regular .unw files
        if useMultilookedUnw:
            logger.info('High band multilooked unwrapped interferogram not found, but low band found. Using regular unwrapped phase for high band.')
        highBandIgram = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename)
        if '.flat' in highBandIgram:
            highBandIgram = highBandIgram.replace('.flat', '.unw')
        elif '.int' in highBandIgram:
            highBandIgram = highBandIgram.replace('.int', '.unw')
        else:
            highBandIgram += '.unw'
        useMultilookedUnw = False
    else:
        highBandIgram = highBandIgramUnw
        if not useMultilookedUnw:
            useMultilookedUnw = True

    outputDir = self.insar.ionosphereDirname
    os.makedirs(outputDir, exist_ok=True)

    outDispersive = os.path.join(outputDir, self.insar.dispersiveFilename)
    sigmaDispersive = outDispersive + ".sig"

    outNonDispersive = os.path.join(outputDir, self.insar.nondispersiveFilename)
    sigmaNonDispersive = outNonDispersive + ".sig"

    maskFile = os.path.join(outputDir, "mask.bil")

    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)

    wvl = referenceFrame.radarWavelegth
    wvlL = self.insar.lowBandRadarWavelength
    wvlH = self.insar.highBandRadarWavelength

    
    f0 = SPEED_OF_LIGHT/wvl
    fL = SPEED_OF_LIGHT/wvlL
    fH = SPEED_OF_LIGHT/wvlH

    pulseLength = referenceFrame.instrument.pulseLength
    chirpSlope = referenceFrame.instrument.chirpSlope
   
    # Total Bandwidth
    B = np.abs(chirpSlope)*pulseLength
    
    
    ###Determine looks
    azLooks, rgLooks = self.insar.numberOfLooks( referenceFrame, self.posting,
                                        self.numberAzimuthLooks, self.numberRangeLooks)

    # estimating the dispersive and non-dispersive components
    # Use unwrapped phase files (.unw) - either multilooked or regular
    if useMultilookedUnw:
        logger.info('Using multilooked unwrapped interferograms for ionosphere estimation')
    else:
        logger.info('Using regular unwrapped interferograms for ionosphere estimation')
    
    # Use original method with unwrapped phase files
    dispersive_nonDispersive(lowBandIgram, highBandIgram, f0, fL, fH, outDispersive, outNonDispersive)

    # If median filter is selected, compute the ionosphere phase standard deviation at a mean coherence value defined by the user
    if self.dispersive_filter_mask_type == "median_filter":
       coh_thres = self.dispersive_filter_coherence_threshold
       std_iono = std_iono_mean_coh(f0,fL,fH,coh_thres,rgLooks,azLooks)
    else:
       std_iono = None
    
    # generating a mask which will help filtering the estimated dispersive and non-dispersive phase
    # Pass the detected unwrapped interferogram paths to getMask
    getMask(self, maskFile, std_iono, lowBandIgram=lowBandIgram, highBandIgram=highBandIgram)
    
    # Calculating the theoretical standard deviation of the estimation based on the coherence of the interferograms
    # Use more accurate number of looks calculation (ALOS-style) if possible
    try:
        # Compute more accurate numberOfLooks considering bandwidth and subband characteristics
        # Get azimuth bandwidth
        azimuthBandwidth = referenceFrame.instrument.pulseRepetitionFrequency * 0.85  # Typical for stripmap
        if hasattr(referenceFrame.instrument, 'azimuthBandwidth'):
            azimuthBandwidth = referenceFrame.instrument.azimuthBandwidth
        
        # Get range sampling rate
        if hasattr(referenceFrame.instrument, 'rangeSamplingRate'):
            rangeSamplingRate = referenceFrame.instrument.rangeSamplingRate
        else:
            rangeSamplingRate = B * 1.2  # Typical oversampling factor
        
        # Get azimuth line interval
        if hasattr(referenceFrame, 'azimuthLineInterval'):
            azimuthLineInterval = referenceFrame.azimuthLineInterval
        else:
            azimuthLineInterval = 1.0 / (azimuthBandwidth / (SPEED_OF_LIGHT / wvl))
        
        # Compute number of looks (ALOS-style formula)
        # Assume subband range bandwidth is 1/3 of original range bandwidth
        subbandRangeBandwidth = B / 3.0
        
        # Get ionospheric looks
        numberRangeLooksIon = getattr(self, 'numberRangeLooksIon', 1)
        numberAzimuthLooksIon = getattr(self, 'numberAzimuthLooksIon', 1)
        if numberRangeLooksIon is None:
            numberRangeLooksIon = getattr(self.insar, 'numberRangeLooksIon', 1)
        if numberAzimuthLooksIon is None:
            numberAzimuthLooksIon = getattr(self.insar, 'numberAzimuthLooksIon', 1)
        
        numberOfLooks = (azimuthLineInterval * azLooks * numberAzimuthLooksIon / (1.0/azimuthBandwidth)) * \
                        (subbandRangeBandwidth / rangeSamplingRate * rgLooks * numberRangeLooksIon)
        
        logger.info('Computed number of looks for subband interferograms: {:.2f}'.format(numberOfLooks))
        logger.info('  Azimuth bandwidth: {:.2f} Hz'.format(azimuthBandwidth))
        logger.info('  Range sampling rate: {:.2e} Hz'.format(rangeSamplingRate))
        logger.info('  Subband range bandwidth: {:.2e} Hz'.format(subbandRangeBandwidth))
        
        # Use the computed numberOfLooks for variance calculation
        totalLooks = numberOfLooks if numberOfLooks > 0 else azLooks * rgLooks
        logger.info('Using number of looks: {:.2f} (simple calculation: {:.2f})'.format(totalLooks, azLooks * rgLooks))
    except Exception as e:
        logger.warning('Failed to compute accurate number of looks: {}. Using simple calculation.'.format(e))
        totalLooks = azLooks * rgLooks
        if useMultilookedUnw and numberRangeLooksIon and numberAzimuthLooksIon:
            totalLooks = totalLooks * numberRangeLooksIon * numberAzimuthLooksIon
    
    theoretical_variance_fromSubBands(self, f0, fL, fH, B, sigmaDispersive, sigmaNonDispersive, totalLooks) 
    
    # Use adaptive Gaussian filtering if explicitly requested, otherwise use original iterative filtering
    useAdaptiveFilter = getattr(self, 'useAdaptiveGaussianFilter', False)
    useAdaptiveFilter = True
    if useAdaptiveFilter:
        # Use adaptive Gaussian filtering (similar to Alos2Proc)
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
        
        # Get filtering parameters (defaults match alosStack.xml)
        # Default values from alosStack.xml:
        # - maximum window size: 301
        # - minimum window size: 11
        # - window size of secondary filtering: 5
        # - apply polynomial fit in adaptive filtering window: True
        # - whether do secondary filtering: True
        size_max = getattr(self, 'filteringWinsizeMaxIon', 301)
        size_min = getattr(self, 'filteringWinsizeMinIon', 11)
        size_secondary = getattr(self, 'filteringWinsizeSecondaryIon', 5)
        std_out0 = getattr(self, 'filterStdIon', None)  # None means auto-determine based on mode
        fitAdaptive = getattr(self, 'fitAdaptiveIon', True)
        filtSecondary = getattr(self, 'filtSecondaryIon', True)
        fitIon = getattr(self, 'fitIon', True)
        filtIon = getattr(self, 'filtIon', True)
        corThresholdFit = getattr(self, 'fitIonCoherenceThreshold', 0.25)
        
        # Check that at least one of fit or filt is enabled
        if (not fitIon) and (not filtIon):
            raise Exception('either fitIon or filtIon should be True when doing ionospheric correction')
        
        # If std_out0 is None, use a reasonable default (matching ALOS-2 high-resolution modes)
        if std_out0 is None:
            std_out0 = 0.015  # Default for stripmap (matches ALOS-2 SPT/SM1 modes), can be overridden by user
        
        if size_min > size_max:
            size_max = size_min
        if size_secondary % 2 != 1:
            size_secondary += 1
            logger.info('Window size of secondary filtering should be odd, changed to {}'.format(size_secondary))
        
        # Global polynomial fitting (ALOS-style) before filtering
        ionos_fit = None
        ionos_orig = ionos.copy()  # Store original for later combination
        if fitIon:
            logger.info('Applying global polynomial fit to ionospheric phase (ALOS-style)')
            # Prepare weight using standard deviation
            wgt = std**2
            wgt[np.nonzero(std==0)] = 0
            
            # Apply coherence threshold if coherence files are available
            ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
            lowBandCor = os.path.join(ifgDirname, self.insar.coherenceFilename)
            ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)
            highBandCor = os.path.join(ifgDirname, self.insar.coherenceFilename)
            
            if os.path.exists(lowBandCor + '.xml') and os.path.exists(highBandCor + '.xml'):
                try:
                    cor_low = np.fromfile(lowBandCor, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
                    cor_high = np.fromfile(highBandCor, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
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
                ionos_fit, coeff = polyfit_2d(ionos_orig.copy(), wgt, 2)
                # Subtract fit from original data (only where data is valid)
                ionos = ionos_orig - ionos_fit * (ionos_orig!=0)
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
            # Combine: filtered residual + polynomial fit
            ionos_final = ionos_filt + ionos_fit * (ionos_filt!=0)
        elif fitIon and not filtIon:
            # If only fit is enabled, use the polynomial fit surface
            ionos_final = ionos_fit
        elif not fitIon and filtIon:
            ionos_final = ionos_filt
        else:
            # Neither fit nor filt enabled, use original
            ionos_final = ionos_orig
        
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
        nonDisp_orig = nonDisp.copy()  # Store original for later combination
        if fitIon:
            wgt = std_nonDisp**2
            wgt[np.nonzero(std_nonDisp==0)] = 0
            if os.path.exists(lowBandCor + '.xml') and os.path.exists(highBandCor + '.xml'):
                try:
                    cor_low = np.fromfile(lowBandCor, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
                    cor_high = np.fromfile(highBandCor, dtype=np.float32).reshape(length*2, width)[1:length*2:2, :]
                    cor = (cor_low + cor_high) / 2.0
                    cor[np.nonzero(cor<0)] = 0.0
                    cor[np.nonzero(cor>1)] = 0.0
                    wgt[np.nonzero(cor<corThresholdFit)] = 0
                except:
                    pass
            index = np.nonzero(wgt!=0)
            if len(index[0]) > 0:
                wgt[index] = 1.0/(wgt[index])
                nonDisp_fit, _ = polyfit_2d(nonDisp_orig.copy(), wgt, 2)
                nonDisp = nonDisp_orig - nonDisp_fit * (nonDisp_orig!=0)
        
        nonDisp_filt = None
        std_nonDisp_filt = None
        if filtIon:
            nonDisp_filt, std_nonDisp_filt, _ = adaptive_gaussian(
                nonDisp.copy(), std_nonDisp.copy(), size_min, size_max, std_out0, fit=fitAdaptive)
            
            # Apply secondary filtering to non-dispersive phase if requested
            if filtSecondary:
                logger.info('Applying secondary filtering to non-dispersive phase with window size {}'.format(size_secondary))
                import scipy.signal as ss
                # Create Gaussian kernel if not already created
                if 'g2d' not in locals() or g2d is None:
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
            nonDisp_final = nonDisp_orig
        
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
        # Original iterative filtering method (default)
        # low pass filtering the dispersive phase
        lowPassFilter(self,outDispersive, sigmaDispersive, maskFile, 
                    self.kernel_x_size, self.kernel_y_size, 
                    self.kernel_sigma_x, self.kernel_sigma_y, 
                    iteration = self.dispersive_filter_iterations, 
                    theta = self.kernel_rotation)

    # low pass filtering the  non-dispersive phase
    lowPassFilter(self,outNonDispersive, sigmaNonDispersive, maskFile, 
                    self.kernel_x_size, self.kernel_y_size,
                    self.kernel_sigma_x, self.kernel_sigma_y,
                    iteration = self.dispersive_filter_iterations,
                    theta = self.kernel_rotation)
            
            
    # Estimating phase unwrapping errors
    mFile , dFile = unwrapp_error_correction(f0, B, outDispersive+".filt", outNonDispersive+".filt", 
                                                    lowBandIgram, highBandIgram)

    # re-estimate the dispersive and non-dispersive phase components by taking into account the unwrapping errors
    outDispersive = outDispersive + ".unwCor"
    outNonDispersive = outNonDispersive + ".unwCor"
    dispersive_nonDispersive(lowBandIgram, highBandIgram, f0, fL, fH, outDispersive, outNonDispersive, m=mFile , d=dFile)

    # low pass filtering the new estimations 
    lowPassFilter(self,outDispersive, sigmaDispersive, maskFile, 
                    self.kernel_x_size, self.kernel_y_size,
                    self.kernel_sigma_x, self.kernel_sigma_y,
                    iteration = self.dispersive_filter_iterations,
                    theta = self.kernel_rotation)

    lowPassFilter(self,outNonDispersive, sigmaNonDispersive, maskFile,
                    self.kernel_x_size, self.kernel_y_size,
                    self.kernel_sigma_x, self.kernel_sigma_y,
                    iteration = self.dispersive_filter_iterations,
                    theta = self.kernel_rotation)
    
    # Resample ionospheric phase back to original multilooked resolution if additional looks were used
    if useMultilookedUnw and (numberRangeLooksIon > 1 or numberAzimuthLooksIon > 1):
        logger.info('Resampling ionospheric phase back to original multilooked resolution')
        
        # Get dimensions of multilooked ionosphere
        img_ion = isceobj.createImage()
        img_ion.load(outDispersive + '.xml')
        width_ion = img_ion.width
        length_ion = img_ion.length
        
        # Get dimensions of original multilooked interferogram
        # Use the same logic as lowBandIgram to find the correct unwrapped interferogram
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
        
        # Try to find the unwrapped interferogram that was actually used
        # First check if we used multilooked unwrapped interferogram
        if useMultilookedUnw and numberRangeLooksIon is not None and numberAzimuthLooksIon is not None:
            # Try multilooked unwrapped interferogram first
            ml2 = '_{}rlks_{}alks'.format(numberRangeLooksIon, numberAzimuthLooksIon)
            originalUnw = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename.replace('.flat', ml2 + '.unw'))
            if not os.path.exists(originalUnw + '.xml'):
                # Try without filt_ prefix
                originalUnw = os.path.join(ifgDirname, self.insar.ifgFilename.replace('.flat', ml2 + '.unw'))
        else:
            # Use regular unwrapped interferogram
            originalUnw = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename)
            if '.flat' in originalUnw:
                originalUnw = originalUnw.replace('.flat', '.unw')
            elif '.int' in originalUnw:
                originalUnw = originalUnw.replace('.int', '.unw')
            else:
                originalUnw += '.unw'
        
        # If still not found, try using lowBandIgram path directly (it was successfully loaded earlier)
        if not os.path.exists(originalUnw + '.xml'):
            if os.path.exists(lowBandIgram + '.xml'):
                originalUnw = lowBandIgram
                logger.info('Using lowBandIgram path for resampling: {}'.format(originalUnw))
        
        if os.path.exists(originalUnw + '.xml'):
            img_orig = isceobj.createImage()
            img_orig.load(originalUnw + '.xml')
            width_orig = img_orig.width
            length_orig = img_orig.length
            
            # Only resample if dimensions are different
            if width_ion != width_orig or length_ion != length_orig:
                from scipy.interpolate import interp1d
                
                # Resample dispersive phase
                ionos_ml = np.fromfile(outDispersive, dtype=np.float32).reshape(length_ion, width_ion)
                
                # Resample in range direction first (creates intermediate array with length_ion rows and width_orig columns)
                index_rg_ml = np.linspace(0, width_ion-1, num=width_ion, endpoint=True)
                if width_orig != width_ion:
                    index_rg_orig = np.linspace(0, width_orig-1, num=width_orig, endpoint=True) * (width_ion-1)/(width_orig-1) if width_orig > 1 else np.array([0])
                else:
                    index_rg_orig = index_rg_ml
                
                # Intermediate array: resampled in range but not yet in azimuth
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
                ionos_final.astype(np.float32).tofile(outDispersive)
                write_xml(outDispersive, width_orig, length_orig, 1, "FLOAT", "BIL")
                
                # Resample non-dispersive phase
                nonDisp_ml = np.fromfile(outNonDispersive, dtype=np.float32).reshape(length_ion, width_ion)
                
                # Resample in range direction first (creates intermediate array with length_ion rows and width_orig columns)
                nonDisp_resampled_rg = np.zeros((length_ion, width_orig), dtype=np.float32)
                for i in range(length_ion):
                    if width_orig == width_ion:
                        nonDisp_resampled_rg[i, :] = nonDisp_ml[i, :]
                    else:
                        f = interp1d(index_rg_ml, nonDisp_ml[i, :], kind='cubic', fill_value="extrapolate", bounds_error=False)
                        nonDisp_resampled_rg[i, :] = f(index_rg_orig)
                
                # Resample in azimuth direction
                if length_orig != length_ion:
                    nonDisp_final = np.zeros((length_orig, width_orig), dtype=np.float32)
                    for j in range(width_orig):
                        f = interp1d(index_az_ml, nonDisp_resampled_rg[:, j], kind='cubic', fill_value="extrapolate", bounds_error=False)
                        nonDisp_final[:, j] = f(index_az_orig)
                else:
                    nonDisp_final = nonDisp_resampled_rg
                
                # Save resampled non-dispersive phase
                nonDisp_final.astype(np.float32).tofile(outNonDispersive)
                write_xml(outNonDispersive, width_orig, length_orig, 1, "FLOAT", "BIL")
                
                logger.info('Ionospheric phase resampled from {}x{} to {}x{}'.format(width_ion, length_ion, width_orig, length_orig))
                
                del ionos_ml, ionos_resampled_rg, ionos_final, nonDisp_ml, nonDisp_resampled_rg, nonDisp_final
            else:
                logger.info('Ionospheric phase already at original resolution, no resampling needed')
        else:
            logger.warning('Original unwrapped interferogram not found, skipping resampling')

