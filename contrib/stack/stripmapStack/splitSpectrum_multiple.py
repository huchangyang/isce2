#!/usr/bin/env python3
#Author: Heresh Fattahi


import numpy as np
import argparse
import os
import isce
import isceobj
import shelve
#import BurstUtils as BU
#from Sentinel1A_TOPS import Sentinel1A_TOPS
#import pyfftw
import copy
import time
#import matplotlib.pyplot as plt
from contrib.alos2proc.alos2proc import rg_filter
from isceobj.Constants import SPEED_OF_LIGHT
from osgeo import gdal


def createParser():
    '''     
    Command line parser.
    '''
            
    parser = argparse.ArgumentParser( description='split the range spectrum of SLC to several sub-bands.')
    parser.add_argument('-s', '--slc', dest='slc', type=str, required=True,
            help='Name of the SLC image or the directory that contains the burst slcs')
    parser.add_argument('-o', '--outDir', dest='outDir', type=str, required=True,
            help='Name of the output directory')
    parser.add_argument('-n', '--number_of_subBands', dest='numberOfSubBands', type=int, default=6,
            help='Number of sub-bands')
    parser.add_argument('-m', '--shelve', dest='shelve', type=str, default=None,
            help='shelve file used to extract metadata')
    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def split(fullBandSlc, lowBandSlc, highBandSlc, fs, bL, bH, fL, fH):
    '''
    Split range spectrum using rg_filter
    
    Parameters:
    -----------
    fullBandSlc: input full-band SLC file (without extension)
    lowBandSlc: output low-band SLC file (without extension)
    highBandSlc: output high-band SLC file (without extension)
    fs: range sampling rate (Hz)
    bL: low-band bandwidth (Hz)
    bH: high-band bandwidth (Hz)
    fL: low-band center frequency (Hz)
    fH: high-band center frequency (Hz)
    '''
    
    # Number of output files
    nout = 2
    
    # Output files
    outputfile = [lowBandSlc, highBandSlc]
    
    # Bandwidth normalized by sampling frequency [0, 1]
    bw = [bL / fs, bH / fs]
    
    # Center frequency normalized by sampling frequency
    bc = [fL / fs, fH / fs]
    
    # rg_filter parameters
    # nfilter: filter length (odd number), using 257 similar to alosStack
    # nfft: FFT length, using 2048 similar to alosStack for better frequency resolution
    # beta: Kaiser window beta, using 0.1 (same as topsStack and alosStack)
    # zero_cf: move center frequency to zero? 0: Yes
    # offset: offset in samples for moving center frequency, using 0.0 for stripmap
    nfilter = 257
    nfft = 2048
    beta = 0.1
    zero_cf = 0
    offset = 0.0
    
    # Call rg_filter
    rg_filter(fullBandSlc, nout, outputfile, bw, bc, 
              nfilter, nfft, beta, zero_cf, offset)

def createSlcImage(slcName, width):

    slc = isceobj.createSlcImage()
    slc.setWidth(width)
    slc.filename = slcName
    slc.setAccessMode('write')
    slc.renderHdr()

def getShape(fileName):

    dataset = gdal.Open(fileName,gdal.GA_ReadOnly)
    return dataset.RasterYSize, dataset.RasterXSize


def extractSubBands(slc, frame, dcL, dcH, bw, LowBand, HighBand, width, outDir):

    radarWavelength = frame.radarWavelegth
    fs = frame.rangeSamplingRate

    outDirH = os.path.join(outDir, HighBand)
    outDirL = os.path.join(outDir, LowBand)
    os.makedirs(outDirH, exist_ok=True)
    os.makedirs(outDirL, exist_ok=True)

    fullBandSlc = os.path.basename(slc)
    lowBandSlc = os.path.join(outDirL, fullBandSlc)
    highBandSlc = os.path.join(outDirH, fullBandSlc)

    split(slc, lowBandSlc, highBandSlc, fs, bw, bw, dcL, dcH)
    # Note: rg_filter automatically creates XML and VRT files for output files
    # No need to call createSlcImage() as rg_filter handles this

    '''
    f0 = SPEED_OF_LIGHT/radarWavelength
    fH = f0 + dcH
    fL = f0 + dcL
    wavelengthL = SPEED_OF_LIGHT/fL
    wavelengthH = SPEED_OF_LIGHT/fH

    frameH = copy.deepcopy(frame)
    frameH.subBandRadarWavelength = wavelengthH
    frameH.image.filename = highBandSlc
    with shelve.open(os.path.join(outDirH, 'data')) as db:
        db['frame'] = frameH

    frameL = copy.deepcopy(frame)
    frameL.subBandRadarWavelength = wavelengthL
    frameL.image.filename = lowBandSlc
    with shelve.open(os.path.join(outDirL, 'data')) as db:
        db['frame'] = frameL 
    '''

def main(iargs=None):
    '''
    Split the range spectrum
    '''
    #Check if the reference and secondary are .slc files then go ahead and split the range spectrum

    tstart = time.time()
    inps = cmdLineParse(iargs)

    length, width = getShape(inps.slc + ".vrt")

    with shelve.open((inps.shelve), flag='r') as db:
            frame = db['frame']

    radarWavelength = frame.radarWavelegth
    pulseLength = frame.instrument.pulseLength
    chirpSlope = frame.instrument.chirpSlope

                #Bandwidth
    totalBandwidth = np.abs(chirpSlope)*pulseLength # Hz

    Nf = inps.numberOfSubBands
    if Nf < 2:
       raise Exception("number of sub-bands should be larger than 1")

 
    if Nf%2 == 1:
        print("number of subbands ({0}) is odd. Currently only even number of sub-bands is supported".format(Nf))
        Nf = Nf - 1
        print("modifying number of subbands to : {0}".format(Nf))
	
    bw = totalBandwidth/Nf
    print("total bandwidth: ", totalBandwidth, " Hz")
    print("band width of sub-bands: ", bw, " Hz")
    ii = int(Nf/2) 
    jj = int(Nf/2) + 1 
    frequency = []
    for i in range(1,Nf,2):
        dcL = -i*bw/2.0
        dcH = i*bw/2.0
        LowBand = "f_" + str(ii)
        HighBand = "f_" + str(jj)
        print("LowBand ", LowBand, " , " , dcL)
        print("HighBand ", HighBand, " , " , dcH)

        f0 = SPEED_OF_LIGHT/radarWavelength
        fH = f0 + dcH
        fL = f0 + dcL

        frequency.append(fL)
        frequency.append(fH)

        extractSubBands(inps.slc, frame, dcL, dcH, bw, LowBand, HighBand, width, inps.outDir)

        wavelengthL = SPEED_OF_LIGHT/fL
        wavelengthH = SPEED_OF_LIGHT/fH
        print("*****************")
        print("fL: ", fL, " Hz")
        print("fH: ", fH, " Hz")
        print("*****************")
        outDirH = os.path.join(inps.outDir, HighBand)
        outDirL = os.path.join(inps.outDir, LowBand)

        fullBandSlc = os.path.basename(inps.slc)
        lowBandSlc = os.path.join(outDirL, fullBandSlc)
        highBandSlc = os.path.join(outDirH, fullBandSlc)

        frameH = copy.deepcopy(frame)
        frameH.subBandRadarWavelength = wavelengthH
        frameH.image.filename = highBandSlc
        with shelve.open(os.path.join(outDirH, 'data')) as db:
            db['frame'] = frameH

        frameL = copy.deepcopy(frame)
        frameL.subBandRadarWavelength = wavelengthL
        frameL.image.filename = lowBandSlc
        with shelve.open(os.path.join(outDirL, 'data')) as db:
            db['frame'] = frameL

        ii = ii - 1
        jj = jj + 1
    
    print("frequencies:", frequency.sort)
    print("frequency difference between first and last sub bands: ", (np.max(frequency) - np.min(frequency))/10**6, " MHz") 


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()



