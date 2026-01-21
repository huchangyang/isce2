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
            
    parser = argparse.ArgumentParser( description='split the range spectrum of SLC')
    parser.add_argument('-s', '--slc', dest='slc', type=str, required=True,
            help='Name of the SLC image or the directory that contains the burst slcs')
    parser.add_argument('-o', '--outDir', dest='outDir', type=str, required=True,
            help='Name of the output directory')
    parser.add_argument('-L', '--dcL', dest='dcL', type=float, default=None,
            help='Low band central frequency [MHz]')
    parser.add_argument('-H', '--dcH', dest='dcH', type=float, default=None,
            help='High band central frequency [MHz]')
    parser.add_argument('-b', '--bwL', dest='bwL', type=float, default=None,
            help='band width of the low-band')
    parser.add_argument('-B', '--bwH', dest='bwH', type=float, default=None,
            help='band width of the high-band')
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

def main(iargs=None):
    '''
    Split the range spectrum
    '''
    #Check if the reference and secondary are .slc files then go ahead and split the range spectrum
    tstart = time.time()
    inps = cmdLineParse(iargs)
    print ('input full-band SLC: ', inps.slc)
    if os.path.isfile(inps.slc):

        
        with shelve.open((inps.shelve), flag='r') as db:
            frame = db['frame']
            try:
              doppler = db['doppler']
            except:
              doppler = None

        radarWavelength = frame.radarWavelegth
        fs = frame.rangeSamplingRate

        pulseLength = frame.instrument.pulseLength
        chirpSlope = frame.instrument.chirpSlope

        #Bandwidth
        totalBandwidth = np.abs(chirpSlope)*pulseLength # Hz


        ###############################################
        if not (inps.dcL and inps.dcH and inps.bwL and inps.bwH):
                # If center frequency and bandwidth of the desired sub-bands are not given,
                # let's choose the one-third of the total bandwidth at the two ends of the 
                # spectrum as low-band and high band
                #pulseLength = frame.instrument.pulseLength
                #chirpSlope = frame.instrument.chirpSlope

                #Bandwidth
                #totalBandwidth = np.abs(chirpSlope)*pulseLength # Hz

                # Dividing the total bandwidth of B to three bands and consider the sub bands on
                # the most left and right hand side as the spectrum of low band and high band SLCs

                # band width of the sub-bands 
                inps.bwL = totalBandwidth/3.0
                inps.bwH = totalBandwidth/3.0
                # center frequency of the low-band
                inps.dcL = -1.0*totalBandwidth/3.0

                # center frequency of the high-band
                inps.dcH = totalBandwidth/3.0

        print("**********************")
        print("Total range bandwidth: ", totalBandwidth)
        print("low-band bandwidth: ", inps.bwL)
        print("high-band bandwidth: ", inps.bwH)
        print("dcL: ", inps.dcL)
        print("dcH: ", inps.dcH)
        print("**********************")

        outDirH = os.path.join(inps.outDir,'HighBand')
        outDirL = os.path.join(inps.outDir,'LowBand')

        os.makedirs(outDirH, exist_ok=True)
        os.makedirs(outDirL, exist_ok=True)

        fullBandSlc = os.path.basename(inps.slc)
        lowBandSlc = os.path.join(outDirL, fullBandSlc)
        highBandSlc = os.path.join(outDirH, fullBandSlc)

        print(inps.slc, lowBandSlc, highBandSlc, fs, inps.bwL, inps.bwH, inps.dcL, inps.dcH)
        print("start")
        split(inps.slc, lowBandSlc, highBandSlc, fs, inps.bwL, inps.bwH, inps.dcL, inps.dcH)
        print("end")
        # Note: rg_filter automatically creates XML and VRT files for output files
        # No need to call createSlcImage() as rg_filter handles this

        f0 = SPEED_OF_LIGHT/radarWavelength
        fH = f0 + inps.dcH
        fL = f0 + inps.dcL
        wavelengthL = SPEED_OF_LIGHT/fL
        wavelengthH = SPEED_OF_LIGHT/fH

        frameH = copy.deepcopy(frame)
        frameH.subBandRadarWavelength = wavelengthH
        frameH.image.filename = highBandSlc
        with shelve.open(os.path.join(outDirH, 'data')) as db:
            db['frame'] = frameH
            if doppler:
               db['doppler'] = doppler  

        frameL = copy.deepcopy(frame)
        frameL.subBandRadarWavelength = wavelengthL
        frameL.image.filename = lowBandSlc
        with shelve.open(os.path.join(outDirL, 'data')) as db:
            db['frame'] = frameL 
            if doppler:
               db['doppler'] = doppler
        
        print ('total processing time: ', time.time()-tstart, ' sec')

   
if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()



