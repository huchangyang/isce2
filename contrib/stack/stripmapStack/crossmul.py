#!/usr/bin/env python3


import os
import argparse
import logging

import isce
import isceobj
from components.stdproc.stdproc import crossmul
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
import numpy as np


def createParser():

    '''
    Command Line Parser.
    '''
    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-m', '--reference', type=str, dest='reference', required=True,
            help='Reference image')
    parser.add_argument('-s', '--secondary', type=str, dest='secondary', required=True,
            help='Secondary image')
    parser.add_argument('-o', '--outdir', type=str, dest='prefix', default='crossmul',
            help='Prefix of output int and amp files')
    parser.add_argument('-a', '--alks', type=int, dest='azlooks', default=1,
            help='Azimuth looks')
    parser.add_argument('-r', '--rlks', type=int, dest='rglooks', default=1,
            help='Range looks')
    parser.add_argument('--mask_invalid', action='store_true', dest='maskInvalid', default=False,
            help='Mask invalid phase in non-overlapping regions (set to zero)')
    
    # Additional looks for ionosphere estimation
    parser.add_argument('--number_range_looks_ion', dest='numberRangeLooksIon', type=int, default=None,
            help='Additional range looks for ionosphere estimation (default: 16)')
    parser.add_argument('--number_azimuth_looks_ion', dest='numberAzimuthLooksIon', type=int, default=None,
            help='Additional azimuth looks for ionosphere estimation (default: 16)')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

def _buildSlcInvalidMask(slcFilename, width, length, azLooks, rgLooks, effectiveSlcLength, logger):
    """
    Build a boolean mask (True=invalid) at interferogram resolution from one SLC.
    Returns None if the mask cannot be computed from this SLC.
    """
    if slcFilename is None or not os.path.exists(slcFilename + '.xml'):
        if slcFilename is not None:
            logger.warning('SLC file not found: {}. Skipping SLC-based masking for this image.'.format(
                slcFilename))
        return None

    slcImg = isceobj.createSlcImage()
    slcImg.load(slcFilename + '.xml')
    slcWidth = slcImg.getWidth()
    slcLength = slcImg.getLength()

    if slcWidth == width and slcLength == length:
        slc = np.memmap(slcFilename, dtype=np.complex64, mode='r', shape=(length, width))
        invalidMask = (np.abs(slc) == 0.0) | ~np.isfinite(slc)
        del slc
        return invalidMask

    if azLooks > 1 or rgLooks > 1:
        logger.info('SLC dimensions ({0}x{1}) do not match interferogram dimensions ({2}x{3}). '
                   'Applying multilooking to SLC {4} (azLooks={5}, rgLooks={6}).'.format(
                   slcWidth, slcLength, width, length, slcFilename, azLooks, rgLooks))

        slc = np.memmap(slcFilename, dtype=np.complex64, mode='r', shape=(slcLength, slcWidth))

        if effectiveSlcLength is not None:
            slcLines = min(effectiveSlcLength, slcLength)
        else:
            slcLines = slcLength

        expectedWidth = int(slcWidth / rgLooks)
        expectedLength = int(slcLines / azLooks)

        if expectedWidth == width and expectedLength == length:
            slcMl = np.zeros((length, width), dtype=np.complex64)
            for i in range(length):
                for j in range(width):
                    i_start = i * azLooks
                    i_end = min((i + 1) * azLooks, slcLines)
                    j_start = j * rgLooks
                    j_end = min((j + 1) * rgLooks, slcWidth)

                    window = slc[i_start:i_end, j_start:j_end]
                    if window.size > 0:
                        slcMl[i, j] = np.mean(window)
                    else:
                        slcMl[i, j] = 0.0

            invalidMask = (np.abs(slcMl) == 0.0) | ~np.isfinite(slcMl)
            del slc, slcMl
            return invalidMask

        logger.warning('Expected multilooked SLC dimensions ({0}x{1}) do not match interferogram dimensions ({2}x{3}) '
                      'for {4}.'.format(expectedWidth, expectedLength, width, length, slcFilename))
        del slc
        return None

    logger.warning('SLC dimensions ({0}x{1}) do not match interferogram dimensions ({2}x{3}) for {4} '
                  'and no looks were provided.'.format(slcWidth, slcLength, width, length, slcFilename))
    return None


def maskInvalidPhase(intFilename, ampFilename, secondarySlcFilename=None, referenceSlcFilename=None,
                     ampThreshold=1e-6, azLooks=1, rgLooks=1, referenceSlcLength=None):
    """
    Mask invalid phase regions by setting them to zero.
    
    Parameters:
    intFilename: Path to interferogram file
    ampFilename: Path to amplitude file (not used, kept for backward compatibility)
    secondarySlcFilename: Path to secondary SLC file (optional, for phase-based masking)
    referenceSlcFilename: Path to reference SLC file (optional, for phase-based masking)
    ampThreshold: Not used (kept for backward compatibility)
    azLooks: Azimuth looks applied to interferogram (default: 1)
    rgLooks: Range looks applied to interferogram (default: 1)
    referenceSlcLength: Effective SLC line count used in crossmul (min of both lengths)
    
    Logic:
    Check reference and secondary SLC amplitudes to determine invalid pixels.
    If either SLC is zero (phase is undefined/invalid), set interferogram phase to zero.
    This masks out non-overlapping regions and any pixel without valid data in both images.
    """
    logger = logging.getLogger('isce.stack.crossmul')
    
    # Read dimensions from XML file
    intImg = isceobj.createIntImage()
    intImg.load(intFilename + '.xml')
    width = intImg.getWidth()
    length = intImg.getLength()
    
    # Read interferogram
    intf = np.memmap(intFilename, dtype=np.complex64, mode='r+', shape=(length, width))

    slcMasks = []
    for slcFilename in (referenceSlcFilename, secondarySlcFilename):
        slcMask = _buildSlcInvalidMask(
            slcFilename, width, length, azLooks, rgLooks, referenceSlcLength, logger)
        if slcMask is not None:
            slcMasks.append(slcMask)

    if slcMasks:
        invalidMask = slcMasks[0]
        for slcMask in slcMasks[1:]:
            invalidMask = invalidMask | slcMask
    else:
        logger.warning('Could not build SLC-based invalid mask; falling back to interferogram-based masking.')
        intfAmp = np.abs(intf)
        invalidMask = (intfAmp == 0.0) | ~np.isfinite(intfAmp)
    
    # Count invalid pixels
    nInvalid = np.sum(invalidMask)
    if nInvalid > 0:
        logger.info('Masking {0} invalid pixels ({1:.2f}%) with zero phase (reference or secondary SLC invalid)'.format(
            nInvalid, 100.0 * nInvalid / invalidMask.size))
        
        # Set invalid regions to zero
        intf[invalidMask] = 0.0 + 0.0j
        
        # Flush to disk
        intf.flush()
    
    del intf
    
    return nInvalid


def multilook_int_amp(intFile, ampFile, outIntFile, outAmpFile, azLooks, rgLooks):
    """
    Multilook interferogram (.int) and amplitude (.amp) files
    """
    from mroipac.looks.Looks import Looks
    
    # Multilook interferogram
    intImg = isceobj.createIntImage()
    intImg.load(intFile + '.xml')
    intImg.filename = intFile
    
    lkObj = Looks()
    lkObj.setDownLooks(azLooks)
    lkObj.setAcrossLooks(rgLooks)
    lkObj.setInputImage(intImg)
    lkObj.setOutputFilename(outIntFile)
    lkObj.looks()
    
    # Multilook amplitude
    ampImg = isceobj.createAmpImage()
    ampImg.load(ampFile + '.xml')
    ampImg.filename = ampFile
    
    lkObjAmp = Looks()
    lkObjAmp.setDownLooks(azLooks)
    lkObjAmp.setAcrossLooks(rgLooks)
    lkObjAmp.setInputImage(ampImg)
    lkObjAmp.setOutputFilename(outAmpFile)
    lkObjAmp.looks()
    
    logger = logging.getLogger('isce.stack.crossmul')
    logger.info('Multilooked interferogram: {} -> {} ({}x{} looks)'.format(
        intFile, outIntFile, azLooks, rgLooks))
    logger.info('Multilooked amplitude: {} -> {} ({}x{} looks)'.format(
        ampFile, outAmpFile, azLooks, rgLooks))


def run(imageSlc1, imageSlc2, resampName, azLooks, rgLooks, maskInvalid=False, 
        numberRangeLooksIon=None, numberAzimuthLooksIon=None):
    objSlc1 = isceobj.createSlcImage()
    #right now imageSlc1 and 2 are just text files, need to open them as image

    IU.copyAttributes(imageSlc1, objSlc1)
    objSlc1.setAccessMode('read')
    objSlc1.createImage()

    objSlc2 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc2, objSlc2)
    objSlc2.setAccessMode('read')
    objSlc2.createImage()

    slcWidth = imageSlc1.getWidth()
    intWidth = int(slcWidth / rgLooks)

    lines = min(imageSlc1.getLength(), imageSlc2.getLength())

    resampAmp = resampName + '.amp'
    resampInt = resampName + '.int'

    objInt = isceobj.createIntImage()
    objInt.setFilename(resampInt)
    objInt.setWidth(intWidth)
    imageInt = isceobj.createIntImage()
    IU.copyAttributes(objInt, imageInt)
    objInt.setAccessMode('write')
    objInt.createImage()

    objAmp = isceobj.createAmpImage()
    objAmp.setFilename(resampAmp)
    objAmp.setWidth(intWidth)
    imageAmp = isceobj.createAmpImage()
    IU.copyAttributes(objAmp, imageAmp)
    objAmp.setAccessMode('write')
    objAmp.createImage()

    objCrossmul = crossmul.createcrossmul()
    objCrossmul.width = slcWidth
    objCrossmul.length = lines
    objCrossmul.LooksDown = azLooks
    objCrossmul.LooksAcross = rgLooks

    objCrossmul.crossmul(objSlc1, objSlc2, objInt, objAmp)

    # Mask invalid phase in non-overlapping regions if requested
    if maskInvalid:
        # Ensure images are finalized before masking (crossmul already finalizes, but ensure XML exists)
        objInt.finalizeImage()
        objAmp.finalizeImage()
        
        # Mask pixels where either reference or secondary SLC is invalid
        referenceSlcFilename = objSlc1.getFilename()
        secondarySlcFilename = objSlc2.getFilename()
        effectiveSlcLength = min(imageSlc1.getLength(), imageSlc2.getLength())
        maskInvalidPhase(resampInt, resampAmp,
                        referenceSlcFilename=referenceSlcFilename,
                        secondarySlcFilename=secondarySlcFilename,
                        ampThreshold=1e-6, azLooks=azLooks, rgLooks=rgLooks,
                        referenceSlcLength=effectiveSlcLength)
    else:
        # Finalize images normally
        for obj in [objInt, objAmp]:
            obj.finalizeImage()

    for obj in [objSlc1, objSlc2]:
        obj.finalizeImage()

    # Apply additional multilooking for ionosphere estimation if requested
    # Only apply if parameters are explicitly provided (not None)
    # For full-band interferograms, these should be None and no additional multilooking should be applied
    if numberRangeLooksIon is not None and numberAzimuthLooksIon is not None:
        if numberRangeLooksIon > 1 or numberAzimuthLooksIon > 1:
            # Create multilooked filenames
            ml2 = '_{}rlks_{}alks'.format(numberRangeLooksIon, numberAzimuthLooksIon)
            resampIntMl = resampInt.replace('.int', ml2 + '.int')
            resampAmpMl = resampAmp.replace('.amp', ml2 + '.amp')
            
            logger = logging.getLogger('isce.stack.crossmul')
            logger.info('Applying additional multilooking for ionosphere estimation ({}x{} looks)'.format(
                numberAzimuthLooksIon, numberRangeLooksIon))
            multilook_int_amp(resampInt, resampAmp, resampIntMl, resampAmpMl, 
                             numberAzimuthLooksIon, numberRangeLooksIon)

    return imageInt, imageAmp


def main(iargs=None):
    inps = cmdLineParse(iargs)

    img1 = isceobj.createImage()
    img1.load(inps.reference + '.xml')

    img2 = isceobj.createImage()
    img2.load(inps.secondary + '.xml')

    os.makedirs(os.path.dirname(inps.prefix), exist_ok=True)

    run(img1, img2, inps.prefix, inps.azlooks, inps.rglooks, maskInvalid=inps.maskInvalid,
        numberRangeLooksIon=getattr(inps, 'numberRangeLooksIon', None),
        numberAzimuthLooksIon=getattr(inps, 'numberAzimuthLooksIon', None))

if __name__ == '__main__':

    main()
    '''
    Main driver.
    '''
