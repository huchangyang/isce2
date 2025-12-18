#

import isce
import isceobj
from iscesys.StdOEL.StdOELPy import create_writer
from mroipac.ampcor.Ampcor import Ampcor

import numpy as np
import os
import shelve
import logging

logger = logging.getLogger('isce.insar.runRefineSecondaryTiming')


def estimateOffsetField(reference, secondary, azoffset=0, rgoffset=0):
    '''
    Estimate offset field between burst and simamp.
    '''


    sim = isceobj.createSlcImage()
    sim.load(secondary+'.xml')
    sim.setAccessMode('READ')
    sim.createImage()

    sar = isceobj.createSlcImage()
    sar.load(reference + '.xml')
    sar.setAccessMode('READ')
    sar.createImage()

    width = sar.getWidth()
    length = sar.getLength()

    objOffset = Ampcor(name='reference_offset1')
    objOffset.configure()
    objOffset.setAcrossGrossOffset(rgoffset)
    objOffset.setDownGrossOffset(azoffset)
    objOffset.setWindowSizeWidth(128)
    objOffset.setWindowSizeHeight(128)
    objOffset.setSearchWindowSizeWidth(40)
    objOffset.setSearchWindowSizeHeight(40)
    margin = 2*objOffset.searchWindowSizeWidth + objOffset.windowSizeWidth

    nAcross = 60
    nDown = 60


    offAc = max(101,-rgoffset)+margin
    offDn = max(101,-azoffset)+margin


    lastAc = int( min(width, sim.getWidth() - offAc) - margin)
    lastDn = int( min(length, sim.getLength() - offDn) - margin)

    if not objOffset.firstSampleAcross:
        objOffset.setFirstSampleAcross(offAc)

    if not objOffset.lastSampleAcross:
        objOffset.setLastSampleAcross(lastAc)

    if not objOffset.firstSampleDown:
        objOffset.setFirstSampleDown(offDn)

    if not objOffset.lastSampleDown:
        objOffset.setLastSampleDown(lastDn)

    if not objOffset.numberLocationAcross:
        objOffset.setNumberLocationAcross(nAcross)

    if not objOffset.numberLocationDown:
        objOffset.setNumberLocationDown(nDown)

    objOffset.setFirstPRF(1.0)
    objOffset.setSecondPRF(1.0)
    objOffset.setImageDataType1('complex')
    objOffset.setImageDataType2('complex')

    objOffset.ampcor(sar, sim)

    sar.finalizeImage()
    sim.finalizeImage()

    result = objOffset.getOffsetField()
    return result


def fitOffsets(field,azrgOrder=0,azazOrder=0,
        rgrgOrder=0,rgazOrder=0,snr=5.0):
    '''
    Estimate constant range and azimuth shifts.
    '''

    # Keep a copy of the original field so that we can access the original
    # per-point covariance / standard deviation information (sigmax, sigmay)
    # after Offoutliers has created a refined subset without covariance.
    originalField = field

    stdWriter = create_writer("log","",True,filename='off.log')

    for distance in [10,5,3,1]:
        inpts = len(field._offsets)
        print("DEBUG %%%%%%%%")
        print(inpts)
        print("DEBUG %%%%%%%%")
        objOff = isceobj.createOffoutliers()
        objOff.wireInputPort(name='offsets', object=field)
        objOff.setSNRThreshold(snr)
        objOff.setDistance(distance)
        objOff.setStdWriter(stdWriter)

        objOff.offoutliers()

        field = objOff.getRefinedOffsetField()
        outputs = len(field._offsets)

        print('%d points left'%(len(field._offsets)))

    # ------------------------------------------------------------------
    # Additional culling based on per-point standard deviation (sigma)
    # stored in the original ampcor offsets as sigmax / sigmay.
    # Threshold is fixed at 0.001 (in pixel units).
    # ------------------------------------------------------------------
    sigmaThreshold = 0.001
    print('Applying sigma threshold: {:.4f}'.format(sigmaThreshold))

    # Build a lookup from original offsets using (x, y) as key, where
    # x = range location, y = azimuth location. We use the string
    # representation to avoid floating point rounding issues.
    originalOffsetMap = {}
    for offsetx in originalField:
        fields = "{}".format(offsetx).split()
        if len(fields) >= 8:
            key = (fields[0], fields[2])  # x, y
            originalOffsetMap[key] = fields

    filtered_offsets = []
    removedSigma = 0
    for offsetx in field:
        fields = "{}".format(offsetx).split()
        if len(fields) < 4:
            # Malformed entry, drop it
            removedSigma += 1
            continue

        key = (fields[0], fields[2])
        orig_fields = originalOffsetMap.get(key, None)
        if (orig_fields is None) or (len(orig_fields) < 8):
            # Cannot recover covariance info, drop this point
            removedSigma += 1
            continue

        sigma_rg = float(orig_fields[5])  # sigmax
        sigma_az = float(orig_fields[6])  # sigmay

        if (abs(sigma_rg) > sigmaThreshold) or (abs(sigma_az) > sigmaThreshold):
            removedSigma += 1
            continue

        filtered_offsets.append(offsetx)

    print('%d points left after sigma culling (removed %d points with sigma > %.4f)' %
          (len(filtered_offsets), removedSigma, sigmaThreshold))

    # Replace the internal list with the sigma-filtered subset so that the
    # subsequent polynomial fit only uses high-quality points.
    field._offsets = filtered_offsets

    aa, dummy = field.getFitPolynomials(azimuthOrder=azazOrder, rangeOrder=azrgOrder, usenumpy=True)
    dummy, rr = field.getFitPolynomials(azimuthOrder=rgazOrder, rangeOrder=rgrgOrder, usenumpy=True)

    azshift = aa._coeffs[0][0]
    rgshift = rr._coeffs[0][0]
    print('Estimated az shift: ', azshift)
    print('Estimated rg shift: ', rgshift)

    return (aa, rr), field


def runRefineSecondaryTiming(self):

    logger.info("Running refine secondary timing") 
    secondaryFrame = self._insar.loadProduct( self._insar.secondarySlcCropProduct)
    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)
    referenceSlc = referenceFrame.getImage().filename

    slvImg = secondaryFrame.getImage()
    secondarySlc = os.path.join(self.insar.coregDirname , self._insar.coarseCoregFilename)
    
    field = estimateOffsetField(referenceSlc, secondarySlc)

    rgratio = referenceFrame.instrument.getRangePixelSize()/secondaryFrame.instrument.getRangePixelSize()
    azratio = secondaryFrame.PRF / referenceFrame.PRF 

    print ('*************************************')
    print ('rgratio, azratio: ', rgratio, azratio)
    print ('*************************************')

    misregDir = self.insar.misregDirname
    os.makedirs(misregDir, exist_ok=True)
 
    outShelveFile = os.path.join(misregDir, self.insar.misregFilename)
    odb = shelve.open(outShelveFile)
    odb['raw_field']  = field
    shifts, cull = fitOffsets(field,azazOrder=0,
            azrgOrder=0,
            rgazOrder=0,
            rgrgOrder=0,
            snr=5.0)
    odb['cull_field'] = cull

    ####Scale by ratio
    for row in shifts[0]._coeffs:
        for ind, val in  enumerate(row):
            row[ind] = val * azratio

    for row in shifts[1]._coeffs:
        for ind, val in enumerate(row):
            row[ind] = val * rgratio


    odb['azpoly'] = shifts[0]
    odb['rgpoly'] = shifts[1]
    odb.close()     

    
    self._insar.saveProduct(shifts[0], outShelveFile + '_az.xml')
    self._insar.saveProduct(shifts[1], outShelveFile + '_rg.xml')
    
    return None








