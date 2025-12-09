
#
# Author: Heresh Fattahi, 2017
# Modified by V. Brancato (10.2019)
#         (Included flattening when rubbersheeting in range is turned on

import isceobj
import logging
from components.stdproc.stdproc import crossmul
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
import os
from osgeo import gdal
import numpy as np

logger = logging.getLogger('isce.insar.runInterferogram')

# Added by V. Brancato 10.09.2019
def write_xml(fileName,width,length,bands,dataType,scheme):
    import os

    img = isceobj.createImage()
    img.setFilename(fileName)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('READ')
    img.bands = bands
    img.dataType = dataType
    img.scheme = scheme
    # renderHdr() creates fileName + '.xml' (e.g., 'lower_32rlks_64alks.int.xml')
    img.renderHdr()
    img.renderVRT()
    
    # For compatibility with computeCoherence which expects fileName without .int extension + '.xml'
    # Create a symlink if the filename contains .int and we need the .xml version
    xml_file = fileName + '.xml'  # Created by renderHdr (e.g., 'lower_32rlks_64alks.int.xml')
    if '.int' in fileName and xml_file.endswith('.int.xml'):
        # Create symlink without .int for computeCoherence compatibility
        # e.g., 'lower_32rlks_64alks.int.xml' -> 'lower_32rlks_64alks.xml'
        alt_xml = fileName.replace('.int', '') + '.xml'
        if not os.path.exists(alt_xml):
            try:
                os.symlink(os.path.basename(xml_file), alt_xml)
            except OSError:
                # If symlink fails (e.g., on Windows or permission issue), try copy
                import shutil
                shutil.copy2(xml_file, alt_xml)
    
    return None

def maskInvalidPhase(intFilename, ampFilename, secondarySlcFilename=None, ampThreshold=1e-6):
    """
    Mask invalid phase regions by setting them to zero.
    
    Parameters:
    intFilename: Path to interferogram file
    ampFilename: Path to amplitude file (not used, kept for backward compatibility)
    secondarySlcFilename: Path to secondary SLC file (optional, for phase-based masking)
    ampThreshold: Not used (kept for backward compatibility)
    
    Logic:
    Check secondary SLC phase to determine invalid pixels.
    If secondary SLC is zero (phase is undefined/invalid), set interferogram phase to zero.
    This masks out non-overlapping regions where secondary image has no data.
    """
    import numpy as np
    
    # Read dimensions from XML file
    intImg = isceobj.createIntImage()
    intImg.load(intFilename + '.xml')
    width = intImg.getWidth()
    length = intImg.getLength()
    
    # Read interferogram
    intf = np.memmap(intFilename, dtype=np.complex64, mode='r+', shape=(length, width))
    
    # Check secondary SLC phase if secondary SLC file is provided
    if secondarySlcFilename is not None and os.path.exists(secondarySlcFilename + '.xml'):
        # Read secondary SLC
        secondarySlcImg = isceobj.createSlcImage()
        secondarySlcImg.load(secondarySlcFilename + '.xml')
        slcWidth = secondarySlcImg.getWidth()
        slcLength = secondarySlcImg.getLength()
        
        # If dimensions match, check secondary SLC phase
        if slcWidth == width and slcLength == length:
            secondarySlc = np.memmap(secondarySlcFilename, dtype=np.complex64, mode='r', shape=(length, width))
            
            # Check if secondary SLC is zero (phase is undefined/invalid)
            # If secondary SLC is zero, the phase is invalid, so mask the interferogram phase
            invalidMask = (np.abs(secondarySlc) == 0.0) | ~np.isfinite(secondarySlc)
            
            del secondarySlc
        else:
            # Dimensions don't match, fall back to checking interferogram
            logger.warning('Secondary SLC dimensions ({0}x{1}) do not match interferogram dimensions ({2}x{3}). '
                          'Falling back to interferogram-based masking.'.format(
                          slcWidth, slcLength, width, length))
            intfAmp = np.abs(intf)
            invalidMask = (intfAmp == 0.0) | ~np.isfinite(intfAmp)
    else:
        # No secondary SLC provided, fall back to checking interferogram
        if secondarySlcFilename is not None:
            logger.warning('Secondary SLC file not found: {}. Falling back to interferogram-based masking.'.format(
                secondarySlcFilename))
        intfAmp = np.abs(intf)
        invalidMask = (intfAmp == 0.0) | ~np.isfinite(intfAmp)
    
    # Count invalid pixels
    nInvalid = np.sum(invalidMask)
    if nInvalid > 0:
        logger.info('Masking {0} invalid pixels ({1:.2f}%) with zero phase (based on secondary SLC phase)'.format(
            nInvalid, 100.0 * nInvalid / invalidMask.size))
        
        # Set invalid regions to zero
        intf[invalidMask] = 0.0 + 0.0j
        
        # Flush to disk
        intf.flush()
    
    del intf
    
    return nInvalid

    	    
def compute_FlatEarth(self,ifgFilename,width,length,radarWavelength):
    from imageMath import IML
    import logging
    
    # If rubbersheeting has been performed add back the range sheet offsets
    
    info = self._insar.loadProduct(self._insar.secondarySlcCropProduct)
    #radarWavelength = info.getInstrument().getRadarWavelength() 
    rangePixelSize = info.getInstrument().getRangePixelSize()
    fact = 4 * np.pi* rangePixelSize / radarWavelength

    cJ = np.complex64(-1j)

    # Open the range sheet offset
    rngOff = os.path.join(self.insar.offsetsDirname, self.insar.rangeOffsetFilename )
    
    print(rngOff)
    if os.path.exists(rngOff):
       rng2 = np.memmap(rngOff, dtype=np.float64, mode='r', shape=(length,width))
    else:
       print('No range offsets provided')
       rng2 = np.zeros((length,width))
    
    # Open the interferogram
    #ifgFilename= os.path.join(self.insar.ifgDirname, self.insar.ifgFilename)
    intf = np.memmap(ifgFilename,dtype=np.complex64,mode='r+',shape=(length,width))
   
    for ll in range(length):
        intf[ll,:] *= np.exp(cJ*fact*rng2[ll,:])
    
    del rng2
    del intf
       
    return 
    
    

def multilook(infile, outname=None, alks=5, rlks=15):
    '''
    Take looks.
    '''

    from mroipac.looks.Looks import Looks

    print('Multilooking {0} ...'.format(infile))

    inimg = isceobj.createImage()
    inimg.load(infile + '.xml')
    
    # Ensure filename is absolute path (critical when current directory changes)
    import os
    if not os.path.isabs(inimg.filename):
        inimg.filename = os.path.abspath(infile)
    else:
        # Even if absolute, ensure it points to the correct file
        inimg.filename = os.path.abspath(infile)

    if outname is None:
        spl = os.path.splitext(inimg.filename)
        ext = '.{0}alks_{1}rlks'.format(alks, rlks)
        outname = spl[0] + ext + spl[1]

    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(inimg)
    lkObj.setOutputFilename(outname)
    lkObj.looks()

    return outname

def computeCoherence(slc1name, slc2name, corname, virtual=True):
    from mroipac.correlation.correlation import Correlation
    import os
                          
    slc1 = isceobj.createImage()
    # Try .int.xml first (created by multilook), then .xml
    xml1 = slc1name + '.int.xml'
    if not os.path.exists(xml1):
        xml1 = slc1name + '.xml'
    slc1.load(xml1)
    slc1.createImage()


    slc2 = isceobj.createImage()
    # Try .int.xml first (created by multilook), then .xml
    xml2 = slc2name + '.int.xml'
    if not os.path.exists(xml2):
        xml2 = slc2name + '.xml'
    slc2.load(xml2)
    slc2.createImage()

    cohImage = isceobj.createOffsetImage()
    cohImage.setFilename(corname)
    cohImage.setWidth(slc1.getWidth())
    cohImage.setAccessMode('write')
    cohImage.createImage()

    cor = Correlation()
    cor.configure()
    cor.wireInputPort(name='slc1', object=slc1)
    cor.wireInputPort(name='slc2', object=slc2)
    cor.wireOutputPort(name='correlation', object=cohImage)
    cor.coregisteredSlcFlag = True
    cor.calculateCorrelation()

    cohImage.finalizeImage()
    slc1.finalizeImage()
    slc2.finalizeImage()
    return

# Modified by V. Brancato on 10.09.2019 (added self)
# Modified by V. Brancato on 11.13.2019 (added radar wavelength for low and high band flattening
def generateIgram(self,imageSlc1, imageSlc2, resampName, azLooks, rgLooks,radarWavelength):
    objSlc1 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc1, objSlc1)
    objSlc1.setAccessMode('read')
    objSlc1.createImage()

    objSlc2 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc2, objSlc2)
    objSlc2.setAccessMode('read')
    objSlc2.createImage()

    slcWidth = imageSlc1.getWidth()
    
    
    if not self.doRubbersheetingRange:
     intWidth = int(slcWidth/rgLooks)    # Modified by V. Brancato intWidth = int(slcWidth / rgLooks)
    else:
     intWidth = int(slcWidth)
    
    lines = min(imageSlc1.getLength(), imageSlc2.getLength())

    if '.flat' in resampName:
        resampAmp = resampName.replace('.flat', '.amp')
    elif '.int' in resampName:
        resampAmp = resampName.replace('.int', '.amp')
    else:
        resampAmp += '.amp'

    if not self.doRubbersheetingRange:
        resampInt = resampName
    else:
        resampInt = resampName + ".full"

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
    
    if not self.doRubbersheetingRange:
       print('Rubbersheeting in range is off, interferogram is already flattened')
       objCrossmul = crossmul.createcrossmul()
       objCrossmul.width = slcWidth
       objCrossmul.length = lines
       objCrossmul.LooksDown = azLooks
       objCrossmul.LooksAcross = rgLooks

       objCrossmul.crossmul(objSlc1, objSlc2, objInt, objAmp)
       
       # Ensure images are finalized before masking (crossmul already finalizes, but ensure XML exists)
       # crossmul.crossmul() already calls finalizeImage() and renderHdr(), so XML should exist
       # But we need to ensure the files are accessible
       
       # Mask invalid phase in non-overlapping regions (where secondary has no data)
       # Use secondary SLC phase to determine invalid regions
       secondarySlcFilename = objSlc2.getFilename()
       maskInvalidPhase(resampInt, resampAmp, secondarySlcFilename=secondarySlcFilename, ampThreshold=1e-6)
    else:
     # Modified by V. Brancato 10.09.2019 (added option to add Range Rubber sheet Flat-earth back)
       print('Rubbersheeting in range is on, removing flat-Earth phase')
       objCrossmul = crossmul.createcrossmul()
       objCrossmul.width = slcWidth
       objCrossmul.length = lines
       objCrossmul.LooksDown = 1
       objCrossmul.LooksAcross = 1
       objCrossmul.crossmul(objSlc1, objSlc2, objInt, objAmp)
       
       # Ensure images are finalized before masking
       objInt.finalizeImage()
       objAmp.finalizeImage()
       
       # Mask invalid phase in non-overlapping regions (where secondary has no data)
       # Use secondary SLC phase to determine invalid regions
       secondarySlcFilename = objSlc2.getFilename()
       maskInvalidPhase(resampInt, resampAmp, secondarySlcFilename=secondarySlcFilename, ampThreshold=1e-6)
       
       # Remove Flat-Earth component
       compute_FlatEarth(self,resampInt,intWidth,lines,radarWavelength)
       
       # Perform Multilook
       multilook(resampInt, outname=resampName, alks=azLooks, rlks=rgLooks)  #takeLooks(objAmp,azLooks,rgLooks)
       multilook(resampAmp, outname=resampAmp.replace(".full",""), alks=azLooks, rlks=rgLooks)  #takeLooks(objInt,azLooks,rgLooks)
       
       # Mask invalid phase again after multilooking
       # For multilooked interferogram, we may not have corresponding multilooked SLC, so use None
       # The function will fall back to interferogram-based masking
       maskInvalidPhase(resampName, resampAmp.replace(".full",""), secondarySlcFilename=None, ampThreshold=1e-6)
       
       #os.system('rm ' + resampInt+'.full* ' + resampAmp + '.full* ')
       # End of modification 
    # Note: objInt and objAmp are finalized by crossmul.crossmul() and/or above for masking
    # Only need to finalize SLC images
    for obj in [objSlc1, objSlc2]:
        obj.finalizeImage()

    return imageInt, imageAmp


def subBandIgram(self, referenceSlc, secondarySlc, subBandDir,radarWavelength):

    img1 = isceobj.createImage()
    img1.load(referenceSlc + '.xml')

    img2 = isceobj.createImage()
    img2.load(secondarySlc + '.xml')

    azLooks = self.numberAzimuthLooks
    rgLooks = self.numberRangeLooks

    ifgDir = os.path.join(self.insar.ifgDirname, subBandDir)

    os.makedirs(ifgDir, exist_ok=True)

    interferogramName = os.path.join(ifgDir , self.insar.ifgFilename)

    generateIgram(self,img1, img2, interferogramName, azLooks, rgLooks,radarWavelength)
    
    # If ionospheric looks are specified, create additional multilooked version
    # In runInterferogram, self is the Insar instance, so get parameters from self
    # Also try self.insar as fallback (for direct StripmapProc usage)
    numberRangeLooksIon = getattr(self, 'numberRangeLooksIon', None)
    numberAzimuthLooksIon = getattr(self, 'numberAzimuthLooksIon', None)
    
    # If not found in self, try to get from self.insar (for direct StripmapProc usage)
    if numberRangeLooksIon is None:
        numberRangeLooksIon = getattr(self.insar, 'numberRangeLooksIon', None)
    if numberAzimuthLooksIon is None:
        numberAzimuthLooksIon = getattr(self.insar, 'numberAzimuthLooksIon', None)
    
    logger.info('Ionospheric looks - Range: {}, Azimuth: {}'.format(numberRangeLooksIon, numberAzimuthLooksIon))
    print('DEBUG: Ionospheric looks - Range: {}, Azimuth: {}'.format(numberRangeLooksIon, numberAzimuthLooksIon))
    print('DEBUG: hasattr(self, numberRangeLooksIon): {}'.format(hasattr(self, 'numberRangeLooksIon')))
    print('DEBUG: hasattr(self.insar, numberRangeLooksIon): {}'.format(hasattr(self.insar, 'numberRangeLooksIon')))
    if hasattr(self, 'numberRangeLooksIon'):
        print('DEBUG: self.numberRangeLooksIon = {}'.format(self.numberRangeLooksIon))
    if hasattr(self.insar, 'numberRangeLooksIon'):
        print('DEBUG: self.insar.numberRangeLooksIon = {}'.format(self.insar.numberRangeLooksIon))
    
    if numberRangeLooksIon is not None and numberAzimuthLooksIon is not None:
        # Create multilooked version for ionospheric estimation
        totalAzLooks = azLooks * numberAzimuthLooksIon
        totalRgLooks = rgLooks * numberRangeLooksIon
        ml2 = '_{}rlks_{}alks'.format(totalRgLooks, totalAzLooks)
        logger.info('Creating additional multilooked interferogram with total looks: {}x{} ({}x{} base * {}x{} ion)'.format(
            totalRgLooks, totalAzLooks, rgLooks, azLooks, numberRangeLooksIon, numberAzimuthLooksIon))
        
        # Determine input file for additional multilooking
        # If doRubbersheetingRange is True, we should use .full file as input
        # Otherwise, use the regular interferogram
        inputFile = interferogramName
        useFullFile = False
        if self.doRubbersheetingRange:
            # Check if .full file exists and use it as input
            fullFile = interferogramName + '.full'
            if os.path.exists(fullFile):
                inputFile = fullFile
                useFullFile = True
                logger.info('Using .full file for additional multilooking: {}'.format(inputFile))
            else:
                logger.warning('.full file not found, using regular interferogram: {}'.format(inputFile))
        
        # Multilook the interferogram
        # Determine the original file extension to preserve it in output filename
        if '.flat' in interferogramName:
            originalExt = '.flat'
        elif '.int' in interferogramName:
            originalExt = '.int'
        else:
            originalExt = '.int'  # Default to .int if no extension found
        
        # Create base name with multilook suffix, preserving original extension
        multilookedBaseName = interferogramName.replace('.flat', '').replace('.int', '').replace('.full', '') + ml2
        multilookedName = multilookedBaseName + originalExt
        
        if not os.path.exists(multilookedName):
            # If using .full file, we need to apply total looks (base + ion)
            # Otherwise, we only apply additional ion looks
            # Pass the full filename with extension to multilook to ensure correct output name
            if useFullFile:
                # Apply total looks to .full file
                multilook(inputFile, outname=multilookedName, 
                         alks=totalAzLooks, rlks=totalRgLooks)
            else:
                # Apply additional looks to already multilooked file
                multilook(inputFile, outname=multilookedName, 
                         alks=numberAzimuthLooksIon, rlks=numberRangeLooksIon)
            
            # Check what extension was created by multilook and rename if needed
            # multilook may create files with different extensions or without extension
            # Force check and rename to ensure correct extension
            import time
            time.sleep(0.1)  # Brief wait to ensure file system is updated
            
            createdFile = None
            createdXml = None
            
            # First check if the desired file already exists (multilook may have created it correctly)
            if os.path.exists(multilookedName) and os.path.exists(multilookedName + '.xml'):
                logger.info('Multilooked interferogram created successfully: {}'.format(multilookedName))
            else:
                # Check for files with extensions first
                if os.path.exists(multilookedBaseName + '.flat'):
                    createdFile = multilookedBaseName + '.flat'
                    createdXml = multilookedBaseName + '.flat.xml'
                elif os.path.exists(multilookedBaseName + '.int'):
                    createdFile = multilookedBaseName + '.int'
                    createdXml = multilookedBaseName + '.int.xml'
                # Check for file without extension (multilook may not add extension)
                elif os.path.exists(multilookedBaseName):
                    createdFile = multilookedBaseName
                    createdXml = multilookedBaseName + '.xml'
                
                if createdFile:
                    # Always rename to ensure correct extension, even if file exists
                    if createdFile != multilookedName:
                        if os.path.exists(createdFile):
                            os.rename(createdFile, multilookedName)
                            logger.info('Renamed multilooked interferogram from {} to {}'.format(createdFile, multilookedName))
                        if os.path.exists(createdXml):
                            os.rename(createdXml, multilookedName + '.xml')
                        # Also rename VRT file if it exists
                        if os.path.exists(createdFile + '.vrt'):
                            os.rename(createdFile + '.vrt', multilookedName + '.vrt')
                        # After renaming, ensure VRT file is created/updated
                        if os.path.exists(multilookedName + '.xml'):
                            img = isceobj.createImage()
                            img.load(multilookedName + '.xml')
                            img.renderVRT()
                    else:
                        logger.info('Multilooked interferogram already has correct name: {}'.format(multilookedName))
                        # Ensure VRT file exists even if file wasn't renamed
                        if os.path.exists(multilookedName + '.xml') and not os.path.exists(multilookedName + '.vrt'):
                            img = isceobj.createImage()
                            img.load(multilookedName + '.xml')
                            img.renderVRT()
                else:
                    logger.warning('Multilooked interferogram file not found after multilooking. Expected: {} or {}'.format(
                        multilookedBaseName, multilookedName))
            
            # Also compute coherence for multilooked interferogram if needed
            # This will be done later in the workflow if needed
    
    return interferogramName

def runSubBandInterferograms(self):
    
    logger.info("Generating sub-band interferograms")

    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)
    secondaryFrame = self._insar.loadProduct( self._insar.secondarySlcCropProduct)

    azLooks, rgLooks = self.insar.numberOfLooks( referenceFrame, self.posting,
                                        self.numberAzimuthLooks, self.numberRangeLooks)

    self.numberAzimuthLooks = azLooks
    self.numberRangeLooks = rgLooks

    print("azimuth and range looks: ", azLooks, rgLooks)

    ###########
    referenceSlc =  referenceFrame.getImage().filename
    lowBandDir = os.path.join(self.insar.splitSpectrumDirname, self.insar.lowBandSlcDirname)
    highBandDir = os.path.join(self.insar.splitSpectrumDirname, self.insar.highBandSlcDirname)
    referenceLowBandSlc = os.path.join(lowBandDir, os.path.basename(referenceSlc))
    referenceHighBandSlc = os.path.join(highBandDir, os.path.basename(referenceSlc))
    ##########
    secondarySlc = secondaryFrame.getImage().filename
    coregDir = os.path.join(self.insar.coregDirname, self.insar.lowBandSlcDirname) 
    secondaryLowBandSlc = os.path.join(coregDir , os.path.basename(secondarySlc))
    coregDir = os.path.join(self.insar.coregDirname, self.insar.highBandSlcDirname)
    secondaryHighBandSlc = os.path.join(coregDir , os.path.basename(secondarySlc))
    ##########

    interferogramName = subBandIgram(self, referenceLowBandSlc, secondaryLowBandSlc, self.insar.lowBandSlcDirname,self.insar.lowBandRadarWavelength)

    interferogramName = subBandIgram(self, referenceHighBandSlc, secondaryHighBandSlc, self.insar.highBandSlcDirname,self.insar.highBandRadarWavelength)
    
    # Note: subBandIgram already handles multilooking if ionospheric looks are specified
    
def runFullBandInterferogram(self):
    logger.info("Generating interferogram")

    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)
    referenceSlc =  referenceFrame.getImage().filename
   
    if (self.doRubbersheetingRange | self.doRubbersheetingAzimuth):    
        secondarySlc = os.path.join(self._insar.coregDirname, self._insar.fineCoregFilename)
    else:
        secondarySlc = os.path.join(self._insar.coregDirname, self._insar.refinedCoregFilename)

    img1 = isceobj.createImage()
    img1.load(referenceSlc + '.xml')

    img2 = isceobj.createImage()
    img2.load(secondarySlc + '.xml')

    azLooks, rgLooks = self.insar.numberOfLooks( referenceFrame, self.posting, 
                            self.numberAzimuthLooks, self.numberRangeLooks) 

    self.numberAzimuthLooks = azLooks
    self.numberRangeLooks = rgLooks

    print("azimuth and range looks: ", azLooks, rgLooks)
    ifgDir = self.insar.ifgDirname

    if os.path.isdir(ifgDir):
        logger.info('Interferogram directory {0} already exists.'.format(ifgDir))
    else:
        os.makedirs(ifgDir)

    interferogramName = os.path.join(ifgDir , self.insar.ifgFilename)
    
    info = self._insar.loadProduct(self._insar.secondarySlcCropProduct)
    radarWavelength = info.getInstrument().getRadarWavelength()
    
    generateIgram(self,img1, img2, interferogramName, azLooks, rgLooks,radarWavelength)


    ###Compute coherence
    cohname = os.path.join(self.insar.ifgDirname, self.insar.correlationFilename)
    computeCoherence(referenceSlc, secondarySlc, cohname+'.full')
    multilook(cohname+'.full', outname=cohname, alks=azLooks, rlks=rgLooks)


    ##Multilook relevant geometry products
    for fname in [self.insar.latFilename, self.insar.lonFilename, self.insar.losFilename]:
        inname =  os.path.join(self.insar.geometryDirname, fname)
        multilook(inname + '.full', outname= inname, alks=azLooks, rlks=rgLooks)

def runInterferogram(self, igramSpectrum = "full"):

    logger.info("igramSpectrum = {0}".format(igramSpectrum))

    if igramSpectrum == "full":
        runFullBandInterferogram(self)


    elif igramSpectrum == "sub":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferograms')
            return
        runSubBandInterferograms(self) 

