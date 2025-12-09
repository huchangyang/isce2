#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Brett George
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# Heresh Fattahi: adopted for stripmapApp and generalized for full-band and sub-band interferograms 


import logging
import isceobj

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from mroipac.filter.Filter import Filter
from mroipac.icu.Icu import Icu
import os

logger = logging.getLogger('isce.insar.runFilter')

def runFilter(self, filterStrength, igramSpectrum = "full"):
    logger.info("Applying power-spectral filter")

    if igramSpectrum == "full":
        logger.info("Filtering the full-band interferogram")
        ifgDirname = self.insar.ifgDirname

    elif igramSpectrum == "low":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferograms')
            return
        logger.info("Filtering the low-band interferogram")
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)

    elif igramSpectrum == "high":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferograms')
            return
        logger.info("Filtering the high-band interferogram")
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)

    # Check if ionospheric looks are specified and look for multilooked interferogram
    # For low and high band, we may need to filter multilooked interferograms for ionospheric estimation
    useMultilookedIgram = False
    if igramSpectrum in ["low", "high"]:
        # In runFilter, self is the Insar instance, so get parameters from self first
        numberRangeLooksIon = getattr(self, 'numberRangeLooksIon', None)
        numberAzimuthLooksIon = getattr(self, 'numberAzimuthLooksIon', None)
        
        # If not found in self, try to get from self.insar (for direct StripmapProc usage)
        if numberRangeLooksIon is None:
            numberRangeLooksIon = getattr(self.insar, 'numberRangeLooksIon', None)
        if numberAzimuthLooksIon is None:
            numberAzimuthLooksIon = getattr(self.insar, 'numberAzimuthLooksIon', None)
        
        if numberRangeLooksIon is not None and numberAzimuthLooksIon is not None:
            # Look for multilooked interferogram for ionospheric estimation
            referenceFrame = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
            azLooks, rgLooks = self.insar.numberOfLooks(referenceFrame, self.posting,
                                                        self.numberAzimuthLooks, self.numberRangeLooks)
            totalAzLooks = azLooks * numberAzimuthLooksIon
            totalRgLooks = rgLooks * numberRangeLooksIon
            ml2 = '_{}rlks_{}alks'.format(totalRgLooks, totalAzLooks)
            
            # Try multilooked interferogram first
            multilookIntFilename = os.path.join(ifgDirname, self.insar.ifgFilename.replace('.flat', ml2 + '.flat'))
            if not os.path.exists(multilookIntFilename + '.xml'):
                multilookIntFilename = os.path.join(ifgDirname, self.insar.ifgFilename.replace('.flat', ml2 + '.int'))
            
            if os.path.exists(multilookIntFilename + '.xml'):
                topoflatIntFilename = multilookIntFilename
                useMultilookedIgram = True
                logger.info('Found multilooked interferogram for filtering: {}'.format(topoflatIntFilename))
            else:
                # Fall back to regular interferogram
                topoflatIntFilename = os.path.join(ifgDirname , self.insar.ifgFilename)
                logger.info('Multilooked interferogram not found, using regular interferogram')
        else:
            # No ionospheric looks specified, use regular interferogram
            topoflatIntFilename = os.path.join(ifgDirname , self.insar.ifgFilename)
    else:
        # Full band, use regular interferogram
        topoflatIntFilename = os.path.join(ifgDirname , self.insar.ifgFilename)

    img1 = isceobj.createImage()
    img1.load(topoflatIntFilename + '.xml')
    widthInt = img1.getWidth()
    
    # Ensure VRT file exists (may be missing after file renaming)
    if not os.path.exists(topoflatIntFilename + '.vrt') and os.path.exists(topoflatIntFilename + '.xml'):
        img1.renderVRT()

    intImage = isceobj.createIntImage()
    intImage.setFilename(topoflatIntFilename)
    intImage.setWidth(widthInt)
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    # If using multilooked interferogram, preserve the multilook suffix in output filename
    if useMultilookedIgram:
        # Extract the multilook suffix from input filename
        if '.flat' in topoflatIntFilename:
            baseName = topoflatIntFilename.replace('.flat', '')
            filtIntFilename = os.path.join(ifgDirname, 'filt_' + os.path.basename(baseName) + '.flat')
        elif '.int' in topoflatIntFilename:
            baseName = topoflatIntFilename.replace('.int', '')
            filtIntFilename = os.path.join(ifgDirname, 'filt_' + os.path.basename(baseName) + '.int')
        else:
            filtIntFilename = os.path.join(ifgDirname, 'filt_' + os.path.basename(topoflatIntFilename))
    else:
        filtIntFilename = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename)
    
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(filtIntFilename)
    filtImage.setWidth(widthInt)
    filtImage.setAccessMode('write')
    filtImage.createImage()
    
    objFilter = Filter()
    objFilter.wireInputPort(name='interferogram',object=intImage)
    objFilter.wireOutputPort(name='filtered interferogram',object=filtImage)
    if filterStrength is not None:
        self.insar.filterStrength = filterStrength
    
    objFilter.goldsteinWerner(alpha=self.insar.filterStrength)

    intImage.finalizeImage()
    filtImage.finalizeImage()
    del filtImage
    
    #Create phase sigma correlation file here
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(filtIntFilename)
    filtImage.setWidth(widthInt)
    filtImage.setAccessMode('read')
    filtImage.createImage()


    phsigImage = isceobj.createImage()
    phsigImage.dataType='FLOAT'
    phsigImage.bands = 1
    phsigImage.setWidth(widthInt)
    phsigImage.setFilename(os.path.join(ifgDirname , self.insar.coherenceFilename))
    phsigImage.setAccessMode('write')
    phsigImage.setImageType('cor')#the type in this case is not for mdx.py displaying but for geocoding method
    phsigImage.createImage()

    # Get amplitude file - use multilooked version if available
    if useMultilookedIgram:
        # Use multilooked amplitude file
        if '.flat' in topoflatIntFilename:
            resampAmpImage = topoflatIntFilename.replace('.flat', '.amp')
        elif '.int' in topoflatIntFilename:
            resampAmpImage = topoflatIntFilename.replace('.int', '.amp')
        else:
            resampAmpImage = topoflatIntFilename + '.amp'
        
        # If multilooked amplitude doesn't exist, fall back to regular amplitude
        if not os.path.exists(resampAmpImage + '.xml'):
            logger.warning('Multilooked amplitude file not found: {}. Using regular amplitude file.'.format(resampAmpImage))
            resampAmpImage = os.path.join(ifgDirname , self.insar.ifgFilename)
            if '.flat' in resampAmpImage:
                resampAmpImage = resampAmpImage.replace('.flat', '.amp')
            elif '.int' in resampAmpImage:
                resampAmpImage = resampAmpImage.replace('.int', '.amp')
            else:
                resampAmpImage += '.amp'
    else:
        resampAmpImage = os.path.join(ifgDirname , self.insar.ifgFilename)
        if '.flat' in resampAmpImage:
            resampAmpImage = resampAmpImage.replace('.flat', '.amp')
        elif '.int' in resampAmpImage:
            resampAmpImage = resampAmpImage.replace('.int', '.amp')
        else:
            resampAmpImage += '.amp'

    ampImage = isceobj.createAmpImage()
    ampImage.setWidth(widthInt)
    ampImage.setFilename(resampAmpImage)
    #IU.copyAttributes(self.insar.resampAmpImage, ampImage)
    #IU.copyAttributes(resampAmpImage, ampImage)
    ampImage.setAccessMode('read')
    ampImage.createImage()


    icuObj = Icu(name='stripmapapp_filter_icu')
    icuObj.configure()
    icuObj.unwrappingFlag = False

    icuObj.icu(intImage = filtImage, ampImage=ampImage, phsigImage=phsigImage)

    filtImage.finalizeImage()
    phsigImage.finalizeImage()
    phsigImage.renderHdr()
    ampImage.finalizeImage()



    # Set the filtered image to be the one geocoded
    # self.insar.topophaseFlatFilename = filtIntFilename
