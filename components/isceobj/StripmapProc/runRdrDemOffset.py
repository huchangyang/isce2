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
# Author: Adapted from Alos2Proc/runRdrDemOffset.py
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import logging
import numpy as np
import datetime

import isceobj
from mroipac.ampcor.Ampcor import Ampcor
from mroipac.looks.Looks import Looks

logger = logging.getLogger('isce.insar.runRdrDemOffset')


def runRdrDemOffset(self):
    '''Estimate offsets between radar image and DEM
    '''
    logger.info("Running rdr dem offset")
    
    # Get reference product
    referenceInfo = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    
    geometryDir = os.path.abspath(self.insar.geometryDirname)
    
    heightFilename = os.path.join(geometryDir, self.insar.heightFilename + '.full')
    heightFilename = os.path.abspath(heightFilename)
    
    # Get reference SLC path and convert to absolute path
    referenceSlc = referenceInfo.getImage().filename
    if not os.path.isabs(referenceSlc):
        referenceSlc = os.path.abspath(referenceSlc)

    # Create working directory
    rdrDemDir = os.path.join(geometryDir, 'rdr_dem_offset')
    os.makedirs(rdrDemDir, exist_ok=True)
    
    # Create catalog for tracking
    catalog = isceobj.Catalog.createCatalog(self._insar.procDoc.name)
    
    # Save current directory and change to working directory
    cwd = os.getcwd()
    try:
        os.chdir(rdrDemDir)
        rdrDemOffset(self, referenceInfo, heightFilename, referenceSlc, catalog=catalog)
    finally:
        os.chdir(cwd)

    # Record the results
    catalog.printToLog(logger, "runRdrDemOffset")
    self._insar.procDoc.addAllFromCatalog(catalog)


def rdrDemOffset(self, referenceInfo, heightFile, referenceSlc, catalog=None, skipTopoUpdate=False):
    '''Core function to estimate radar-DEM offsets
    
    Args:
        skipTopoUpdate: If True, skip calling updateTopoWithOffset (for stripmapStack, 
                       where topo will be re-run separately)
    '''
    # Get geometry directory (absolute path)
    geometryDir = os.path.abspath(self.insar.geometryDirname)
    # DEM pixel size (assumed to be 30m for simplicity)
    # For simplicity, we assume all DEMs have 30m resolution
    # This is a common resolution for DEMs like SRTM, ASTER GDEM, etc.
    demDeltaLon = 20.0  # DEM pixel size in range direction (meters)
    demDeltaLat = 20.0  # DEM pixel size in azimuth direction (meters)
    logger.info('DEM pixel size (assumed): {:.2f} m (range), {:.2f} m (azimuth)'.format(demDeltaLon, demDeltaLat))
    
    # Get SAR pixel sizes and first-level looks
    rangePixelSize = referenceInfo.getInstrument().getRangePixelSize()  # Range pixel size in meters
    prf = referenceInfo.PRF  # Pulse repetition frequency
    # Azimuth pixel size = velocity / PRF
    # For stripmap, we can approximate using platform velocity
    # Try to get azimuth pixel size from referenceInfo
    try:
        azimuthPixelSize = referenceInfo.getInstrument().getAzimuthPixelSize()
        if azimuthPixelSize is None:
            raise AttributeError('getAzimuthPixelSize returned None')
        logger.info('Azimuth pixel size from instrument: {:.2f} m'.format(azimuthPixelSize))
    except (AttributeError, NotImplementedError, TypeError):
        # If not available, estimate from PRF and platform velocity
        # For most SAR systems, azimuth pixel size ≈ velocity / PRF
        # This is an approximation - may need adjustment for specific sensors
        orbit = referenceInfo.getOrbit()
        tmid = referenceInfo.getSensingMid()
        sv = orbit.interpolateOrbit(tmid, method='hermite')
        velocity = np.linalg.norm(sv.getVelocity())
        azimuthPixelSize = velocity / prf
        logger.info('Estimated azimuth pixel size from velocity/PRF: {:.2f} m (velocity: {:.2f} m/s)'.format(
            azimuthPixelSize, velocity))
    
    # Automatically determine number of looks for simulation
    # If SAR resolution is already coarser than DEM, use single look
    # Otherwise, calculate looks to match DEM resolution
    if rangePixelSize > demDeltaLon:
        rangeLooks = 1  # SAR resolution already coarser than DEM
        logger.info('SAR range resolution ({:.2f} m) > DEM ({:.2f} m), using single look'.format(
            rangePixelSize, demDeltaLon))
    else:
        rangeLooks = max(1, int(demDeltaLon / rangePixelSize + 0.5))
        logger.info('Calculated range looks: {} (SAR: {:.2f} m, DEM: {:.2f} m)'.format(
            rangeLooks, rangePixelSize, demDeltaLon))
    
    if azimuthPixelSize > demDeltaLat:
        azimuthLooks = 1  # SAR resolution already coarser than DEM
        logger.info('SAR azimuth resolution ({:.2f} m) > DEM ({:.2f} m), using single look'.format(
            azimuthPixelSize, demDeltaLat))
    else:
        azimuthLooks = max(1, int(demDeltaLat / azimuthPixelSize + 0.5))
        logger.info('Calculated azimuth looks: {} (SAR: {:.2f} m, DEM: {:.2f} m)'.format(
            azimuthLooks, azimuthPixelSize, demDeltaLat))
    rangeLooks = 3
    azimuthLooks = 3
    logger.info('Selected multilook parameters: {} range looks, {} azimuth looks'.format(
        rangeLooks, azimuthLooks))
    
    # Check if simamp.rdr exists (generated by topo.py) - prefer this over sim.rdr
    # simamp.rdr is in the geometry directory (parent of current rdr_dem_offset directory)
    simampFile = os.path.join(geometryDir, 'simamp.rdr')
    simampExists = (os.path.exists(simampFile) and 
                    os.path.exists(simampFile + '.xml') and 
                    os.path.exists(simampFile + '.vrt'))
    
    # Check if sim.rdr already exists (in current working directory)
    # Changed from sim.float to sim.rdr
    simLookFile = 'sim.rdr'
    simExists = (os.path.exists(simLookFile) and 
                 os.path.exists(simLookFile + '.xml') and 
                 os.path.exists(simLookFile + '.vrt'))
    
    # Check if amp.rdr already exists
    # Changed from amp.float to amp.rdr
    referenceAmplitudeFilename = 'amp.rdr'
    ampExists = (os.path.exists(referenceAmplitudeFilename) and 
                 os.path.exists(referenceAmplitudeFilename + '.xml') and 
                 os.path.exists(referenceAmplitudeFilename + '.vrt'))
    
    # Generate simulated radar image if not exists
    # Prefer simamp.rdr from topo.py if it exists, otherwise generate sim.rdr
    if simampExists:
        logger.info("Found simamp.rdr from topo.py, using it for correlation")
        # Create a symlink or copy to current directory for convenience
        # But we'll use the absolute path directly
        simLookFile = simampFile
        simExists = True
    elif not simExists:
        logger.info("Simulated radar image not found, generating from DEM...")
        # Directly generate sim.rdr instead of sim.float
        simulateRadar(heightFile, simLookFile, scale=3.0, offset=100.0)
        simExists = True
    else:
        logger.info("Simulated radar image (sim.rdr) already exists, skipping generation")
    
    # Load sim image to get dimensions
    # Handle both absolute path (simamp.rdr) and relative path (sim.rdr)
    sim = isceobj.createImage()
    simXmlPath = simLookFile + '.xml'
    if not os.path.isabs(simXmlPath):
        simXmlPath = os.path.abspath(simXmlPath)
    sim.load(simXmlPath)
    
    # Generate amplitude from reference SLC if not exists
    if not ampExists:
        logger.info("Amplitude image not found, computing from SLC...")
        
        # Load SLC image to get dimensions
        slcImage = isceobj.createSlcImage()
        slcImageXmlPath = referenceSlc + '.xml'
        if not os.path.isabs(slcImageXmlPath):
            slcImageXmlPath = os.path.abspath(slcImageXmlPath)
        
        slcImage.load(slcImageXmlPath)
        
        # After loading, ensure both filename and extraFilename are absolute paths
        currentFilename = slcImage.getFilename()
        absoluteFilename = None
        if not os.path.isabs(currentFilename):
            xmlDir = os.path.dirname(slcImageXmlPath)
            filenameOnly = os.path.basename(currentFilename)
            absoluteFilename = os.path.join(xmlDir, filenameOnly)
            slcImage.setFilename(absoluteFilename)
        else:
            absoluteFilename = currentFilename
        
        # Get dimensions
        width = slcImage.getWidth()
        length = slcImage.getLength()
        
        # Read complex SLC data directly from file and compute amplitude
        slcDataFile = absoluteFilename
        if not os.path.exists(slcDataFile):
            if hasattr(slcImage, 'extraFilename') and slcImage.extraFilename:
                if os.path.isabs(slcImage.extraFilename):
                    slcDataFile = slcImage.extraFilename.replace('.vrt', '')
                else:
                    slcDataFile = os.path.join(os.path.dirname(absoluteFilename), 
                                              os.path.basename(slcImage.extraFilename.replace('.vrt', '')))
        
        # Read complex SLC data and compute amplitude
        chunk_size = 1000
        ampData = np.zeros((length, width), dtype=np.float32)
        
        logger.info("Computing amplitude from SLC")
        with open(slcDataFile, 'rb') as slcfp:
            for i in range(0, length, chunk_size):
                lines_to_read = min(chunk_size, length - i)
                logger.info("Processing chunk %6d of %6d" % (i//chunk_size + 1, (length + chunk_size - 1)//chunk_size))
                
                # Read complex data (complex64 = 2 * float32 per pixel)
                complex_data = np.fromfile(slcfp, dtype=np.complex64, 
                                         count=lines_to_read * width)
                complex_data = complex_data.reshape(lines_to_read, width)
                
                # Compute amplitude: |complex| = sqrt(real^2 + imag^2)
                ampData[i:i+lines_to_read, :] = np.abs(complex_data)
        
        # Write amplitude data as .rdr file (changed from .float)
        ampImage = isceobj.createImage()
        ampImage.setDataType('FLOAT')
        ampImage.setFilename(referenceAmplitudeFilename)
        ampImage.setWidth(width)
        ampImage.setLength(length)
        ampImage.setAccessMode('write')
        ampImage.createImage()
        
        # Write amplitude data to file
        with open(referenceAmplitudeFilename, 'wb') as ampfp:
            ampData.astype(np.float32).tofile(ampfp)
        
        slcImage.finalizeImage()
        ampImage.finalizeImage()
        ampImage.renderHdr()
        logger.info("Amplitude image (amp.rdr) generated successfully")
    else:
        logger.info("Amplitude image (amp.rdr) already exists, skipping computation")

    # Get image dimensions for offset calculation
    width = sim.width
    length = sim.length

    # Initial number of offsets to use
    # For StripmapProc, we use a fixed number since wbdOut is not available
    numberOfOffsets = 800  # Same as Alos2Proc default
    numberOfOffsetsRange = int(np.sqrt(numberOfOffsets))
    numberOfOffsetsAzimuth = int(np.sqrt(numberOfOffsets))
    
    # Adjust if image is too small
    if numberOfOffsetsRange > int(width/2):
        numberOfOffsetsRange = int(width/2)
    if numberOfOffsetsAzimuth > int(length/2):
        numberOfOffsetsAzimuth = int(length/2)

    if numberOfOffsetsRange < 10:
        numberOfOffsetsRange = 10
    if numberOfOffsetsAzimuth < 10:
        numberOfOffsetsAzimuth = 10

    if catalog is not None:
        catalog.addItem('number of range offsets', '{}'.format(numberOfOffsetsRange), 'runRdrDemOffset')
        catalog.addItem('number of azimuth offsets', '{}'.format(numberOfOffsetsAzimuth), 'runRdrDemOffset')

    # Multilook parameters - already calculated above based on DEM and SAR resolution
    # Use the automatically determined values

    # Apply multilook to images if needed
    if (rangeLooks == 1) and (azimuthLooks == 1):
        # No multilook needed, use original images
        ampLookFile = referenceAmplitudeFilename
        simLookFile = simLookFile
    else:
        # Apply multilook to both images
        ampLookFile = 'amp_{}rlks_{}alks.rdr'.format(rangeLooks, azimuthLooks)
        simLookFileMultilooked = 'sim_{}rlks_{}alks.rdr'.format(rangeLooks, azimuthLooks)
        
        # Check if multilooked images already exist
        ampLookExists = (os.path.exists(ampLookFile) and 
                        os.path.exists(ampLookFile + '.xml') and 
                        os.path.exists(ampLookFile + '.vrt'))
        simLookExists = (os.path.exists(simLookFileMultilooked) and 
                        os.path.exists(simLookFileMultilooked + '.xml') and 
                        os.path.exists(simLookFileMultilooked + '.vrt'))
        
        if not ampLookExists:
            logger.info('Multilooking amplitude image: {} x {} looks'.format(rangeLooks, azimuthLooks))
            
            # Use looks.py command line tool to avoid file size check issues
            cmd = "looks.py -i {} -o {} -r {} -a {}".format(
                referenceAmplitudeFilename, ampLookFile, rangeLooks, azimuthLooks)
            runCmd(cmd)
            
            logger.info('Multilooked amplitude image created: {}'.format(ampLookFile))
        else:
            logger.info('Multilooked amplitude image already exists: {}'.format(ampLookFile))
        
        if not simLookExists:
            logger.info('Multilooking simulated radar image: {} x {} looks'.format(rangeLooks, azimuthLooks))
            
            # Use looks.py command line tool
            # If simLookFile is absolute path (simamp.rdr), convert to relative for looks.py
            # looks.py will handle the path correctly
            simLookFileForCmd = simLookFile
            if os.path.isabs(simLookFile):
                # For absolute paths, looks.py should handle them correctly
                # But we need to ensure the output is in the current directory
                simLookFileForCmd = simLookFile
            
            cmd = "looks.py -i {} -o {} -r {} -a {}".format(
                simLookFileForCmd, simLookFileMultilooked, rangeLooks, azimuthLooks)
            runCmd(cmd)
            
            logger.info('Multilooked simulated image created: {}'.format(simLookFileMultilooked))
        else:
            logger.info('Multilooked simulated image already exists: {}'.format(simLookFileMultilooked))
        
        # Update simLookFile to use the multilooked version (always relative path in current dir)
        if not ((rangeLooks == 1) and (azimuthLooks == 1)):
            simLookFile = simLookFileMultilooked
    
    # Get multilooked image dimensions
    simLook = isceobj.createImage()
    # simLookFile is now always relative (multilooked version in current dir)
    simLook.load(simLookFile + '.xml')
    widthLooked = simLook.width
    lengthLooked = simLook.length
    simLook.finalizeImage()

    # Matching
    ampcor = Ampcor(name='insarapp_slcs_ampcor')
    ampcor.configure()

    mMag = isceobj.createImage()
    mMag.load(ampLookFile + '.xml')
    mMag.setAccessMode('read')
    mMag.createImage()

    sMag = isceobj.createImage()
    sMag.load(simLookFile + '.xml')
    sMag.setAccessMode('read')
    sMag.createImage()

    ampcor.setImageDataType1('real')
    ampcor.setImageDataType2('real')

    ampcor.setReferenceSlcImage(mMag)
    ampcor.setSecondarySlcImage(sMag)

    # MATCH REGION
    # Alos2Proc sets gross offset to 1 if it's 0
    rgoff = 0
    azoff = 0
    if rgoff == 0:
        rgoff = 1
    if azoff == 0:
        azoff = 1
    firstSample = 1
    if rgoff < 0:
        firstSample = int(35 - rgoff)
    firstLine = 1
    if azoff < 0:
        firstLine = int(35 - azoff)
    
    ampcor.setAcrossGrossOffset(rgoff)
    ampcor.setDownGrossOffset(azoff)
    ampcor.setFirstSampleAcross(firstSample)
    ampcor.setLastSampleAcross(widthLooked)
    ampcor.setNumberLocationAcross(numberOfOffsetsRange)
    ampcor.setFirstSampleDown(firstLine)
    ampcor.setLastSampleDown(lengthLooked)
    ampcor.setNumberLocationDown(numberOfOffsetsAzimuth)

    # MATCH PARAMETERS - Use same as Alos2Proc
    # For stricter matching, can increase window size (better correlation quality)
    ampcor.setWindowSizeWidth(128)  # Can increase to 128 for better quality (slower)
    ampcor.setWindowSizeHeight(128)  # Can increase to 128 for better quality (slower)
    ampcor.setSearchWindowSizeWidth(60)  # Same as Alos2Proc
    ampcor.setSearchWindowSizeHeight(60)  # Same as Alos2Proc
    
    # Set stricter SNR and covariance thresholds for better quality matches
    # Default thresholdSNR is 0.001, lower values are more lenient
    # For stricter matching, use higher thresholdSNR (e.g., 0.01 or 0.1)
    ampcor.thresholdSNR = 0.1  # Stricter: only accept matches with SNR >= 0.01 (default: 0.001)
    
    # Default thresholdCov is 1000.0, higher values are more lenient
    # For stricter matching, use lower thresholdCov (e.g., 500.0 or 100.0)
    ampcor.thresholdCov = 100.0  # Stricter: only accept matches with covariance <= 500.0 (default: 1000.0)

    # REST OF THE STUFF
    # Images are already multilooked, so ampcor uses looks=1
    ampcor.setAcrossLooks(1)
    ampcor.setDownLooks(1)
    # Increase oversampling factor for better sub-pixel accuracy (slower but more accurate)
    ampcor.setOversamplingFactor(128)  # Increased from 64 for better precision
    ampcor.setZoomWindowSize(16)
    ampcor.setDebugFlag(False)
    ampcor.setDisplayFlag(False)

    # Run ampcor
    ampcor.ampcor()
    offsets = ampcor.getOffsetField()
    ampcorOffsetFile = 'ampcor.off'
    writeOffset(offsets, ampcorOffsetFile)
    
    # Log how many offsets were found initially
    initialCount = len(offsets._offsets) if hasattr(offsets, '_offsets') else 'unknown'
    logger.info('Initial number of offsets from ampcor: {}'.format(initialCount))
    
    # If still too few points, try even more lenient settings
    if hasattr(offsets, '_offsets') and len(offsets._offsets) < 100:
        logger.warning('Very few points from ampcor ({}), matching quality may be poor'.format(len(offsets._offsets)))

    # Finalize images
    mMag.finalizeImage()
    sMag.finalizeImage()

    # Cull offsets and fit constant shifts (similar to runRefineSecondaryTiming)
    try:
        from iscesys.StdOEL.StdOELPy import create_writer
        
        field = offsets
        stdWriter = create_writer("log", "", True, filename='off.log')
        
        # Cull offsets using iterative distance sequence (similar to Alos2Proc's fitoff approach)
        # Reference: Alos2Proc uses fitoff with nsig=1.5, maxrms=0.5, minpoint=50
        # We use iterative distance-based culling to achieve similar effect
        snrThreshold = 2  # Similar to Alos2Proc's nsig=1.5 (stricter threshold)

        # Optional thresholds for later covariance-based culling (in pixel^2 units).
        # Points whose estimated uncertainty (covariance) is larger than the threshold will be removed.
        sigmaThresholdRange = 0.01      # threshold for range covariance
        sigmaThresholdAzimuth = 0.1     # threshold for azimuth covariance

        # Optional statistical culling based purely on the distribution of rg / az offsets.
        offsetNSigmaRange = 1.0
        offsetNSigmaAzimuth = 2.0
        # Use distance sequence to progressively remove outliers
        # Similar to fitoff's iterative approach, we tighten the distance threshold progressively
        for distance in [10, 5, 3, 1]:  # Progressive tightening (similar to fitoff iterations)
            pointsBefore = len(field._offsets)
            objOff = isceobj.createOffoutliers()
            objOff.wireInputPort(name='offsets', object=field)
            objOff.setSNRThreshold(snrThreshold)
            objOff.setDistance(distance)
            objOff.setStdWriter(stdWriter)
            objOff.offoutliers()
            field = objOff.getRefinedOffsetField()
            pointsAfter = len(field._offsets)
            logger.info('{} points left after culling at distance {} with SNR threshold {} (removed {} points)'.format(
                pointsAfter, distance, snrThreshold, pointsBefore - pointsAfter))

            # No early stopping - let the culling process complete all distance steps
            # This ensures we remove outliers progressively, similar to fitoff's iterative approach
            # Reference: Alos2Proc's fitoff continues until convergence or minpoint is reached

        if (offsetNSigmaRange is not None) or (offsetNSigmaAzimuth is not None):
            rg_list = []
            az_list = []
            for offsetx in field:
                offsetStr = "{}".format(offsetx)
                fields = offsetStr.split()
                if len(fields) >= 4:
                    try:
                        rg_list.append(float(fields[1]))  # dx: range offset
                        az_list.append(float(fields[3]))  # dy: azimuth offset
                    except ValueError:
                        continue

            if len(rg_list) >= 2 and len(az_list) >= 2:
                rg_array = np.array(rg_list, dtype=np.float64)
                az_array = np.array(az_list, dtype=np.float64)

                rg_mean = float(np.mean(rg_array))
                az_mean = float(np.mean(az_array))
                rg_std = float(np.std(rg_array))
                az_std = float(np.std(az_array))

                filtered_offsets_stat = []
                removedStat = 0

                for offsetx in field:
                    offsetStr = "{}".format(offsetx)
                    fields = offsetStr.split()
                    if len(fields) < 4:
                        removedStat += 1
                        continue
                    try:
                        rg_off = float(fields[1])
                        az_off = float(fields[3])
                    except ValueError:
                        removedStat += 1
                        continue

                    drop = False
                    if offsetNSigmaRange is not None and rg_std > 0.0:
                        if abs(rg_off - rg_mean) > offsetNSigmaRange * rg_std:
                            drop = True
                    if offsetNSigmaAzimuth is not None and az_std > 0.0:
                        if abs(az_off - az_mean) > offsetNSigmaAzimuth * az_std:
                            drop = True

                    if drop:
                        removedStat += 1
                    else:
                        filtered_offsets_stat.append(offsetx)

                logger.info(
                    '{} points left after statistical offset-based culling (removed {} points; '
                    'range mean = {:.4f}, std = {:.4f}, Nσ = {}; '
                    'azimuth mean = {:.4f}, std = {:.4f}, Nσ = {}).'.format(
                        len(filtered_offsets_stat), removedStat,
                        rg_mean, rg_std, offsetNSigmaRange,
                        az_mean, az_std, offsetNSigmaAzimuth)
                )

                field._offsets = filtered_offsets_stat
            else:
                logger.info('Not enough points for statistical offset-based culling ({} points). '
                            'Skipping this step.'.format(len(field._offsets)))

        # Optionally apply an additional culling based on per-point standard deviation / covariance.
        # We use the covariance terms stored in the original offset field (from ampcor),
        # and keep only the points whose uncertainties are below the specified thresholds.
        if (sigmaThresholdRange is not None) or (sigmaThresholdAzimuth is not None):
            # Build a lookup table from original offsets using (range line, range sample) as key
            originalOffsetMap = {}
            for offsetx in offsets:
                offsetStr = "{}".format(offsetx)
                fields = offsetStr.split()
                if len(fields) >= 8:
                    ref_line = int(float(fields[0]))
                    ref_sample = float(fields[1])
                    originalOffsetMap[(ref_line, ref_sample)] = fields

            filtered_offsets = []
            removedSigma = 0
            for offsetx in field:
                offsetStr = "{}".format(offsetx)
                fields = offsetStr.split()
                if len(fields) < 4:
                    # Malformed entry, skip it
                    removedSigma += 1
                    continue

                ref_line = int(float(fields[0]))
                ref_sample = float(fields[1])
                key = (ref_line, ref_sample)

                orig_fields = originalOffsetMap.get(key, None)
                if (orig_fields is None) or (len(orig_fields) < 8):
                    # Cannot recover covariance info for this point, drop it
                    removedSigma += 1
                    continue

                # Covariance / standard deviation terms from the original ampcor offsets
                cov_range = float(orig_fields[5])
                cov_azimuth = float(orig_fields[6])

                # Apply thresholds (if set). Use absolute value in case of negative covariances.
                if ((sigmaThresholdRange is not None and abs(cov_range) > sigmaThresholdRange) or
                    (sigmaThresholdAzimuth is not None and abs(cov_azimuth) > sigmaThresholdAzimuth)):
                    removedSigma += 1
                    continue

                filtered_offsets.append(offsetx)

            logger.info('{} points left after sigma-based culling (removed {} points with large covariance; '
                        'range threshold = {}, azimuth threshold = {})'.format(
                            len(filtered_offsets), removedSigma,
                            sigmaThresholdRange, sigmaThresholdAzimuth))

            # Replace the internal list of offsets with the sigma-filtered subset
            field._offsets = filtered_offsets

        # Save culled offsets to a new .off file
        # Note: getRefinedOffsetField() may lose some fields (SNR, Corr, AzOffset)
        # So we need to match culled offsets with original offsets to preserve all fields
        culledOffsetFile = 'ampcor_culled.off'
        writeOffsetWithOriginalInfo(field, offsets, culledOffsetFile)
        logger.info('Saved culled offsets to: {} ({} points)'.format(culledOffsetFile, len(field._offsets)))
        
        # Check final number of points (similar to Alos2Proc's minpoint=50 check)
        # Alos2Proc requires at least 50 points, we use a similar threshold
        minPointsForFitting = 50  # Similar to Alos2Proc's minpoint=50
        if len(field._offsets) < minPointsForFitting:
            logger.warning('Too few points left after culling, {} left (minimum {} required for fitting)'.format(len(field._offsets), minPointsForFitting))
            logger.warning('Do not estimate offsets between radar and dem')
            self._insar.radarDemRangeOffset = 0.0
            self._insar.radarDemAzimuthOffset = 0.0
            self._insar.radarDemAffineTransform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            if catalog is not None:
                catalog.addItem('warning message', 
                              'too few points left after culling, {} left'.format(len(field._offsets)), 
                              'runRdrDemOffset')
            return
        elif len(field._offsets) < 100:  # Warn if below 100 points (similar to Alos2Proc's approach)
            logger.warning('Low number of points ({}), but attempting to fit offsets'.format(len(field._offsets)))

        # Fit constant offsets (zero-order polynomials)
        aa, dummy = field.getFitPolynomials(azimuthOrder=0, rangeOrder=0, usenumpy=True)
        dummy, rr = field.getFitPolynomials(azimuthOrder=0, rangeOrder=0, usenumpy=True)
        
        # Get offsets in multilooked pixel units
        az_offset_multilooked = aa._coeffs[0][0]
        rg_offset_multilooked = rr._coeffs[0][0]
        
        # Convert from multilooked pixel units to single-look pixel units
        # If we used N looks, the offset needs to be multiplied by N to get single-look units
        rg_offset = rg_offset_multilooked * rangeLooks
        az_offset = az_offset_multilooked * azimuthLooks
        
        logger.info('Estimated range offset (multilooked): {:.6f} pixels ({} looks)'.format(rg_offset_multilooked, rangeLooks))
        logger.info('Estimated azimuth offset (multilooked): {:.6f} pixels ({} looks)'.format(az_offset_multilooked, azimuthLooks))
        logger.info('Converted range offset (single-look): {:.6f} pixels'.format(rg_offset))
        logger.info('Converted azimuth offset (single-look): {:.6f} pixels'.format(az_offset))
        
        # Store single-look pixel offsets (these will be used in runGeo2rdr)
        self._insar.radarDemRangeOffset = rg_offset
        self._insar.radarDemAzimuthOffset = az_offset
        
        # Store as affine transform format: [1, 0, 0, 1, rg_offset, az_offset]
        self._insar.radarDemAffineTransform = [1.0, 0.0, 0.0, 1.0, rg_offset, az_offset]
        
        # Save offsets to text file in geometry directory
        offsetFile = os.path.join(geometryDir, 'rdr_dem_offsets.txt')
        try:
            with open(offsetFile, 'w') as f:
                f.write('# Radar-DEM offsets estimated by rdrDemOffset\n')
                f.write('# Format: range_offset and azimuth_offset are in pixels (single-look)\n')
                f.write('# affine_transform: [a, b, c, d, e, f] where [e, f] are range and azimuth offsets\n')
                f.write('range_offset: {:.6f}\n'.format(rg_offset))
                f.write('azimuth_offset: {:.6f}\n'.format(az_offset))
                f.write('affine_transform: {}\n'.format(self._insar.radarDemAffineTransform))
            logger.info('Saved offsets to file: {}'.format(offsetFile))
        except Exception as e:
            logger.warning('Failed to save offsets to file: {}'.format(e))
        
        if catalog is not None:
            catalog.addItem('radar dem range offset', '{:.6f}'.format(rg_offset), 'runRdrDemOffset')
            catalog.addItem('radar dem azimuth offset', '{:.6f}'.format(az_offset), 'runRdrDemOffset')
        
        # Update referenceInfo with corrected geometry so subsequent steps use the corrected values
        if abs(rg_offset) > 0.01 or abs(az_offset) > 0.01:
            logger.info('Updating referenceInfo with corrected geometry')
            
            # Determine orbit direction (ascending/descending) to decide correction sign
            # Get pass direction from frame (same approach as used in TopsProc/runIon.py)
            passDirection = None
            try:
                if hasattr(referenceInfo, 'frame') and hasattr(referenceInfo.frame, 'passDirection'):
                    passDirection = referenceInfo.frame.passDirection
            except Exception as e:
                logger.warning('Could not get pass direction from frame: {}'.format(e))
            
            # Apply range offset correction
            # Note: ampcor returns offset as "secondary relative to reference"
            # The /2 factor accounts for round-trip propagation: ground error = round-trip error / 2
            rangePixelSize = referenceInfo.getInstrument().getRangePixelSize()
            rangeOffsetMeters = rg_offset * rangePixelSize
            originalStartingRange = referenceInfo.startingRange
            correctedStartingRange = originalStartingRange + rangeOffsetMeters
            referenceInfo.startingRange = correctedStartingRange
            
            # Apply azimuth offset correction
            # Similar logic: correction sign depends on orbit direction
            prf = referenceInfo.PRF
            azimuthOffsetSeconds = az_offset / prf
            originalSensingStart = referenceInfo.getSensingStart()
            correctedSensingStart = originalSensingStart + datetime.timedelta(seconds=azimuthOffsetSeconds)
            referenceInfo.sensingStart = correctedSensingStart
            
            logger.info('Updated referenceInfo.startingRange: {:.6f} m -> {:.6f} m (offset: {:.6f} pixels = {:.2f} m)'.format(
                originalStartingRange, correctedStartingRange, rg_offset, rangeOffsetMeters))
            logger.info('Updated referenceInfo.sensingStart: {} -> {} (offset: {:.6f} pixels = {:.6f} s)'.format(
                originalSensingStart, correctedSensingStart, az_offset, azimuthOffsetSeconds))
            
            # Save the updated product so subsequent steps can use the corrected values
            self._insar.saveProduct(referenceInfo, self._insar.referenceSlcCropProduct)
            logger.info('Saved updated referenceInfo product with corrected geometry')
        
        # Re-run topo with corrected geometry to update lat.rdr.full, lon.rdr.full, z.rdr.full
        # Skip this if called from stripmapStack (skipTopoUpdate=True), where topo will be re-run separately
        if not skipTopoUpdate and (abs(rg_offset) > 0.01 or abs(az_offset) > 0.01):
            logger.info('Re-running topo with corrected geometry based on estimated offsets')
            logger.info('This will update lat.rdr.full, lon.rdr.full, and z.rdr.full files')
            updateTopoWithOffset(self, referenceInfo, rg_offset, az_offset, referenceSlc=referenceSlc)
        elif skipTopoUpdate:
            logger.info('Skipping topo update (will be handled by stripmapStack caller)')
        
    except Exception as e:
        logger.warning('Could not fit constant offsets, using zero offsets: {}'.format(e))
        self._insar.radarDemRangeOffset = 0.0
        self._insar.radarDemAzimuthOffset = 0.0
        self._insar.radarDemAffineTransform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        
        # Save zero offsets to text file as well
        geometryDir = os.path.abspath(self.insar.geometryDirname)
        offsetFile = os.path.join(geometryDir, 'rdr_dem_offsets.txt')
        try:
            with open(offsetFile, 'w') as f:
                f.write('# Radar-DEM offsets estimated by rdrDemOffset\n')
                f.write('# Format: range_offset and azimuth_offset are in pixels (single-look)\n')
                f.write('# affine_transform: [a, b, c, d, e, f] where [e, f] are range and azimuth offsets\n')
                f.write('# Warning: Could not fit offsets, using zero offsets\n')
                f.write('range_offset: 0.000000\n')
                f.write('azimuth_offset: 0.000000\n')
                f.write('affine_transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]\n')
            logger.info('Saved zero offsets to file: {}'.format(offsetFile))
        except Exception as e2:
            logger.warning('Failed to save offsets to file: {}'.format(e2))
        
        if catalog is not None:
            catalog.addItem('warning message', 
                          'Could not fit constant offsets, using zero offsets', 
                          'runRdrDemOffset')
        return


def simulateRadar(hgtfile, simfile, scale=3.0, offset=100.0):
    '''
    Simulate a radar image by computing gradient of a DEM image.
    '''
    hgt = isceobj.createImage()
    hgt.load(hgtfile + '.xml')

    chunk_length = 1000
    chunk_width = hgt.width
    num_chunk = int(hgt.length / chunk_length)
    chunk_length_last = hgt.length - num_chunk * chunk_length

    simData = np.zeros((chunk_length, chunk_width), dtype=np.float32)

    hgtfp = open(hgtfile, 'rb')
    simfp = open(simfile, 'wb')

    logger.info("Simulating a radar image using topography")
    for i in range(num_chunk):
        logger.info("Processing chunk %6d of %6d" % (i+1, num_chunk))
        hgtData = np.fromfile(hgtfp, dtype=np.float64, 
                              count=chunk_length*chunk_width).reshape(chunk_length, chunk_width)
        simData[:, 0:chunk_width-1] = scale * np.diff(hgtData, axis=1) + offset
        simData.astype(np.float32).tofile(simfp)

    if chunk_length_last != 0:
        hgtData = np.fromfile(hgtfp, dtype=np.float64, 
                              count=chunk_length_last*chunk_width).reshape(chunk_length_last, chunk_width)
        simData[0:chunk_length_last, 0:chunk_width-1] = scale * np.diff(hgtData, axis=1) + offset
        (simData[0:chunk_length_last, :]).astype(np.float32).tofile(simfp)

    hgtfp.close()
    simfp.close()
    # Changed from 'float' to 'rdr' - but still use FLOAT data type
    # The fileType parameter is used for XML creation, but we want .rdr extension
    create_xml(simfile, hgt.width, hgt.length, 'rdr')


def create_xml(fileName, width, length, fileType):
    '''
    Create XML file for an image (similar to Alos2ProcPublic.create_xml).
    '''
    if fileType == 'slc':
        image = isceobj.createSlcImage()
    elif fileType == 'int':
        image = isceobj.createIntImage()
    elif fileType == 'amp':
        image = isceobj.createAmpImage()
    elif fileType == 'float' or fileType == 'rdr':
        # Support both 'float' and 'rdr' - both use FLOAT data type
        # 'rdr' is just a naming convention, data type is still FLOAT
        image = isceobj.createImage()
        image.setDataType('FLOAT')
    elif fileType == 'double':
        image = isceobj.createImage()
        image.setDataType('DOUBLE')
    else:
        image = isceobj.createImage()

    image.setFilename(fileName)
    image.extraFilename = fileName + '.vrt'
    image.setWidth(width)
    image.setLength(length)
    image.renderHdr()


def writeOffset(offset, fileName):
    '''
    Write offset file in ampcor format (similar to Alos2ProcPublic.writeOffset).
    
    .off file format (ampcor standard):
    Each line contains 8 fields:
    1. Reference image line number (integer, azimuth position)
    2. Reference image sample number (float, range position, sub-pixel precision)
    3. Secondary image line number (integer, azimuth position)
    4. Secondary image sample number (float, range position, sub-pixel precision)
    5. Range offset (float, pixels) - offset in range direction
    6. Azimuth offset (float, pixels) - offset in azimuth direction
    7. SNR (float) - Signal-to-Noise Ratio of the correlation
    8. Correlation coefficient (float) - Quality metric of the match
    
    Format: {:8d} {:10.3f} {:8d} {:12.3f} {:11.5f} {:11.6f} {:11.6f} {:11.6f}
    '''
    # Write header line with column descriptions
    # Note: Offset.__str__() returns: x dx y dy snr sigmax sigmay sigmaxy
    # Column names reflect actual data content
    header = "# {:>7} {:>10} {:>7} {:>12} {:>11} {:>11} {:>11} {:>11}\n".format(
        "RgLoc", "RgOffset", "AzLoc", "AzOffset", "SNR", "RgCov", "AzCov", "CrossCov"
    )
    header += "# {:>7} {:>10} {:>7} {:>12} {:>11} {:>11} {:>11} {:>11}\n".format(
        "(int)", "(pixels)", "(int)", "(pixels)", "", "(cov)", "(cov)", "(cov)"
    )
    
    offsetsPlain = header
    for offsetx in offset:
        offsetsPlainx = "{}".format(offsetx)
        offsetsPlainx = offsetsPlainx.split()
        offsetsPlain = offsetsPlain + "{:8d} {:10.3f} {:8d} {:12.3f} {:11.5f} {:11.6f} {:11.6f} {:11.6f}\n".format(
            int(float(offsetsPlainx[0])),      # Ref line (azimuth)
            float(offsetsPlainx[1]),           # Ref sample (range, sub-pixel)
            int(float(offsetsPlainx[2])),      # Sec line (azimuth)
            float(offsetsPlainx[3]),           # Sec sample (range, sub-pixel)
            float(offsetsPlainx[4]),           # Range offset (pixels)
            float(offsetsPlainx[5]),           # Azimuth offset (pixels)
            float(offsetsPlainx[6]),           # SNR
            float(offsetsPlainx[7])            # Correlation coefficient
        )

    with open(fileName, 'w') as f:
        f.write(offsetsPlain)


def writeOffsetWithOriginalInfo(culledField, originalField, fileName):
    '''
    Write culled offset file, preserving all fields from original offsets.
    
    This function matches culled offsets with original offsets to preserve
    SNR, correlation, and azimuth offset information that may be lost in
    the refined offset field.
    
    Args:
        culledField: Refined offset field after culling
        originalField: Original offset field before culling
        fileName: Output filename
    '''
    # Create a mapping from (ref_line, ref_sample) to original offset entry
    originalOffsetMap = {}
    for offsetx in originalField:
        offsetStr = "{}".format(offsetx)
        fields = offsetStr.split()
        if len(fields) >= 8:
            ref_line = int(float(fields[0]))
            ref_sample = float(fields[1])
            key = (ref_line, ref_sample)
            originalOffsetMap[key] = fields
    
    # Write header
    # Note: Offset.__str__() returns: x dx y dy snr sigmax sigmay sigmaxy
    # Column names reflect actual data content
    header = "# {:>7} {:>10} {:>7} {:>12} {:>11} {:>11} {:>11} {:>11}\n".format(
        "RgLoc", "RgOffset", "AzLoc", "AzOffset", "SNR", "RgCov", "AzCov", "CrossCov"
    )
    header += "# {:>7} {:>10} {:>7} {:>12} {:>11} {:>11} {:>11} {:>11}\n".format(
        "(int)", "(pixels)", "(int)", "(pixels)", "", "(cov)", "(cov)", "(cov)"
    )
    
    offsetsPlain = header
    
    # Write culled offsets, matching with original to preserve all fields
    for offsetx in culledField:
        offsetStr = "{}".format(offsetx)
        fields = offsetStr.split()
        if len(fields) >= 5:  # At least ref_line, ref_sample, sec_line, sec_sample, rg_offset
            ref_line = int(float(fields[0]))
            ref_sample = float(fields[1])
            key = (ref_line, ref_sample)
            
            # Try to find matching original offset to preserve all fields
            if key in originalOffsetMap:
                orig_fields = originalOffsetMap[key]
                # Use original fields which have all 8 values
                offsetsPlain = offsetsPlain + "{:8d} {:10.3f} {:8d} {:12.3f} {:11.5f} {:11.6f} {:11.6f} {:11.6f}\n".format(
                    int(float(orig_fields[0])),      # Ref line (azimuth)
                    float(orig_fields[1]),           # Ref sample (range, sub-pixel)
                    int(float(orig_fields[2])),      # Sec line (azimuth)
                    float(orig_fields[3]),           # Sec sample (range, sub-pixel)
                    float(orig_fields[4]),           # Range offset (pixels) - use from original
                    float(orig_fields[5]),           # Azimuth offset (pixels) - preserve from original
                    float(orig_fields[6]),           # SNR - preserve from original
                    float(orig_fields[7])            # Correlation coefficient - preserve from original
                )
            else:
                # Fallback: use culled offset fields, fill missing with zeros
                sec_line = int(float(fields[2])) if len(fields) > 2 else int(float(fields[0]))
                sec_sample = float(fields[3]) if len(fields) > 3 else float(fields[1])
                rg_offset = float(fields[4]) if len(fields) > 4 else 0.0
                az_offset = float(fields[5]) if len(fields) > 5 else 0.0
                snr = float(fields[6]) if len(fields) > 6 else 0.0
                corr = float(fields[7]) if len(fields) > 7 else 0.0
                offsetsPlain = offsetsPlain + "{:8d} {:10.3f} {:8d} {:12.3f} {:11.5f} {:11.6f} {:11.6f} {:11.6f}\n".format(
                    ref_line, ref_sample, sec_line, sec_sample, rg_offset, az_offset, snr, corr
        )

    with open(fileName, 'w') as f:
        f.write(offsetsPlain)


def runCmd(cmd, silent=0):
    '''
    Run a command and check status.
    '''
    if silent == 0:
        logger.info("Running: {}".format(cmd))
    status = os.system(cmd)
    if status != 0:
        raise Exception('Error when running:\n{}\n'.format(cmd))


def updateTopoWithOffset(self, referenceInfo, rangeOffset, azimuthOffset, referenceSlc=None):
    '''Re-run topo with corrected geometry based on estimated offsets
    Note: referenceInfo should already have corrected startingRange and sensingStart
    This function updates lat.rdr.full, lon.rdr.full, z.rdr.full in geometryDir
    '''
    from zerodop.topozero import createTopozero
    from isceobj.Planet.Planet import Planet
    from isceobj.Util.Poly2D import Poly2D
    from isceobj.Constants import SPEED_OF_LIGHT
    
    logger.info("Updating topo geometry files with radar-DEM offsets")
    
    # Get geometry directory (absolute path)
    geometryDir = os.path.abspath(self.insar.geometryDirname)
    
    # IMPORTANT: Update self.insar.geometryDirname to point to the correct location
    # This ensures subsequent steps (like geo2rdr) can find the geometry files
    self.insar.geometryDirname = geometryDir
    logger.info("Updated self.insar.geometryDirname to: {}".format(geometryDir))
    
    os.makedirs(geometryDir, exist_ok=True)
    logger.info("Target geometry directory: {}".format(geometryDir))
    
    demFilename = self.verifyDEM()
    objDem = isceobj.createDemImage()
    objDem.load(demFilename + '.xml')
    
    intImage = referenceInfo.getImage()
    planet = referenceInfo.getInstrument().getPlatform().getPlanet()
    
    topo = createTopozero()
    
    topo.slantRangePixelSpacing = 0.5 * SPEED_OF_LIGHT / referenceInfo.rangeSamplingRate
    topo.prf = referenceInfo.PRF
    topo.radarWavelength = referenceInfo.radarWavelegth
    topo.orbit = referenceInfo.orbit
    
    # IMPORTANT: For .full files, we need to use full resolution dimensions
    # Load the SLC image XML directly to get the actual full resolution dimensions
    # This ensures we get the correct dimensions even if referenceInfo.getImage() returns multilooked dimensions
    if referenceSlc is not None:
        # Use the provided referenceSlc path (should be absolute path)
        slcXmlPath = referenceSlc + '.xml'
        if os.path.exists(slcXmlPath):
            slcImageForDims = isceobj.createSlcImage()
            slcImageForDims.load(slcXmlPath)
            fullWidth = slcImageForDims.getWidth()
            fullLength = slcImageForDims.getLength()
            slcImageForDims.finalizeImage()
            logger.info('Loaded full resolution dimensions from SLC XML: width={}, length={}'.format(fullWidth, fullLength))
        else:
            # Fallback: try to get from referenceInfo.getImage()
            logger.warning('SLC XML not found at {}, using referenceInfo.getImage() dimensions'.format(slcXmlPath))
            fullWidth = intImage.getWidth()
            fullLength = intImage.getLength()
            logger.info('Using intImage dimensions: width={}, length={}'.format(fullWidth, fullLength))
    else:
        # Fallback: try to get from referenceInfo.getImage() or its filename
        slcImage = referenceInfo.getImage()
        slcXmlPath = slcImage.filename + '.xml' if hasattr(slcImage, 'filename') and slcImage.filename else None
        if slcXmlPath and os.path.exists(slcXmlPath):
            slcImageReloaded = isceobj.createSlcImage()
            slcImageReloaded.load(slcXmlPath)
            fullWidth = slcImageReloaded.getWidth()
            fullLength = slcImageReloaded.getLength()
            slcImageReloaded.finalizeImage()
            logger.info('Loaded full resolution dimensions from slcImage.filename: width={}, length={}'.format(fullWidth, fullLength))
        else:
            # Final fallback: use intImage dimensions
            fullWidth = intImage.getWidth()
            fullLength = intImage.getLength()
            logger.warning('Could not load SLC XML, using intImage dimensions: width={}, length={}'.format(fullWidth, fullLength))
    
    # Set topo dimensions to full resolution (for .full files)
    topo.width = fullWidth
    topo.length = fullLength
    logger.info('Topo will generate .full files with dimensions: width={}, length={}'.format(topo.width, topo.length))
    
    topo.wireInputPort(name='dem', object=objDem)
    topo.wireInputPort(name='planet', object=planet)
    topo.numberRangeLooks = 1
    topo.numberAzimuthLooks = 1
    topo.lookSide = referenceInfo.getInstrument().getPlatform().pointingDirection
    
    # Use corrected values from referenceInfo (already updated in rdrDemOffset)
    topo.rangeFirstSample = referenceInfo.startingRange
    topo.sensingStart = referenceInfo.getSensingStart()
    
    logger.info('Using corrected geometry from referenceInfo:')
    logger.info('  rangeFirstSample: {:.6f} m'.format(topo.rangeFirstSample))
    logger.info('  sensingStart: {}'.format(topo.sensingStart))
    
    topo.demInterpolationMethod = 'BIQUINTIC'
    
    # Set filenames using os.path.join with geometryDir (same as runTopo)
    latFile = os.path.join(geometryDir, self.insar.latFilename + '.full')
    lonFile = os.path.join(geometryDir, self.insar.lonFilename + '.full')
    losFile = os.path.join(geometryDir, self.insar.losFilename + '.full')
    hgtFile = os.path.join(geometryDir, self.insar.heightFilename + '.full')
    
    topo.latFilename = latFile
    topo.lonFilename = lonFile
    topo.losFilename = losFile
    topo.heightFilename = hgtFile
    
    logger.info('Topo output files:')
    logger.info('  latFilename: {}'.format(topo.latFilename))
    logger.info('  lonFilename: {}'.format(topo.lonFilename))
    logger.info('  losFilename: {}'.format(topo.losFilename))
    logger.info('  heightFilename: {}'.format(topo.heightFilename))
    
    # Delete old geometry files to force regeneration
    for f in [latFile, lonFile, losFile, hgtFile]:
        if os.path.exists(f):
            logger.info('Deleting old geometry file: {}'.format(f))
            os.remove(f)
        # Also remove .xml and .vrt if they exist
        for ext in ['.xml', '.vrt']:
            if os.path.exists(f + ext):
                logger.info('Deleting old geometry file: {}'.format(f + ext))
                os.remove(f + ext)
    
    # Doppler adjustment
    dop = [x/1.0 for x in referenceInfo._dopplerVsPixel]
    doppler = Poly2D()
    doppler.setWidth(topo.width // topo.numberRangeLooks)
    doppler.setLength(topo.length // topo.numberAzimuthLooks)
    
    if self._insar.referenceGeometrySystem.lower().startswith('native'):
        doppler.initPoly(rangeOrder=len(dop)-1, azimuthOrder=0, coeffs=[dop])
    else:
        doppler.initPoly(rangeOrder=0, azimuthOrder=0, coeffs=[[0.]])
    
    topo.polyDoppler = doppler
    
    # Re-run topo to update geometry files
    logger.info('Calling topo.topo() with corrected geometry...')
    topo.topo()
    
    # Check where files were actually created and move to target directory if needed
    currentDir = os.getcwd()
    logger.info('Current working directory: {}'.format(currentDir))
    
    # List of geometry files to check and move
    geometryFiles = [
        (self.insar.latFilename + '.full', latFile),
        (self.insar.lonFilename + '.full', lonFile),
        (self.insar.losFilename + '.full', losFile),
        (self.insar.heightFilename + '.full', hgtFile)
    ]
    
    # Check if files were created in current directory or subdirectory and move them
    for filename, targetFile in geometryFiles:
        # Check in current directory
        currentPath = os.path.join(currentDir, filename)
        # Check in geometry subdirectory (if topo created one)
        geometrySubPath = os.path.join(currentDir, 'geometry', filename)
        # Target directory path (ensure absolute path)
        targetPath = os.path.abspath(targetFile)
        
        sourcePath = None
        # First check where the NEW file was created (after topo.topo())
        # Priority: geometry subdirectory > current directory
        if os.path.exists(geometrySubPath):
            # File was created in geometry subdirectory
            logger.info('Found geometry file in subdirectory: {}'.format(geometrySubPath))
            sourcePath = os.path.abspath(geometrySubPath)
        elif os.path.exists(currentPath):
            # File was created in current directory
            logger.info('Found geometry file in current directory: {}'.format(currentPath))
            sourcePath = os.path.abspath(currentPath)
        elif os.path.exists(targetPath):
            # File is already in correct location
            logger.info('Geometry file already in correct location: {}'.format(targetPath))
            continue
        
        if sourcePath:
            # Check if source and target are the same (normalize paths)
            if os.path.abspath(sourcePath) == os.path.abspath(targetPath):
                logger.info('Source and target are the same, skipping move: {}'.format(sourcePath))
                continue
            
            # If target file exists and is different from source, remove it first (it's the old file)
            if os.path.exists(targetPath) and os.path.abspath(targetPath) != os.path.abspath(sourcePath):
                logger.info('Removing old geometry file: {}'.format(targetPath))
                os.remove(targetPath)
                # Also remove associated .xml and .vrt files
                for ext in ['.xml', '.vrt']:
                    oldExt = targetPath + ext
                    if os.path.exists(oldExt):
                        os.remove(oldExt)
            
            # Move file and associated .xml and .vrt files to target directory using runCmd
            logger.info('Moving {} to {}'.format(sourcePath, targetPath))
            cmd = 'mv {} {}'.format(sourcePath, targetPath)
            runCmd(cmd)
            
            # Move .xml and .vrt files if they exist
            for ext in ['.xml', '.vrt']:
                sourceExt = sourcePath + ext
                targetExt = targetPath + ext
                if os.path.exists(sourceExt):
                    logger.info('Moving {} to {}'.format(sourceExt, targetExt))
                    cmd = 'mv {} {}'.format(sourceExt, targetExt)
                    runCmd(cmd)
    
    # Verify files were created in the correct location
    for f in [latFile, lonFile, losFile, hgtFile]:
        if os.path.exists(f):
            logger.info('Geometry file created successfully: {}'.format(f))
        else:
            logger.warning('Geometry file not found after topo: {}'.format(f))
    
    # Update estimated bounding box
    self._insar.estimatedBbox = [topo.minimumLatitude, topo.maximumLatitude,
                                topo.minimumLongitude, topo.maximumLongitude]
    
    logger.info('Topo geometry files updated successfully in: {}'.format(geometryDir))

