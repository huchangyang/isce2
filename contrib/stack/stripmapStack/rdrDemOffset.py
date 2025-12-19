#!/usr/bin/env python3
# author

import os
import argparse
import shelve
import datetime
import shutil
import numpy as np
import types
import isce
import isceobj
import logging

logger = logging.getLogger('isce.stripmapStack.rdrDemOffset')


def createParser():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser(description='Estimate radar-DEM offsets for stripmap stack')
    parser.add_argument('-m', '--reference', dest='reference', type=str, required=True,
            help='Dir with reference frame')
    parser.add_argument('-d', '--dem', dest='dem', type=str, required=True,
            help='Input DEM to use')
    parser.add_argument('-o', '--output', dest='outdir', type=str, required=True,
            help='Output directory (geometry directory)')
    parser.add_argument('-a','--alks', dest='alks', type=int, default=1,
            help='Number of azimuth looks for multilooking geometry files')
    parser.add_argument('-r','--rlks', dest='rlks', type=int, default=1,
            help='Number of range looks for multilooking geometry files')
    return parser


def cmdLineParse(iargs=None):
    parser = createParser()
    return parser.parse_args(args=iargs)


class InsarAdapter(object):
    """Adapter class to make StripmapProc's rdrDemOffset work with stripmapStack"""
    def __init__(self, frame, geometryDir, demFilename):
        self.frame = frame
        self.geometryDir = geometryDir
        self.demFilename = demFilename
        
        # Create a minimal insar-like object
        self.insar = type('obj', (object,), {})()
        self.insar.geometryDirname = geometryDir
        self.insar.latFilename = 'lat.rdr'
        self.insar.lonFilename = 'lon.rdr'
        self.insar.losFilename = 'los.rdr'
        self.insar.heightFilename = 'hgt.rdr'
        self.insar.referenceGeometrySystem = 'zero'  # or 'native' if needed
        
        self._insar = type('obj', (object,), {})()
        self._insar.numberRangeLooks = 1
        self._insar.numberAzimuthLooks = 1
        self._insar.estimatedBbox = None
        
        # Add referenceGeometrySystem to _insar for compatibility with updateTopoWithOffset
        self._insar.referenceGeometrySystem = 'zero'  # or 'native' if needed
        
        # Store offsets
        self._insar.radarDemRangeOffset = 0.0
        self._insar.radarDemAzimuthOffset = 0.0
        self._insar.radarDemAffineTransform = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        
        # Add saveProduct method to _insar for compatibility
        def saveProduct(self, product, productPath):
            # In stripmapStack, we work directly with frames, so we don't need to save
            # The frame is already updated in place
            pass
        self._insar.saveProduct = types.MethodType(saveProduct, self._insar)
        
        # Store referenceSlcCropProduct for compatibility
        self._insar.referenceSlcCropProduct = 'referenceSlcCropProduct'
    
    def verifyDEM(self):
        """Return DEM filename"""
        return self.demFilename
    
    def loadProduct(self, productPath):
        """Load a product - for reference SLC, return frame wrapped as product"""
        # In stripmapStack, we have frame directly, so we return it
        # We need to wrap it in a way that StripmapProc expects
        return self.frame
    
    def saveProduct(self, product, productPath):
        """Save product - for stripmapStack, we may not need to save"""
        # In stripmapStack, we work directly with frames
        pass


def main(iargs=None):
    inps = cmdLineParse(iargs)
    
    # Load reference frame
    framePath = os.path.join(inps.reference, 'data')
    db = shelve.open(framePath)
    frame = db['frame']
    db.close()
    
    # Create geometry directory if it doesn't exist
    os.makedirs(inps.outdir, exist_ok=True)
    
    # Create adapter
    adapter = InsarAdapter(frame, inps.outdir, inps.dem)
    
    # Import the core function from StripmapProc
    from isceobj.StripmapProc.runRdrDemOffset import rdrDemOffset
    
    # Get reference SLC path
    # In stripmapStack, SLC is typically in the reference directory
    referenceSlc = os.path.join(inps.reference, os.path.basename(inps.reference) + '.slc')
    if not os.path.exists(referenceSlc):
        # Try alternative naming
        referenceSlc = os.path.join(inps.reference, 'reference.slc')
        if not os.path.exists(referenceSlc):
            # Look for any .slc file
            slc_files = [f for f in os.listdir(inps.reference) if f.endswith('.slc')]
            if slc_files:
                referenceSlc = os.path.join(inps.reference, slc_files[0])
            else:
                raise FileNotFoundError('Could not find reference SLC file in {}'.format(inps.reference))
    
    # Get height filename (should be hgt.rdr.full in geometry directory)
    heightFilename = os.path.join(inps.outdir, 'hgt.rdr.full')
    if not os.path.exists(heightFilename):
        # Try without .full extension
        heightFilename = os.path.join(inps.outdir, 'hgt.rdr')
        if not os.path.exists(heightFilename):
            raise FileNotFoundError('Could not find height file: {}'.format(heightFilename))
    
    # Convert to absolute paths BEFORE changing directory
    heightFilename = os.path.abspath(heightFilename)
    referenceSlc = os.path.abspath(referenceSlc)
    
    # Create working directory for intermediate files (similar to runRdrDemOffset)
    rdrDemDir = os.path.join(inps.outdir, 'rdr_dem_offset')
    os.makedirs(rdrDemDir, exist_ok=True)
    
    # Create a referenceInfo-like object from frame
    # We need to wrap frame to match what StripmapProc expects
    class ReferenceInfoWrapper:
        def __init__(self, frame):
            self.frame = frame
        
        def getImage(self):
            # Return an image object for the SLC
            slcImage = isceobj.createSlcImage()
            slcImage.setFilename(referenceSlc)
            slcImage.setWidth(frame.getNumberOfSamples())
            slcImage.setLength(frame.getNumberOfLines())
            return slcImage
        
        def getInstrument(self):
            return frame.getInstrument()
        
        def getOrbit(self):
            return frame.getOrbit()
        
        @property
        def orbit(self):
            """Orbit property for compatibility with StripmapProc"""
            return frame.getOrbit()
        
        def getSensingStart(self):
            return frame.getSensingStart()
        
        def getSensingMid(self):
            return frame.getSensingMid()
        
        def getSensingStop(self):
            return frame.getSensingStop()
        
        @property
        def startingRange(self):
            return frame.startingRange
        
        @startingRange.setter
        def startingRange(self, value):
            frame.startingRange = value
        
        @property
        def sensingStart(self):
            return frame.getSensingStart()
        
        @sensingStart.setter
        def sensingStart(self, value):
            if hasattr(frame, 'setSensingStart'):
                frame.setSensingStart(value)
            else:
                frame.sensingStart = value
        
        @property
        def PRF(self):
            return frame.PRF
        
        @property
        def rangeSamplingRate(self):
            return frame.rangeSamplingRate
        
        @property
        def radarWavelegth(self):
            return frame.radarWavelegth
        
        @property
        def _dopplerVsPixel(self):
            return frame._dopplerVsPixel
    
    referenceInfo = ReferenceInfoWrapper(frame)
    
    # Save current directory and change to working directory
    cwd = os.getcwd()
    try:
        os.chdir(rdrDemDir)
        
        # Call the core function
        # Set skipTopoUpdate=True so that topo will be re-run in rdrDemOffset.py after frame update
        logger.info('Calling StripmapProc rdrDemOffset...')
        rdrDemOffset(adapter, referenceInfo, heightFilename, referenceSlc, catalog=None, skipTopoUpdate=True)
    finally:
        os.chdir(cwd)
    
    # Move rdr_dem_offsets.txt from geometry directory to rdr_dem_offset subdirectory
    # This ensures all intermediate files (except topo .full files) are in rdr_dem_offset
    geometryDir = os.path.abspath(inps.outdir)
    offsetFileInGeometry = os.path.join(geometryDir, 'rdr_dem_offsets.txt')
    offsetFileInRdrDemDir = os.path.join(rdrDemDir, 'rdr_dem_offsets.txt')
    if os.path.exists(offsetFileInGeometry):
        logger.info('Moving rdr_dem_offsets.txt from {} to {}'.format(offsetFileInGeometry, offsetFileInRdrDemDir))
        shutil.move(offsetFileInGeometry, offsetFileInRdrDemDir)
    
    # Update frame with corrected geometry if offsets were estimated
    if abs(adapter._insar.radarDemRangeOffset) > 0.01 or abs(adapter._insar.radarDemAzimuthOffset) > 0.01:
        logger.info('Updating frame with corrected geometry')
        # Get corrected values (already updated in rdrDemOffset via ReferenceInfoWrapper)
        correctedStartingRange = referenceInfo.startingRange
        correctedSensingStart = referenceInfo.sensingStart
        
        # Open database with writeback=True to enable in-place attribute modification
        framePath = os.path.join(inps.reference, 'data')
        db = shelve.open(framePath, writeback=True)
        try:
            # Get frame from database and modify only the two attributes
            dbFrame = db['frame']
            dbFrame.startingRange = correctedStartingRange
            if hasattr(dbFrame, 'setSensingStart'):
                dbFrame.setSensingStart(correctedSensingStart)
            else:
                dbFrame.sensingStart = correctedSensingStart
            
            # With writeback=True, changes are tracked automatically
            # Just need to sync to ensure changes are written to disk
            db.sync()
            logger.info('Updated frame attributes: startingRange={:.6f}, sensingStart={}'.format(
                correctedStartingRange, correctedSensingStart))
        finally:
            db.close()
        logger.info('Saved updated frame with corrected geometry')
        
        # Directly adjust topo output files with estimated offsets instead of re-running topo
        logger.info('Directly adjusting topo geometry files with estimated offsets')
        adjustTopoFilesWithOffset(adapter, inps.outdir, 
                                  adapter._insar.radarDemRangeOffset,
                                  adapter._insar.radarDemAzimuthOffset)
        
        # Skip re-running topo since we've already adjusted the files directly
        # The adjusted files are ready to use
        logger.info('Skipping topo re-run since files have been directly adjusted')
        skipTopoRerun = True
        
        if not skipTopoRerun:
            # Re-run topo with corrected geometry to regenerate all geometry files
            # This will regenerate .full files with corrected geometry and also perform multilook
            logger.info('Re-running topo with corrected geometry to regenerate all geometry files')
            
            # Import topo module
            from stripmapStack import topo
            
            # Build command line arguments for topo.main()
            # Ensure alks and rlks are integers (they might be strings from config file)
            alks = int(inps.alks) if hasattr(inps, 'alks') else 1
            rlks = int(inps.rlks) if hasattr(inps, 'rlks') else 1
            
            # Build argument list for topo.main()
            topo_args = [
                '-m', inps.reference,
                '-d', inps.dem,
                '-o', inps.outdir,
                '-a', str(alks),
                '-r', str(rlks)
            ]
            
            # Add optional flags if they exist in inps
            if hasattr(inps, 'nativedop') and inps.nativedop:
                topo_args.append('-n')
            if hasattr(inps, 'legendre') and inps.legendre:
                topo_args.append('-l')
            if hasattr(inps, 'useGPU') and inps.useGPU:
                topo_args.append('--useGPU')
            
            logger.info('Calling topo.main() with arguments: {}'.format(' '.join(topo_args)))
            topo.main(topo_args)
            logger.info('Topo completed successfully, geometry files regenerated')
        
        # Generate simamp_new.rdr using the updated height file to test the correction effect
        logger.info('Generating simamp_new.rdr using corrected geometry for testing...')
        try:
            from iscesys.StdOEL.StdOELPy import create_writer
            
            # Determine the height file name based on multilook settings
            # In stripmapStack, topo generates hgt.rdr (multilooked) and hgt.rdr.full (full resolution)
            # We'll use the multilooked version if it exists, otherwise use .full
            hgtFileMultilooked = os.path.join(inps.outdir, 'hgt.rdr')
            hgtFileFull = os.path.join(inps.outdir, 'hgt.rdr.full')
            
            # Prefer multilooked version for simamp (consistent with original simamp.rdr)
            if os.path.exists(hgtFileMultilooked):
                hgtFile = hgtFileMultilooked
                logger.info('Using multilooked height file: {}'.format(hgtFile))
            elif os.path.exists(hgtFileFull):
                hgtFile = hgtFileFull
                logger.info('Using full resolution height file: {}'.format(hgtFile))
            else:
                raise FileNotFoundError('Could not find height file (hgt.rdr or hgt.rdr.full)')
            
            simampNewFile = os.path.join(inps.outdir, 'simamp_new.rdr')
            
            stdWriter = create_writer("log", "", True, filename=os.path.join(inps.outdir, 'simamp_new.log'))
            objShade = isceobj.createSimamplitude()
            objShade.setStdWriter(stdWriter)
            
            # Load the updated height image
            hgtImage = isceobj.createImage()
            hgtImage.load(hgtFile + '.xml')
            hgtImage.setAccessMode('read')
            hgtImage.createImage()
            
            # Create simamp_new.rdr image
            simImage = isceobj.createImage()
            simImage.setFilename(simampNewFile)
            simImage.dataType = 'FLOAT'
            simImage.setAccessMode('write')
            simImage.setWidth(hgtImage.getWidth())
            simImage.setLength(hgtImage.getLength())
            simImage.createImage()
            
            # Generate simulated amplitude using the corrected height file
            objShade.simamplitude(hgtImage, simImage, shade=3.0)
            
            simImage.renderHdr()
            hgtImage.finalizeImage()
            simImage.finalizeImage()
            
            logger.info('Successfully generated simamp_new.rdr at: {}'.format(simampNewFile))
            logger.info('You can now compare simamp_new.rdr with the original simamp.rdr to verify the correction effect')
        except Exception as e:
            logger.warning('Failed to generate simamp_new.rdr: {}'.format(e))
            import traceback
            logger.warning(traceback.format_exc())
    
    logger.info('rdr_dem_offset completed successfully')
    return


def adjustTopoFilesWithOffset(adapter, geometryDir, rangeOffset, azimuthOffset):
    '''Directly adjust topo output files using estimated offsets
    
    This function adjusts all topo geometry files by applying pixel shifts based on the estimated
    offsets, rather than re-running topo. This is more direct and avoids potential issues
    with startingRange/sensingStart updates.
    
    In stripmapStack:
    - Full resolution files are in merged/geom_reference/ (if merged directory exists)
    - Multilooked files are in geom_reference/ (the geometryDir passed in)
    
    Args:
        adapter: InsarAdapter object with geometry directory info
        geometryDir: Geometry directory path (typically geom_reference, contains multilooked files)
        rangeOffset: Range offset in single-look pixels
        azimuthOffset: Azimuth offset in single-look pixels
    '''
    # Try to import scipy for sub-pixel interpolation, fallback to numpy if not available
    try:
        from scipy import ndimage
        has_scipy = True
    except ImportError:
        has_scipy = False
        logger.warning('scipy not available, using numpy for integer-pixel shifts only')
    
    logger.info("Directly adjusting topo geometry files with radar-DEM offsets")
    logger.info("Range offset: {:.6f} pixels (single-look)".format(rangeOffset))
    logger.info("Azimuth offset: {:.6f} pixels (single-look)".format(azimuthOffset))
    
    # In stripmapStack, check for both full resolution and multilooked files
    # Full resolution files are typically in merged/geom_reference/
    # Multilooked files are in geom_reference/ (the geometryDir)
    geometryDirAbs = os.path.abspath(geometryDir)
    
    # Find merged/geom_reference directory (full resolution files)
    # Strategy: go up from geometryDir to find merged directory
    # In stripmapStack structure: .../merged/geom_reference/ (full resolution)
    #                             .../geom_reference/ (multilooked)
    mergedGeomDir = None
    currentDir = geometryDirAbs
    for _ in range(5):  # Check up to 5 levels up
        parentDir = os.path.dirname(currentDir)
        if os.path.basename(parentDir) == 'merged':
            candidateMergedGeomDir = os.path.join(parentDir, 'geom_reference')
            if os.path.exists(candidateMergedGeomDir):
                mergedGeomDir = candidateMergedGeomDir
                logger.info('Found merged/geom_reference directory: {}'.format(mergedGeomDir))
                break
        currentDir = parentDir
        if currentDir == parentDir:  # Reached root
            break
    
    if mergedGeomDir is None:
        logger.info('Could not find merged/geom_reference directory, will only check {}'.format(geometryDirAbs))
    
    # All topo output files (in order of importance)
    topoFiles = [
        'lat.rdr',
        'lon.rdr', 
        'los.rdr',
        'hgt.rdr',
        'incLocal.rdr',
        'shadowMask.rdr'
    ]
    
    # Check both full resolution (merged/geom_reference) and multilooked (geom_reference) locations
    filesToAdjust = []
    
    for filename in topoFiles:
        fileFound = False
        
        # First check merged/geom_reference (full resolution)
        if mergedGeomDir is not None:
            fullResFile = os.path.join(mergedGeomDir, filename)
            if os.path.exists(fullResFile):
                filesToAdjust.append((os.path.splitext(filename)[0], fullResFile))
                fileFound = True
                logger.info('Found full resolution file: {}'.format(fullResFile))
        
        # Then check geometryDir (multilooked)
        multilookedFile = os.path.join(geometryDirAbs, filename)
        if os.path.exists(multilookedFile):
            filesToAdjust.append((os.path.splitext(filename)[0], multilookedFile))
            fileFound = True
            logger.info('Found multilooked file: {}'.format(multilookedFile))
        
        if not fileFound:
            logger.warning('File not found: {} (checked merged/geom_reference and {})'.format(
                filename, geometryDirAbs))
    
    if not filesToAdjust:
        logger.error('No geometry files found to adjust. Please run topo first.')
        return
    
    # We'll get dimensions for each file individually since different files may have different data types
    # (e.g., shadowMask is BYTE, others are DOUBLE or FLOAT)
    
    # Convert offsets from single-look to the resolution of the geometry files
    # The geometry files are typically at full resolution (single-look) for .full files
    # or multilooked for non-.full files
    # Offset is "sim relative to amp", so we need to shift in the opposite direction
    # to correct the geometry
    
    # Determine correction sign based on offset interpretation
    # If offset > 0 means sim is to the right/after amp, we need to shift geometry left/before
    # So we use negative offset for correction
    rangeShift = -rangeOffset  # Negative because we're correcting the geometry
    azimuthShift = -azimuthOffset  # Negative because we're correcting the geometry
    
    logger.info('Applying shifts: range={:.6f} pixels, azimuth={:.6f} pixels'.format(
        rangeShift, azimuthShift))
    
    # Process each geometry file
    for fileType, filePath in filesToAdjust:
        logger.info('Adjusting {} file: {}'.format(fileType, filePath))
        
        try:
            # Load image to get dimensions and data type
            img = isceobj.createImage()
            img.load(filePath + '.xml')
            img.setAccessMode('read')
            img.createImage()
            
            # Get dimensions for this specific file
            fileWidth = img.getWidth()
            fileLength = img.getLength()
            
            # Determine data type from image
            # shadowMask is BYTE, others are typically DOUBLE or FLOAT
            dataType = img.getDataType()
            if dataType == 'BYTE':
                dtype = np.uint8
            elif dataType == 'FLOAT':
                dtype = np.float32
            elif dataType == 'DOUBLE':
                dtype = np.float64
            else:
                # Default to float64 for safety
                dtype = np.float64
                logger.warning('Unknown data type {} for {}, using float64'.format(dataType, fileType))
            
            # Check if multi-band (like los, incLocal which have 2 bands)
            bands = 1
            if hasattr(img, 'bands') and img.bands:
                bands = img.bands
            elif hasattr(img, 'getBands'):
                bands = img.getBands()
            
            logger.info('File {}: width={}, length={}, bands={}, dtype={}'.format(
                fileType, fileWidth, fileLength, bands, dtype))
            
            # Read data with correct dtype
            totalPixels = fileWidth * fileLength * bands
            data = np.fromfile(filePath, dtype=dtype, count=totalPixels)
            
            # Reshape based on bands
            if bands > 1:
                # Multi-band: reshape to (length, width, bands)
                data = data.reshape(fileLength, fileWidth, bands)
            else:
                # Single band: reshape to (length, width)
                data = data.reshape(fileLength, fileWidth)
            
            # Apply shift
            # Note: shift order is (row, col) = (azimuth, range)
            # For multi-band data, shift all bands together
            if has_scipy:
                # Use scipy for sub-pixel interpolation
                if bands > 1:
                    # Multi-band: shift each band separately
                    shifted_bands = []
                    for b in range(bands):
                        shifted_band = ndimage.shift(data[:, :, b], (azimuthShift, rangeShift), 
                                                    order=1, mode='nearest', prefilter=False)
                        shifted_bands.append(shifted_band)
                    shifted_data = np.stack(shifted_bands, axis=2)
                else:
                    # Single band
                    shifted_data = ndimage.shift(data, (azimuthShift, rangeShift), 
                                                order=1, mode='nearest', prefilter=False)
            else:
                # Fallback to numpy for integer-pixel shifts only
                az_shift_int = int(np.round(azimuthShift))
                rg_shift_int = int(np.round(rangeShift))
                logger.warning('Using integer-pixel shift only: azimuth={}, range={}'.format(
                    az_shift_int, rg_shift_int))
                if bands > 1:
                    # Multi-band: shift each band separately
                    shifted_bands = []
                    for b in range(bands):
                        shifted_band = np.roll(data[:, :, b], (-az_shift_int, -rg_shift_int), axis=(0, 1))
                        # Fill edges
                        if az_shift_int > 0:
                            shifted_band[:az_shift_int, :] = shifted_band[az_shift_int, :]
                        elif az_shift_int < 0:
                            shifted_band[az_shift_int:, :] = shifted_band[az_shift_int-1, :]
                        if rg_shift_int > 0:
                            shifted_band[:, :rg_shift_int] = shifted_band[:, rg_shift_int:rg_shift_int+1]
                        elif rg_shift_int < 0:
                            shifted_band[:, rg_shift_int:] = shifted_band[:, rg_shift_int-1:rg_shift_int]
                        shifted_bands.append(shifted_band)
                    shifted_data = np.stack(shifted_bands, axis=2)
                else:
                    # Single band
                    shifted_data = np.roll(data, (-az_shift_int, -rg_shift_int), axis=(0, 1))
                    # Fill edges with nearest values
                    if az_shift_int > 0:
                        shifted_data[:az_shift_int, :] = shifted_data[az_shift_int, :]
                    elif az_shift_int < 0:
                        shifted_data[az_shift_int:, :] = shifted_data[az_shift_int-1, :]
                    if rg_shift_int > 0:
                        shifted_data[:, :rg_shift_int] = shifted_data[:, rg_shift_int:rg_shift_int+1]
                    elif rg_shift_int < 0:
                        shifted_data[:, rg_shift_int:] = shifted_data[:, rg_shift_int-1:rg_shift_int]
            
            # Write adjusted data
            # Create backup first
            backupPath = filePath + '.backup'
            if not os.path.exists(backupPath):
                logger.info('Creating backup: {}'.format(backupPath))
                shutil.copy2(filePath, backupPath)
                if os.path.exists(filePath + '.xml'):
                    shutil.copy2(filePath + '.xml', backupPath + '.xml')
                if os.path.exists(filePath + '.vrt'):
                    shutil.copy2(filePath + '.vrt', backupPath + '.vrt')
            
            # Write adjusted data with correct dtype
            with open(filePath, 'wb') as f:
                shifted_data.astype(dtype).tofile(f)
            
            img.finalizeImage()
            
            logger.info('Successfully adjusted {} file'.format(fileType))
            
        except Exception as e:
            logger.error('Failed to adjust {} file: {}'.format(fileType, e))
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info('Finished adjusting topo geometry files')
    logger.info('Backup files created with .backup extension')
    logger.info('You can compare the adjusted files with the originals to verify the correction')


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()
