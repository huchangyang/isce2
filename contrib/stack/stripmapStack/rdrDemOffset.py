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


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()
