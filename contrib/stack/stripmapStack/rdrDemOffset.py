#!/usr/bin/env python3

import os
import argparse
import shelve
import datetime
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
    db = shelve.open(os.path.join(inps.reference, 'data'))
    frame = db['frame']
    db.close()
    
    # Create geometry directory if it doesn't exist
    os.makedirs(inps.outdir, exist_ok=True)
    
    # Save absolute path of geometry directory BEFORE creating working directory
    originalGeometryDir = os.path.abspath(inps.outdir)
    
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
        
        # Call the core function with originalGeometryDir
        logger.info('Calling StripmapProc rdrDemOffset...')
        rdrDemOffset(adapter, referenceInfo, heightFilename, referenceSlc, catalog=None, originalGeometryDir=originalGeometryDir)
    finally:
        os.chdir(cwd)
    
    # Update frame with corrected geometry if offsets were estimated
    if abs(adapter._insar.radarDemRangeOffset) > 0.01 or abs(adapter._insar.radarDemAzimuthOffset) > 0.01:
        logger.info('Updating frame with corrected geometry')
        frame.startingRange = referenceInfo.startingRange
        frame.sensingStart = referenceInfo.sensingStart
        
        # Save updated frame
        db = shelve.open(os.path.join(inps.reference, 'data'), writeback=True)
        db['frame'] = frame
        db.close()
        logger.info('Saved updated frame with corrected geometry')
        
        # Multilook geometry files if needed (same as topo.py)
        # Always perform multilook after updating .full files to ensure .rdr files are regenerated
        # from the updated geometry, even if alks=1, rlks=1 (which means no actual downsampling)
        # Ensure alks and rlks are integers (they might be strings from config file)
        alks = int(inps.alks) if hasattr(inps, 'alks') else 1
        rlks = int(inps.rlks) if hasattr(inps, 'rlks') else 1
        logger.info('Multilook parameters: alks={}, rlks={} (product={})'.format(alks, rlks, alks * rlks))
        
        # Always run multilook to regenerate .rdr files from updated .full files
        # This ensures consistency even if alks=1, rlks=1 (no downsampling, but still regenerates files)
        logger.info('Multilooking geometry files with alks={}, rlks={} (regenerating .rdr from updated .full)'.format(alks, rlks))
        # Import multilook function from topo.py
        # Use absolute import to ensure we get the right module
        import sys
        import importlib.util
        topo_path = os.path.join(os.path.dirname(__file__), 'topo.py')
        spec = importlib.util.spec_from_file_location("topo", topo_path)
        topo_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(topo_module)
        runMultilook = topo_module.runMultilook
        
        # Determine output directory (same logic as topo.py)
        # topo.py uses: os.path.join(os.path.dirname(os.path.dirname(info.outdir)), 'geom_reference')
        # But in rdrDemOffset, outdir is already the geometry directory
        # So we need to find the parent directory structure
        # Typically: workDir/stack_folder/geom_reference -> workDir/stack_folder/geom_reference (same)
        out_dir = inps.outdir  # Use the same directory for multilooked files
        
        # Run multilook on the updated .full files
        # Input: .full files in geometry directory
        # Output: multilooked .rdr files in the same directory (overwriting existing ones)
        runMultilook(in_dir=inps.outdir, out_dir=out_dir, alks=alks, rlks=rlks,
                    in_ext='.rdr.full', out_ext='.rdr', method='gdal',
                    fbase_list=['hgt', 'lat', 'lon', 'los'])
        logger.info('Multilooked geometry files generated successfully')
    
    logger.info('rdr_dem_offset completed successfully')
    return


if __name__ == '__main__':
    '''
    Main driver.
    '''
    main()
