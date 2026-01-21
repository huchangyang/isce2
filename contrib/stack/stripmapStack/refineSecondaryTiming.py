#!/usr/bin/env python3

import numpy as np 
import argparse
import os
import isce
import isceobj
import shelve
import datetime
from isceobj.Location.Offset import OffsetField
from iscesys.StdOEL.StdOELPy import create_writer
from mroipac.ampcor.Ampcor import Ampcor
import pickle


def createParser():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='Generate offset field between two Sentinel swaths')
    parser.add_argument('-m','--reference', type=str, dest='reference', required=True,
            help='Reference image')
    parser.add_argument('--mm', type=str, dest='metareference', default=None,
            help='Reference meta data dir')
    parser.add_argument('-s', '--secondary', type=str, dest='secondary', required=True,
            help='Secondary image')
    parser.add_argument('--ss', type=str, dest='metasecondary', default=None,
            help='Secondary meta data dir')
    parser.add_argument('-o', '--outfile',type=str, required=True, dest='outfile',
            help='Misregistration in subpixels')

    parser.add_argument('--aa', dest='azazorder', type=int, default=0,
            help = 'Azimuth order of azimuth offsets')
    parser.add_argument('--ar', dest='azrgorder', type=int, default=0,
            help = 'Range order of azimuth offsets')

    parser.add_argument('--ra', dest='rgazorder', type=int, default=0,
            help = 'Azimuth order of range offsets')
    parser.add_argument('--rr', dest='rgrgorder', type=int, default=0,
            help = 'Range order of range offsets')
    parser.add_argument('--ao', dest='azoff', type=int, default=0,
            help='Azimuth gross offset')
    parser.add_argument('--ro', dest='rgoff', type=int, default=0,
            help='Range gross offset')
    parser.add_argument('-t', '--thresh', dest='snrthresh', type=float, default=5.0,
            help='SNR threshold')
    parser.add_argument('--water-mask', dest='watermask', type=str, default=None,
            help='Water mask file to exclude water pixels. Can be GeoTIFF, binary file, or numpy array. Water pixels should be non-zero. If not provided, will auto-detect from run_01_reference output (geom_reference/waterMask.rdr).')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def estimateOffsetField(reference, secondary, azoffset=0, rgoffset=0, searchWindowSizes=None, startWindowIndex=0):
    '''
    Estimate offset field between burst and simamp.
    Uses adaptive search window: tries smaller windows first, increases if needed.
    
    Args:
        reference: Reference image path
        secondary: Secondary image path
        azoffset: Azimuth gross offset
        rgoffset: Range gross offset
        searchWindowSizes: List of search window sizes to try (default: [20, 50, 100])
        startWindowIndex: Index in searchWindowSizes to start from (default: 0)
    
    Returns:
        (OffsetField object, window_size_used): Tuple of field and the window size that was used
        The returned field contains the original offsets with sigma information preserved
    '''
    if searchWindowSizes is None:
        searchWindowSizes = [20, 50, 100]  # Try small first, then larger if needed
    
    minPoints = 50  # Minimum number of points to consider match successful
    result = None
    windowSizeUsed = None
    
    for i in range(startWindowIndex, len(searchWindowSizes)):
        searchSize = searchWindowSizes[i]
        print(f'Trying search window size: {searchSize}x{searchSize}')

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
        objOffset.setWindowSizeWidth(256)
        objOffset.setWindowSizeHeight(256)
        objOffset.setSearchWindowSizeWidth(searchSize)
        objOffset.setSearchWindowSizeHeight(searchSize)
        margin = 2*objOffset.searchWindowSizeWidth + objOffset.windowSizeWidth

        objOffset.thresholdSNR = 0.01

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

        current_result = objOffset.getOffsetField()
        numPoints = len(current_result._offsets)

        sar.finalizeImage()
        sim.finalizeImage()

        print(f'  Found {numPoints} offset points with search window {searchSize}x{searchSize}')
        
        # Save result and window size
        result = current_result
        windowSizeUsed = searchSize
        
        # If we have enough points, use this result and exit early (no need for larger window)
        if numPoints >= minPoints:
            print(f'  Successfully matched with {numPoints} points using search window {searchSize}x{searchSize}')
            return result, windowSizeUsed
        
        # If points are too few and there are larger windows available, try next window
        if numPoints < minPoints and i < len(searchWindowSizes) - 1:
            print(f'  Too few points ({numPoints} < {minPoints}), trying larger search window...')
            continue
        
        # This is the last window and we still don't have enough points
        # Use this result anyway (best we can do)
        print(f'  WARNING: Only {numPoints} points found even with largest search window {searchSize}x{searchSize}')
        return result, windowSizeUsed


def loadWaterMask(maskfile, reference_image):
    '''
    Load water mask file. Supports multiple formats:
    - GeoTIFF/other GDAL formats
    - NumPy .npy file
    - Binary file (same size as reference image)
    
    Returns: 2D numpy array where non-zero values indicate water pixels
    '''
    if maskfile is None or not os.path.exists(maskfile):
        return None
    
    try:
        # Try GDAL first (for GeoTIFF, etc.)
        try:
            from osgeo import gdal
            ds = gdal.Open(maskfile, gdal.GA_ReadOnly)
            if ds is not None:
                mask = ds.GetRasterBand(1).ReadAsArray()
                ds = None
                print('Loaded water mask from GDAL file: %s (shape: %s)' % (maskfile, mask.shape))
                return mask
        except:
            pass
        
        # Try NumPy format
        if maskfile.endswith('.npy'):
            mask = np.load(maskfile)
            print('Loaded water mask from NumPy file: %s (shape: %s)' % (maskfile, mask.shape))
            return mask
        
        # Try loading as ISCE .rdr file (binary format with XML metadata)
        rdr_xml = maskfile + '.xml' if not maskfile.endswith('.xml') else maskfile
        if maskfile.endswith('.rdr') or os.path.exists(rdr_xml):
            try:
                # Try using ISCE image reader
                mask_img = isceobj.createImage()
                mask_img.load(rdr_xml)
                mask_img.setAccessMode('read')
                mask_img.createImage()
                width = mask_img.getWidth()
                length = mask_img.getLength()
                
                # Read the mask data
                mask = np.zeros((length, width), dtype=np.uint8)
                for i in range(length):
                    line = mask_img.getLine(i)
                    if isinstance(line, np.ndarray):
                        mask[i, :] = line
                    else:
                        # Convert to numpy array if needed
                        mask[i, :] = np.array(line, dtype=np.uint8)
                
                mask_img.finalizeImage()
                print('Loaded water mask from ISCE .rdr file: %s (shape: %s)' % (maskfile, mask.shape))
                return mask
            except Exception as e:
                # If .rdr loading fails, try other methods
                pass
        
        # Try loading as binary file (need reference image dimensions)
        if reference_image is not None:
            try:
                ref_img = isceobj.createSlcImage()
                ref_img.load(reference_image + '.xml')
                ref_img.setAccessMode('READ')
                ref_img.createImage()
                width = ref_img.getWidth()
                length = ref_img.getLength()
                ref_img.finalizeImage()
                
                # Try loading as binary file
                mask = np.fromfile(maskfile, dtype=np.uint8).reshape(length, width)
                print('Loaded water mask from binary file: %s (shape: %s)' % (maskfile, mask.shape))
                return mask
            except:
                pass
        
        print('WARNING: Could not load water mask from %s, skipping' % maskfile)
        return None
    except Exception as e:
        print('WARNING: Error loading water mask: %s, skipping' % str(e))
        return None


def findWaterMaskPath(reference_path, outfile_path, explicit_mask=None):
    '''
    Automatically find water mask path from run_01_reference output.
    Tries multiple possible locations based on typical ISCE workflow structure.
    
    Args:
        reference_path: Path to reference SLC
        outfile_path: Path to output file (used to infer workDir)
        explicit_mask: Explicitly provided mask path (takes priority)
    
    Returns:
        Path to water mask file, or None if not found
    '''
    # If explicitly provided, use it
    if explicit_mask is not None:
        if os.path.exists(explicit_mask):
            return explicit_mask
        else:
            print('WARNING: Explicitly provided water mask not found: %s' % explicit_mask)
            return None
    
    # Try to infer workDir from outfile path
    # Typical structure: workDir/refineSecondaryTiming/pairs/date1_date2/misreg
    # or: workDir/refineSecondaryTiming/dates/date/misreg
    possible_paths = []
    
    # Method 1: From outfile path
    # Typical structure: workDir/refineSecondaryTiming/pairs/date1_date2/misreg
    # or: workDir/refineSecondaryTiming/dates/date/misreg
    if outfile_path:
        outfile_dir = os.path.dirname(os.path.abspath(outfile_path))
        # Try going up to workDir level
        if 'refineSecondaryTiming' in outfile_dir:
            workDir = outfile_dir.split('refineSecondaryTiming')[0].rstrip(os.sep)
            # Try merged/geom_reference first (full resolution)
            possible_paths.append(os.path.join(workDir, 'merged', 'geom_reference', 'waterMask.rdr'))
            # Fallback to geom_reference (multilooked)
            possible_paths.append(os.path.join(workDir, 'geom_reference', 'waterMask.rdr'))
    
    # Method 2: From reference path
    # Typical: workDir/coregSLC/Coarse/date/date.slc
    # or: workDir/merged/SLC/date/date.slc
    if reference_path:
        ref_path_abs = os.path.abspath(reference_path)
        ref_dir = os.path.dirname(ref_path_abs)
        parts = ref_dir.split(os.sep)
        
        # Find workDir by going up from coregSLC, merged/SLC, or SLC
        if 'coregSLC' in parts:
            idx = parts.index('coregSLC')
            workDir = os.sep.join(parts[:idx])
            # Try merged/geom_reference first (full resolution)
            possible_paths.append(os.path.join(workDir, 'merged', 'geom_reference', 'waterMask.rdr'))
            possible_paths.append(os.path.join(workDir, 'geom_reference', 'waterMask.rdr'))
        elif 'merged' in parts and 'SLC' in parts:
            # merged/SLC structure
            idx = parts.index('merged')
            workDir = os.sep.join(parts[:idx])
            possible_paths.append(os.path.join(workDir, 'merged', 'geom_reference', 'waterMask.rdr'))
        elif 'SLC' in parts:
            idx = parts.index('SLC')
            workDir = os.sep.join(parts[:idx])
            # Try merged/geom_reference first (full resolution)
            possible_paths.append(os.path.join(workDir, 'merged', 'geom_reference', 'waterMask.rdr'))
            possible_paths.append(os.path.join(workDir, 'geom_reference', 'waterMask.rdr'))
        
        # Also try going up a few levels from reference directory
        for level in [1, 2, 3]:
            try:
                parent = os.path.dirname(ref_dir)
                for _ in range(level - 1):
                    parent = os.path.dirname(parent)
                # Try merged/geom_reference first
                possible_paths.append(os.path.join(parent, 'merged', 'geom_reference', 'waterMask.rdr'))
                possible_paths.append(os.path.join(parent, 'geom_reference', 'waterMask.rdr'))
            except:
                pass
    
    # Try each possible path
    for path in possible_paths:
        if os.path.exists(path):
            print('Found water mask at: %s' % path)
            return path
        # Also try without .xml extension
        if path.endswith('.xml'):
            path_no_xml = path[:-4]
            if os.path.exists(path_no_xml):
                print('Found water mask at: %s' % path_no_xml)
                return path_no_xml
    
    return None


def applyWaterMask(field, water_mask, reference_image=None):
    '''
    Apply water mask to filter out offset points in water regions.
    Note: In waterMask.rdr, 1 = land, 0 = water. We filter out water pixels (value = 0).
    
    Args:
        field: OffsetField object
        water_mask: 2D numpy array (None if no mask), 1=land, 0=water
        reference_image: Reference image path (for coordinate mapping if needed)
    
    Returns:
        Filtered OffsetField
    '''
    if water_mask is None:
        return field
    
    print('Applying water mask to filter out water pixels...')
    original_count = len(field._offsets)
    
    filtered_offsets = []
    removed_water = 0
    
    for offsetx in field._offsets:
        fields = "{}".format(offsetx).split()
        if len(fields) < 4:
            filtered_offsets.append(offsetx)  # Keep malformed entries for now
            continue
        
        try:
            # Get pixel coordinates
            x = int(float(fields[0]))  # Range sample (column)
            y = int(float(fields[2]))  # Azimuth line (row)
            
            # Check if within mask bounds
            if y >= 0 and y < water_mask.shape[0] and x >= 0 and x < water_mask.shape[1]:
                # In waterMask.rdr: 1 = land, 0 = water
                # Filter out water pixels (value = 0), keep land pixels (value = 1)
                if water_mask[y, x] == 0:
                    removed_water += 1
                    continue
            
            # Keep non-water points
            filtered_offsets.append(offsetx)
        except (ValueError, IndexError) as e:
            # If coordinate conversion fails, keep the point
            filtered_offsets.append(offsetx)
            continue
    
    field._offsets = filtered_offsets
    print('After water mask filtering: %d points left (removed %d water pixels)' % 
          (len(filtered_offsets), removed_water))
    
    return field


def fitOffsets(field,azrgOrder=0,azazOrder=0,
        rgrgOrder=0,rgazOrder=0,snr=5.0,water_mask=None,reference_image=None):
    '''
    Estimate constant range and azimuth shifts.
    Robust outlier removal inspired by autoRIFT:
    1. Initial SNR filtering
    2. Iterative MAD-based outlier removal (similar to autoRIFT's filtDisp)
    3. Residual-based filtering after polynomial fitting
    '''


    # Keep a copy of the original field so that we can access the original
    # per-point covariance / standard deviation information (sigmax, sigmay)
    # after Offoutliers has created a refined subset without covariance.
    originalField = field

    print('Starting with %d offset points' % len(field._offsets))
    
    # Step 0: Apply water mask if provided (before any other filtering)
    if water_mask is not None:
        field = applyWaterMask(field, water_mask, reference_image)

    stdWriter = create_writer("log","",True,filename='off.log')

    # Step 1: Initial SNR filtering with large distance threshold
    # This preserves points even with large systematic offsets
    objOff = isceobj.createOffoutliers()
    objOff.wireInputPort(name='offsets', object=field)
    objOff.setSNRThreshold(snr)
    objOff.setDistance(100.0)  # Large distance to avoid removing valid large offsets
    objOff.setStdWriter(stdWriter)
    objOff.offoutliers()
    field = objOff.getRefinedOffsetField()
    print('After initial SNR filtering: %d points left' % len(field._offsets))
    
    # Step 2: Sigma filtering (moved earlier as it's most effective)
    # This is based on per-point matching uncertainty, independent of offset values
    print('Points before sigma culling: %d' % len(field._offsets))
    
    # Adaptive sigma threshold: stricter if we have many points, more lenient if few
    if len(field._offsets) >= 20:
        sigmaThreshold = 0.001  # Standard threshold
    elif len(field._offsets) >= 10:
        sigmaThreshold = 0.01   # More lenient
    else:
        sigmaThreshold = 0.1    # Very lenient to preserve points
        print('WARNING: Few points remaining, using lenient sigma threshold: %.4f' % sigmaThreshold)
    
    # Build a lookup from original offsets using (x, y) as key
    originalOffsetMap = {}
    for offsetx in originalField:
        fields = "{}".format(offsetx).split()
        if len(fields) >= 8:
            key = (fields[0], fields[2])  # x, y
            originalOffsetMap[key] = fields

    sigma_filtered_offsets = []
    removedSigma = 0
    
    # Only perform sigma filtering if we have enough points
    if len(field._offsets) < 3:
        print('Too few points for sigma culling, skipping')
        sigma_filtered_offsets = field._offsets
    else:
        for offsetx in field:
            fields = "{}".format(offsetx).split()
            if len(fields) < 4:
                removedSigma += 1
                continue

            key = (fields[0], fields[2])
            orig_fields = originalOffsetMap.get(key, None)
            if (orig_fields is None) or (len(orig_fields) < 8):
                removedSigma += 1
                continue

            sigma_rg = float(orig_fields[5])  # sigmax
            sigma_az = float(orig_fields[6])  # sigmay

            if (abs(sigma_rg) > sigmaThreshold) or (abs(sigma_az) > sigmaThreshold):
                removedSigma += 1
                continue

            sigma_filtered_offsets.append(offsetx)
        
        print('After sigma culling (threshold %.4f): %d points left (removed %d)' %
              (sigmaThreshold, len(sigma_filtered_offsets), removedSigma))
        
        # If too many points removed, relax threshold and retry
        if len(sigma_filtered_offsets) < len(field._offsets) * 0.3 and len(field._offsets) >= 5:
            print('Too many points removed, retrying with relaxed threshold')
            relaxed_threshold = sigmaThreshold * 10.0
            sigma_filtered_offsets = []
            removedSigma = 0
            for offsetx in field:
                fields = "{}".format(offsetx).split()
                if len(fields) < 4:
                    removedSigma += 1
                    continue
                key = (fields[0], fields[2])
                orig_fields = originalOffsetMap.get(key, None)
                if (orig_fields is None) or (len(orig_fields) < 8):
                    removedSigma += 1
                    continue
                sigma_rg = float(orig_fields[5])
                sigma_az = float(orig_fields[6])
                if (abs(sigma_rg) > relaxed_threshold) or (abs(sigma_az) > relaxed_threshold):
                    removedSigma += 1
                    continue
                sigma_filtered_offsets.append(offsetx)
            print('After relaxed sigma culling (threshold %.4f): %d points left' %
                  (relaxed_threshold, len(sigma_filtered_offsets)))
    
    field._offsets = sigma_filtered_offsets
    
    # Step 3: Residual-based filtering (after polynomial fitting)
    # This is more robust than distance-based methods for spatially varying offsets
    if len(field._offsets) >= 5:
        # Extract offset values and locations
        rg_offsets = []
        az_offsets = []
        x_coords = []
        y_coords = []
        valid_offsets = []
        
        for offsetx in field._offsets:
            fields = "{}".format(offsetx).split()
            if len(fields) >= 4:
                try:
                    # Offset string format: x dx y dy snr sigmax sigmay sigmaxy
                    # fields[0] = x (range location)
                    # fields[1] = dx (range offset)
                    # fields[2] = y (azimuth location)
                    # fields[3] = dy (azimuth offset)
                    x = float(fields[0])
                    y = float(fields[2])
                    rg_val = float(fields[1])  # dx = range offset
                    az_val = float(fields[3])  # dy = azimuth offset
                    x_coords.append(x)
                    y_coords.append(y)
                    rg_offsets.append(rg_val)
                    az_offsets.append(az_val)
                    valid_offsets.append(offsetx)
                except:
                    continue
        
        if len(rg_offsets) >= 5:
            rg_offsets = np.array(rg_offsets)
            az_offsets = np.array(az_offsets)
            x_coords = np.array(x_coords)
            y_coords = np.array(y_coords)
            
            # autoRIFT-style iterative MAD filtering
            # Parameters similar to autoRIFT's DISP_FILT class
            MadScalar = 4.0  # Similar to autoRIFT's default
            Iter = 3  # Number of iterations
            
            # Initialize mask (all points valid initially)
            mask = np.ones(len(valid_offsets), dtype=bool)
            
            # Iterative MAD-based outlier removal (inspired by autoRIFT's filtDisp)
            for iteration in range(Iter):
                if np.sum(mask) < 3:
                    break
                
                # Calculate median and MAD for current valid points
                rg_valid = rg_offsets[mask]
                az_valid = az_offsets[mask]
                
                rg_median = np.median(rg_valid)
                az_median = np.median(az_valid)
                
                rg_mad = np.median(np.abs(rg_valid - rg_median))
                az_mad = np.median(np.abs(az_valid - az_median))
                
                # Avoid division by zero
                if rg_mad < 1e-10:
                    rg_mad = np.std(rg_valid) if len(rg_valid) > 1 else 1.0
                if az_mad < 1e-10:
                    az_mad = np.std(az_valid) if len(az_valid) > 1 else 1.0
                
                # Calculate thresholds (similar to autoRIFT)
                rg_threshold = MadScalar * rg_mad
                az_threshold = MadScalar * az_mad
                
                # Minimum threshold to avoid being too strict
                # Similar to autoRIFT's DxMadmin/DyMadmin
                min_threshold = 0.1  # Minimum pixel threshold
                rg_threshold = max(rg_threshold, min_threshold)
                az_threshold = max(az_threshold, min_threshold)
                
                # Update mask: keep points within threshold
                new_mask = (
                    (np.abs(rg_offsets - rg_median) <= rg_threshold) &
                    (np.abs(az_offsets - az_median) <= az_threshold)
                )
                
                removed = np.sum(mask) - np.sum(new_mask)
                print('Iteration %d: %d points valid, removed %d outliers (rg_mad=%.3f, az_mad=%.3f)' %
                      (iteration + 1, np.sum(new_mask), removed, rg_mad, az_mad))
                
                # If removing too many points, stop and use previous mask
                if removed > np.sum(mask) * 0.5:
                    print('Too many points removed (%.1f%%), stopping iteration' %
                          (100.0 * removed / np.sum(mask)))
                    break
                
                mask = new_mask
            
        # Extract offset values for residual calculation
        rg_offsets = []
        az_offsets = []
        valid_offsets = []
        
        for offsetx in field._offsets:
            fields = "{}".format(offsetx).split()
            if len(fields) >= 4:
                try:
                    x = float(fields[0])
                    y = float(fields[2])
                    rg_val = float(fields[1])  # dx = range offset
                    az_val = float(fields[3])  # dy = azimuth offset
                    rg_offsets.append(rg_val)
                    az_offsets.append(az_val)
                    valid_offsets.append(offsetx)
                except:
                    continue
        
        if len(valid_offsets) >= max(3, (azazOrder + 1) * (azrgOrder + 1), (rgazOrder + 1) * (rgrgOrder + 1)):
            try:
                # Perform polynomial fit on sigma-filtered points
                temp_field = OffsetField()
                temp_field._offsets = valid_offsets
                
                aa_init, dummy = temp_field.getFitPolynomials(
                    azimuthOrder=azazOrder, rangeOrder=azrgOrder, usenumpy=True)
                dummy, rr_init = temp_field.getFitPolynomials(
                    azimuthOrder=rgazOrder, rangeOrder=rgrgOrder, usenumpy=True)
                
                # Calculate residuals for all points
                residuals_rg = []
                residuals_az = []
                residual_mask = []
                
                for i, offsetx in enumerate(valid_offsets):
                    fields = "{}".format(offsetx).split()
                    if len(fields) < 4:
                        continue
                    try:
                        rg_obs = float(fields[1])  # dx = range offset
                        az_obs = float(fields[3])  # dy = azimuth offset
                        
                        # Evaluate polynomial (for 0-order, use constant term)
                        az_pred = aa_init._coeffs[0][0]
                        rg_pred = rr_init._coeffs[0][0]
                        
                        rg_residual = abs(rg_obs - rg_pred)
                        az_residual = abs(az_obs - az_pred)
                        
                        residuals_rg.append(rg_residual)
                        residuals_az.append(az_residual)
                        residual_mask.append(i)
                    except:
                        continue
                
                if len(residuals_rg) > 0:
                    residuals_rg = np.array(residuals_rg)
                    residuals_az = np.array(residuals_az)
                    
                    # Use MAD on residuals
                    rg_res_median = np.median(residuals_rg)
                    az_res_median = np.median(residuals_az)
                    rg_res_mad = np.median(np.abs(residuals_rg - rg_res_median))
                    az_res_mad = np.median(np.abs(residuals_az - az_res_median))
                    
                    if rg_res_mad < 1e-10:
                        rg_res_mad = np.std(residuals_rg) if len(residuals_rg) > 1 else 1.0
                    if az_res_mad < 1e-10:
                        az_res_mad = np.std(residuals_az) if len(residuals_az) > 1 else 1.0
                    
                    # Filter based on residual MAD
                    residual_threshold_factor = 3.0
                    residual_mask_array = (
                        (residuals_rg <= rg_res_median + residual_threshold_factor * rg_res_mad) &
                        (residuals_az <= az_res_median + residual_threshold_factor * az_res_mad)
                    )
                    
                    # Apply residual filtering
                    residual_filtered_offsets = [valid_offsets[residual_mask[i]] 
                                                 for i in range(len(residual_mask)) 
                                                 if residual_mask_array[i]]
                    
                    print('After residual-based filtering: %d points left (removed %d)' %
                          (len(residual_filtered_offsets), len(valid_offsets) - len(residual_filtered_offsets)))
                    
                    field._offsets = residual_filtered_offsets
                else:
                    print('Residual calculation failed, using sigma-filtered points')
            except Exception as e:
                print('Residual-based filtering failed, using sigma-filtered points: %s' % str(e))
        else:
            print('Too few points for residual filtering, using sigma-filtered points')
    
    # Step 4: Optional MAD filtering (as final cleanup, less critical now)
    # This can help remove any remaining outliers, but is less critical after sigma filtering
    if len(field._offsets) >= 10:  # Only if we have enough points
        # Too few points, use original method but with adaptive thresholds and rollback
        print('Too few points for robust filtering, using adaptive method')
        distance_thresholds = [10, 5, 3, 1]
        backup_offsets = field._offsets[:]  # Keep backup
        for distance in distance_thresholds:
            inpts = len(field._offsets)
            if inpts < 3:
                break
            objOff = isceobj.createOffoutliers()
            objOff.wireInputPort(name='offsets', object=field)
            objOff.setSNRThreshold(snr)
            objOff.setDistance(distance)
            objOff.setStdWriter(stdWriter)
            objOff.offoutliers()
            new_field = objOff.getRefinedOffsetField()
            removed = inpts - len(new_field._offsets)
            print('After distance threshold %.1f: %d points left (removed %d)' % 
                  (distance, len(new_field._offsets), removed))
            # If removing too many points, revert and stop
            if removed > inpts * 0.5:
                print('Too many points removed (%.1f%%), reverting to previous state' % 
                      (100.0 * removed / inpts))
                field._offsets = backup_offsets
                break
            # Update field and backup for next iteration
            field = new_field
            backup_offsets = field._offsets[:]


    # Check if we have enough points for polynomial fitting
    minPoints = max(3, (azazOrder + 1) * (azrgOrder + 1), (rgazOrder + 1) * (rgrgOrder + 1))
    if len(field._offsets) < minPoints:
        print('ERROR: Only %d points left, minimum %d points needed for polynomial fitting' % 
              (len(field._offsets), minPoints))
        print('Suggestions:')
        print('  1) Increase search window size (currently 100x100)')
        print('  2) Adjust gross offsets (--ao and --ro) to reduce initial misregistration')
        print('  3) Reduce SNR threshold (--thresh, current: %.2f)' % snr)
        raise ValueError('Insufficient offset points (%d) for polynomial fitting. Need at least %d points.' % 
                        (len(field._offsets), minPoints))

    aa, dummy = field.getFitPolynomials(azimuthOrder=azazOrder, rangeOrder=azrgOrder, usenumpy=True)
    dummy, rr = field.getFitPolynomials(azimuthOrder=rgazOrder, rangeOrder=rgrgOrder, usenumpy=True)

    azshift = aa._coeffs[0][0]
    rgshift = rr._coeffs[0][0]
    print('Estimated az shift: ', azshift)
    print('Estimated rg shift: ', rgshift)

    return (aa, rr), field


def main(iargs=None):
    '''
    Generate offset fields burst by burst.
    '''

    inps = cmdLineParse(iargs)

    # Load water mask first (needed for quality check)
    water_mask = None
    water_mask_path = None
    
    if inps.watermask is not None:
        water_mask_path = inps.watermask
    else:
        # Auto-detect from run_01_reference output
        water_mask_path = findWaterMaskPath(inps.reference, inps.outfile)
    
    if water_mask_path is not None:
        water_mask = loadWaterMask(water_mask_path, inps.reference)
        if water_mask is None:
            print('WARNING: Water mask file found but could not be loaded: %s' % water_mask_path)
    else:
        print('No water mask found, skipping water pixel filtering')

    # Try adaptive search window with quality check
    searchWindowSizes = [20, 50, 100]
    startWindowIndex = 0
    field = None
    windowSizeUsed = None
    
    while startWindowIndex < len(searchWindowSizes):
        field, windowSizeUsed = estimateOffsetField(inps.reference, inps.secondary,
                azoffset=inps.azoff, rgoffset=inps.rgoff,
                searchWindowSizes=searchWindowSizes, startWindowIndex=startWindowIndex)
        
        # Quick quality check: apply water mask, SNR, and sigma filtering
        # This mimics the actual filtering pipeline to get accurate point count
        test_field = OffsetField()
        test_field._offsets = field._offsets[:]  # Copy offsets (preserves sigma info)
        original_test_field = OffsetField()
        original_test_field._offsets = field._offsets[:]  # Keep original for sigma filtering
        
        if water_mask is not None:
            test_field = applyWaterMask(test_field, water_mask, inps.reference)
        
        # Quick SNR filtering
        stdWriter = create_writer("log","",True,filename='off.log')
        objOff = isceobj.createOffoutliers()
        objOff.wireInputPort(name='offsets', object=test_field)
        objOff.setSNRThreshold(inps.snrthresh)
        objOff.setDistance(100.0)
        objOff.setStdWriter(stdWriter)
        objOff.offoutliers()
        test_field = objOff.getRefinedOffsetField()
        
        # Quick sigma filtering (similar to fitOffsets)
        if len(test_field._offsets) >= 3:
            # Build lookup map from original field
            originalOffsetMap = {}
            for offsetx in original_test_field._offsets:
                fields = "{}".format(offsetx).split()
                if len(fields) >= 8:
                    key = (fields[0], fields[2])  # x, y
                    originalOffsetMap[key] = fields
            
            # Apply sigma filtering with adaptive threshold
            if len(test_field._offsets) >= 20:
                sigmaThreshold = 0.001
            elif len(test_field._offsets) >= 10:
                sigmaThreshold = 0.01
            else:
                sigmaThreshold = 0.1
            
            sigma_filtered_offsets = []
            for offsetx in test_field._offsets:
                fields = "{}".format(offsetx).split()
                if len(fields) < 4:
                    continue
                key = (fields[0], fields[2])
                orig_fields = originalOffsetMap.get(key, None)
                if (orig_fields is None) or (len(orig_fields) < 8):
                    continue
                sigma_rg = float(orig_fields[5])  # sigmax
                sigma_az = float(orig_fields[6])  # sigmay
                if (abs(sigma_rg) <= sigmaThreshold) and (abs(sigma_az) <= sigmaThreshold):
                    sigma_filtered_offsets.append(offsetx)
            
            # If too many removed, try relaxed threshold
            if len(sigma_filtered_offsets) < len(test_field._offsets) * 0.3 and len(test_field._offsets) >= 5:
                relaxed_threshold = sigmaThreshold * 10.0
                sigma_filtered_offsets = []
                for offsetx in test_field._offsets:
                    fields = "{}".format(offsetx).split()
                    if len(fields) < 4:
                        continue
                    key = (fields[0], fields[2])
                    orig_fields = originalOffsetMap.get(key, None)
                    if (orig_fields is None) or (len(orig_fields) < 8):
                        continue
                    sigma_rg = float(orig_fields[5])
                    sigma_az = float(orig_fields[6])
                    if (abs(sigma_rg) <= relaxed_threshold) and (abs(sigma_az) <= relaxed_threshold):
                        sigma_filtered_offsets.append(offsetx)
            
            test_field._offsets = sigma_filtered_offsets
        
        numPointsAfterFilter = len(test_field._offsets)
        print(f'After quick quality check (water mask + SNR + sigma): {numPointsAfterFilter} points left')
        
        # If we have enough points after filtering, use this result
        if numPointsAfterFilter >= 50:
            print(f'Quality check passed: {numPointsAfterFilter} points remain after filtering')
            break
        
        # If not enough points and there are larger windows, try next window
        if numPointsAfterFilter < 50 and startWindowIndex < len(searchWindowSizes) - 1:
            print(f'Quality check failed: only {numPointsAfterFilter} points after filtering, trying larger window...')
            startWindowIndex += 1
            continue
        
        # Last window, use it anyway
        print(f'WARNING: Only {numPointsAfterFilter} points after filtering even with largest window')
        break

    if os.path.exists(inps.outfile):
        os.remove(inps.outfile)

    outDir = os.path.dirname(inps.outfile)
    os.makedirs(outDir, exist_ok=True)

    if inps.metareference is not None:
        referenceShelveDir = os.path.join(outDir, 'referenceShelve')
        os.makedirs(referenceShelveDir, exist_ok=True)

        cmd = 'cp ' + inps.metareference + '/data* ' + referenceShelveDir
        os.system(cmd)
        

    if inps.metasecondary is not None:
        secondaryShelveDir = os.path.join(outDir, 'secondaryShelve')
        os.makedirs(secondaryShelveDir, exist_ok=True)
        cmd = 'cp ' + inps.metasecondary + '/data* ' + secondaryShelveDir
        os.system(cmd)

    rgratio = 1.0
    azratio = 1.0

    if (inps.metareference is not None) and (inps.metasecondary is not None):
        
       # with shelve.open( os.path.join(inps.metareference, 'data'), 'r') as db:
        with shelve.open( os.path.join(referenceShelveDir, 'data'), 'r') as db:
            mframe = db['frame']

       # with shelve.open( os.path.join(inps.metasecondary, 'data'), 'r') as db:
        with shelve.open( os.path.join(secondaryShelveDir, 'data'), 'r') as db:
            sframe = db['frame']

        rgratio = mframe.instrument.getRangePixelSize()/sframe.instrument.getRangePixelSize()
        azratio = sframe.PRF / mframe.PRF

    print ('*************************************')
    print ('rgratio, azratio: ', rgratio, azratio)
    print ('*************************************')       

    odb = shelve.open(inps.outfile)
    odb['raw_field']  = field
    shifts, cull = fitOffsets(field,azazOrder=inps.azazorder,
            azrgOrder=inps.azrgorder,
            rgazOrder=inps.rgazorder,
            rgrgOrder=inps.rgrgorder,
            snr=inps.snrthresh,
            water_mask=water_mask,
            reference_image=inps.reference)
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

if __name__ == '__main__':
    main()



