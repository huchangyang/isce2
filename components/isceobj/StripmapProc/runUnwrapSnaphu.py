#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Piyush Agram
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






# giangi: taken Piyush code for snaphu and adapted

import sys
import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from isceobj.Constants import SPEED_OF_LIGHT
import copy
import os

def runSnaphu(self, igramSpectrum = "full", costMode = None,initMethod = None, defomax = None, initOnly = None):

    if costMode is None:
        costMode   = 'DEFO'
    
    if initMethod is None:
        initMethod = 'MST'
    
    if  defomax is None:
        defomax = 4.0
    
    if initOnly is None:
        initOnly = False
   
    print("igramSpectrum: ", igramSpectrum)

    if igramSpectrum == "full":
        ifgDirname = self.insar.ifgDirname

    elif igramSpectrum == "low":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferogram unwrapping')
            return
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)

    elif igramSpectrum == "high":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferogram unwrapping')
            return
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)


    wrapName = os.path.join(ifgDirname , 'filt_' + self.insar.ifgFilename)

    if '.flat' in wrapName:
        unwrapName = wrapName.replace('.flat', '.unw')
    elif '.int' in wrapName:
        unwrapName = wrapName.replace('.int', '.unw')
    else:
        unwrapName = wrapName + '.unw'

    corName = os.path.join(ifgDirname , self.insar.coherenceFilename)

    referenceFrame = self._insar.loadProduct( self._insar.referenceSlcCropProduct)
    wavelength = referenceFrame.getInstrument().getRadarWavelength()
    img1 = isceobj.createImage()
    img1.load(wrapName + '.xml')
    width = img1.getWidth()
    #width      = self.insar.resampIntImage.width

    orbit = referenceFrame.orbit
    prf = referenceFrame.PRF
    elp = copy.copy(referenceFrame.instrument.platform.planet.ellipsoid)
    sv = orbit.interpolate(referenceFrame.sensingMid, method='hermite')
    hdg = orbit.getHeading()
    llh = elp.xyz_to_llh(sv.getPosition())
    elp.setSCH(llh[0], llh[1], hdg)

    earthRadius = elp.pegRadCur
    sch, vsch = elp.xyzdot_to_schdot(sv.getPosition(), sv.getVelocity())
    azimuthSpacing = vsch[0] * earthRadius / ((earthRadius + sch[2]) *prf)


    earthRadius = elp.pegRadCur
    altitude   = sch[2]
    rangeLooks = 1  # self.numberRangeLooks #insar.topo.numberRangeLooks
    azimuthLooks = 1 # self.numberAzimuthLooks #insar.topo.numberAzimuthLooks

    if not self.numberAzimuthLooks:
        self.numberAzimuthLooks = 1

    if not self.numberRangeLooks:
        self.numberRangeLooks = 1

    azres = referenceFrame.platform.antennaLength/2.0
    azfact = self.numberAzimuthLooks * azres / azimuthSpacing

    rBW = referenceFrame.instrument.pulseLength * referenceFrame.instrument.chirpSlope
    rgres = abs(SPEED_OF_LIGHT / (2.0 * rBW))
    rngfact = rgres/referenceFrame.getInstrument().getRangePixelSize()

    corrLooks = self.numberRangeLooks * self.numberAzimuthLooks/(azfact*rngfact) 
    maxComponents = 20

    snp = Snaphu()
    snp.setInitOnly(initOnly)
    snp.setInput(wrapName)
    snp.setOutput(unwrapName)
    snp.setWidth(width)
    snp.setCostMode(costMode)
    snp.setEarthRadius(earthRadius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(corName)
    snp.setInitMethod(initMethod)
    #snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(maxComponents)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(rangeLooks)
    snp.setAzimuthLooks(azimuthLooks)
    snp.setCorFileFormat('FLOAT_DATA')
    snp.prepare()
    snp.unwrap()
    ######Render XML
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setAccessMode('read')
    outImage.renderHdr()
    outImage.renderVRT()
    #####Check if connected components was created
    if snp.dumpConnectedComponents:
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName+'.conncomp')
        #At least one can query for the name used
        self.insar.connectedComponentsFilename = unwrapName+'.conncomp'
        connImage.setWidth(width)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        connImage.renderHdr()
        connImage.renderVRT()

    return

'''
def runUnwrapMcf(self):
    runSnaphu(self, igramSpectrum = "full", costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    runSnaphu(self, igramSpectrum = "low", costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    runSnaphu(self, igramSpectrum = "high", costMode = 'SMOOTH',initMethod = 'MCF', defomax = 2, initOnly = True)
    return
'''

def runUnwrap(self, igramSpectrum = "full"):
    """
    Automatically choose appropriate unwrapping method based on image size.
    Use tiled processing when either dimension exceeds 32000 pixels.
    """
    # Get interferogram directory
    if igramSpectrum == "full":
        ifgDirname = self.insar.ifgDirname
    elif igramSpectrum == "low":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferogram unwrapping')
            return
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
    elif igramSpectrum == "high":
        if not self.doDispersive:
            print('Estimating dispersive phase not requested ... skipping sub-band interferogram unwrapping')
            return
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)

    # Get interferogram filename
    wrapName = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename)
    
    # Read image dimensions
    img1 = isceobj.createImage()
    img1.load(wrapName + '.xml')
    width = img1.getWidth()
    length = img1.getLength()
    
    print(f"Image dimensions: width={width}, length={length}")
    
    # Choose processing method based on image size
    if width > 32000 or length > 32000:
        print("Large image detected. Using tiled unwrapping...")
        
        # Calculate number of tiles and tile size
        # Target tile size is about 20000 pixels for efficient processing
        target_tile_size = 20000
        
        # Calculate required number of tiles (round up)
        num_tiles_x = (width + target_tile_size - 1) // target_tile_size
        num_tiles_y = (length + target_tile_size - 1) // target_tile_size
        
        # Calculate actual tile size (ensure full image coverage)
        tile_width = width // num_tiles_x
        tile_length = length // num_tiles_y
        
        # Set overlap region (minimum of 10% tile size and 300 pixels)
        overlap_x = min(int(tile_width * 0.1), 300)
        overlap_y = min(int(tile_length * 0.1), 300)
        
        print(f"Tiling parameters:")
        print(f"Number of tiles: {num_tiles_x}x{num_tiles_y}")
        print(f"Tile size: {tile_width}x{tile_length}")
        print(f"Overlap size: {overlap_x}x{overlap_y}")
        
        # Use tiled processing
        self.runSnaphuWithTiling(
            igramSpectrum=igramSpectrum,
            costMode='SMOOTH',
            initMethod='MCF',
            defomax=2,
            tile_rows=num_tiles_y,  # Note: passing number of tiles, not tile size
            tile_cols=num_tiles_x,
            overlap_row=overlap_y,
            overlap_col=overlap_x
        )
    else:
        print("Using standard unwrapping...")
        # Use standard processing
        runSnaphu(
            self,
            igramSpectrum=igramSpectrum,
            costMode='SMOOTH',
            initMethod='MCF',
            defomax=2,
            initOnly=True
        )
    
    return

def runSnaphuWithTiling(self, igramSpectrum="full", costMode=None, initMethod=None, defomax=None, 
                       tile_rows=None, tile_cols=None, overlap_row=None, overlap_col=None):
    """
    Run snaphu with tiling for large interferograms
    
    Parameters:
    -----------
    igramSpectrum : str
        Spectrum type ('full', 'low', or 'high')
    costMode : str
        Cost mode for unwrapping ('DEFO', 'SMOOTH', or 'TOPO')
    initMethod : str
        Initialization method ('MST' or 'MCF')
    defomax : float
        Maximum deformation in cycles
    tile_rows : int
        Number of tiles in row direction
    tile_cols : int
        Number of tiles in column direction
    overlap_row : int
        Overlap size in row direction
    overlap_col : int
        Overlap size in column direction
    """
    
    if costMode is None:
        costMode = 'DEFO'
    if initMethod is None:
        initMethod = 'MST'
    if defomax is None:
        defomax = 4.0
    if tile_rows is None or tile_cols is None:
        raise ValueError("Tile dimensions must be specified")
    if overlap_row is None or overlap_col is None:
        raise ValueError("Overlap dimensions must be specified")

    # Determine interferogram directory
    if igramSpectrum == "full":
        ifgDirname = self.insar.ifgDirname
    elif igramSpectrum == "low":
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.lowBandSlcDirname)
    elif igramSpectrum == "high":
        ifgDirname = os.path.join(self.insar.ifgDirname, self.insar.highBandSlcDirname)

    # Set file names
    wrapName = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename)
    if '.flat' in wrapName:
        unwrapName = wrapName.replace('.flat', '.unw')
    elif '.int' in wrapName:
        unwrapName = wrapName.replace('.int', '.unw')
    else:
        unwrapName = wrapName + '.unw'
    corName = os.path.join(ifgDirname, self.insar.coherenceFilename)

    # Get image dimensions
    img1 = isceobj.createImage()
    img1.load(wrapName + '.xml')
    width = img1.getWidth()
    length = img1.getLength()

    # Create snaphu configuration file
    conf_file = "snaphu.conf"
    with open(conf_file, 'w') as f:
        # Basic parameters
        f.write(f"INFILE {wrapName}\n")
        f.write(f"OUTFILE {unwrapName}\n")
        f.write(f"CORRFILE {corName}\n")
        f.write(f"LINELENGTH {width}\n")
        f.write(f"NLINES {length}\n")
        
        # Tile parameters
        f.write(f"NTILEROW {tile_rows}\n")
        f.write(f"NTILECOL {tile_cols}\n")
        f.write(f"ROWOVRLP {overlap_row}\n")
        f.write(f"COLOVRLP {overlap_col}\n")
        
        # Cost mode setting
        if costMode.upper() == 'DEFO':
            f.write("STATCOSTMODE DEFO\n")
        elif costMode.upper() == 'SMOOTH':
            f.write("STATCOSTMODE SMOOTH\n")
        elif costMode.upper() == 'TOPO':
            f.write("STATCOSTMODE TOPO\n")
            
        # Initialization method
        if initMethod.upper() == 'MCF':
            f.write("INITMETHOD MCF\n")
        else:
            f.write("INITMETHOD MST\n")
            
        # Deformation maximum cycles
        if defomax:
            f.write(f"DEFOMAX_CYCLE {defomax}\n")

    # Build and execute snaphu command
    cmd = f"snaphu -f {conf_file}"
    print(f"Executing command: {cmd}")
    status = os.system(cmd)
    
    if status != 0:
        raise Exception('Snaphu execution failed')

    # Create output image metadata
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setAccessMode('read')
    outImage.renderHdr()
    outImage.renderVRT()

    return


