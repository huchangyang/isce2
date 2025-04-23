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
    # Get interferogram directory and filename
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
        
        # Calculate number of tiles needed to make each dimension < 32000
        num_tiles_x = (width + 31999) // 32000
        num_tiles_y = (length + 31999) // 32000
        
        # Calculate overlap size (20% of tile size, minimum 500 pixels)
        avg_tile_width = width // num_tiles_x
        avg_tile_length = length // num_tiles_y
        
        overlap_col = max(int(avg_tile_width * 0.2), 500)
        overlap_row = max(int(avg_tile_length * 0.2), 500)
        
        # Verify that tiles + overlap satisfy SNAPHU constraints
        if (num_tiles_y + overlap_row > length or 
            num_tiles_x + overlap_col > width or
            num_tiles_y * num_tiles_y > length or
            num_tiles_x * num_tiles_x > width):
            print("Warning: Adjusting overlap sizes to meet SNAPHU constraints")
            
            if num_tiles_y + overlap_row > length:
                overlap_row = max(0, length - num_tiles_y)
            if num_tiles_x + overlap_col > width:
                overlap_col = max(0, width - num_tiles_x)
        
        print("Tiling parameters:")
        print(f"Number of tiles: {num_tiles_x}x{num_tiles_y}")
        print(f"Average tile size: {avg_tile_width}x{avg_tile_length}")
        print(f"Overlap size: {overlap_col}x{overlap_row}")

        runSnaphuWithTiling(
            self,
            igramSpectrum=igramSpectrum,
            costMode='SMOOTH',
            initMethod='MCF',
            defomax=2,
            initOnly=False,
            tile_rows=num_tiles_y,
            tile_cols=num_tiles_x,
            overlap_row=overlap_row,
            overlap_col=overlap_col
        )
    else:
        print("Using standard unwrapping...")
        runSnaphu(
            self,
            igramSpectrum=igramSpectrum,
            costMode='SMOOTH',
            initMethod='MCF',
            defomax=2,
            initOnly=True
        )
    
    return

def runSnaphuWithTiling(self, igramSpectrum, costMode=None, initMethod=None, defomax=None, initOnly=False, 
                       tile_rows=1000, tile_cols=1000, overlap_row=100, overlap_col=100):
    """
    Run Snaphu with tiling for large images
    """
    # Set default parameters
    if costMode is None:
        costMode = 'DEFO'
    if initMethod is None:
        initMethod = 'MST'
    if defomax is None:
        defomax = 4.0

    # Get interferogram directory and filenames
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

    # Set file names
    wrapName = os.path.join(ifgDirname, 'filt_' + self.insar.ifgFilename)
    if '.flat' in wrapName:
        unwrapName = wrapName.replace('.flat', '.unw')
    elif '.int' in wrapName:
        unwrapName = wrapName.replace('.int', '.unw')
    else:
        unwrapName = wrapName + '.unw'
    corName = os.path.join(ifgDirname, self.insar.coherenceFilename)

    # Get image dimensions from XML file
    img1 = isceobj.createImage()
    img1.load(wrapName + '.xml')
    width = img1.getWidth()
    length = img1.getLength()

    # Get reference frame parameters (same as in runSnaphu)
    referenceFrame = self._insar.loadProduct(self._insar.referenceSlcCropProduct)
    wavelength = referenceFrame.getInstrument().getRadarWavelength()
    
    orbit = referenceFrame.orbit
    prf = referenceFrame.PRF
    elp = copy.copy(referenceFrame.instrument.platform.planet.ellipsoid)
    sv = orbit.interpolate(referenceFrame.sensingMid, method='hermite')
    hdg = orbit.getHeading()
    llh = elp.xyz_to_llh(sv.getPosition())
    elp.setSCH(llh[0], llh[1], hdg)

    earthRadius = elp.pegRadCur
    sch, vsch = elp.xyzdot_to_schdot(sv.getPosition(), sv.getVelocity())
    altitude = sch[2]

    rangeLooks = 1
    azimuthLooks = 1
    maxComponents = 20
    minRegionSize = 1000
    tileCostThreshold = 200
    # Create configuration file
    configName = os.path.join(ifgDirname, 'snaphu.conf')
    with open(configName, 'w') as f:
        # File format settings
        f.write('CORRFILEFORMAT  FLOAT_DATA\n')
        
        # Tile parameters
        f.write(f'NTILEROW  {tile_rows}\n')
        f.write(f'NTILECOL  {tile_cols}\n')
        f.write(f'ROWOVRLP  {overlap_row}\n')
        f.write(f'COLOVRLP  {overlap_col}\n')
        
        # Parameters from runSnaphu
        f.write(f'EARTHRADIUS  {earthRadius}\n')
        f.write(f'LAMBDA  {wavelength}\n')
        f.write(f'ALTITUDE  {altitude}\n')
        f.write(f'RANGERES  {rangeLooks}\n')
        f.write(f'AZRES  {azimuthLooks}\n')
        f.write(f'MAXNCOMPS  {maxComponents}\n')
        f.write(f'MINREGIONSIZE  {minRegionSize}\n')
        f.write(f'TILECOSTTHRESH  {tileCostThreshold}\n')
        
        if defomax is not None and costMode == 'DEFO':
            f.write(f'DEFOMAX_CYCLE  {defomax}\n')
        
        # Other parameters (commented out)
        f.write('\n# Cost parameters\n')
        f.write('# MAXCOST  1000.0\n')
        f.write('# COSTSCALE  100.0\n')
        
        f.write('\n# Tile specific parameters\n')
        f.write('# TILECOSTTHRESH  500\n')
        f.write('# TILEEDGEWEIGHT  2.5\n')
        f.write('# SCNDRYARCFLOWMAX  8\n')
        f.write('# NSHORTCYCLE  200\n')
        
        f.write('\n# Statistical cost parameters\n')
        f.write('# INITDZSTEP  100.0\n')
        f.write('# MAXFLOW  4.0\n')
        
        f.write('\n# Processing parameters\n')
        f.write('# MAXNEWNODECONST  0.0008\n')
        f.write('# MAXCYCLEFRACTION  0.00001\n')

    # Build snaphu command
    cmd = f"snaphu {wrapName} {width}"
    
    # Add cost mode
    if costMode == 'DEFO':
        cmd += " -d"
    elif costMode == 'SMOOTH':
        cmd += " -s"
    elif costMode == 'TOPO':
        cmd += " -t"
    
    # Add initialization method
    if initMethod == 'MCF':
        cmd += " --mcf"
    
    # Add correlation file
    if os.path.exists(corName):
        cmd += f" -c {corName}"
    
    # Add configuration file
    cmd += f" -f {configName}"
    
    # Add output file
    cmd += f" -o {unwrapName}"
    
    print(f"Executing command: {cmd}")
    status = os.system(cmd)
    
    if status != 0:
        print(f"Snaphu failed with status {status}")
        raise Exception('Snaphu execution failed')

    # Create output image metadata
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setAccessMode('read')
    outImage.renderHdr()
    outImage.renderVRT()

    # Check if connected components was created
    if os.path.exists(unwrapName + '.conncomp'):
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName + '.conncomp')
        self.insar.connectedComponentsFilename = unwrapName + '.conncomp'
        connImage.setWidth(width)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        connImage.renderHdr()
        connImage.renderVRT()

    return
