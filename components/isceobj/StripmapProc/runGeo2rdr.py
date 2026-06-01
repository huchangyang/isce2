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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






import isceobj
import stdproc
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from isceobj.Util.Polynomial import Polynomial
from isceobj.Util.Poly2D import Poly2D
from isceobj.Constants import SPEED_OF_LIGHT
import logging
import numpy as np
import datetime
import os

logger = logging.getLogger('isce.insar.runGeo2rdr') 


def _is_identity_affine(affineTransform, tol=1.0e-10):
    identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    if affineTransform is None or len(affineTransform) != 6:
        return True
    return all(abs(float(val) - ref) <= tol for val, ref in zip(affineTransform, identity))


def _render_offset_xml(filename, width, length):
    image = isceobj.createImage()
    image.setFilename(filename)
    image.setWidth(width)
    image.setLength(length)
    image.setAccessMode('READ')
    image.bands = 1
    image.dataType = 'DOUBLE'
    image.scheme = 'BIP'
    image.renderHdr()


def rectifyRangeOffsetWithAffine(rangeOffsetFile, affineTransform, chunkLength=256):
    '''
    Resample the geo2rdr range-offset field into the reference radar grid.

    The radar-DEM affine transform maps reference radar coordinates to the
    DEM/simulated radar coordinates where geo2rdr produced the original range
    offsets.  Rectification therefore samples the original range-offset image at
    affine(reference_pixel) and writes the result back to rangeOffsetFile.
    '''
    if _is_identity_affine(affineTransform):
        logger.info('Radar-DEM affine transform is identity; range offset rectification skipped')
        return

    if not os.path.exists(rangeOffsetFile):
        logger.warning('Range offset file not found, cannot rectify with radar-DEM affine: {}'.format(
            rangeOffsetFile))
        return

    image = isceobj.createImage()
    image.load(rangeOffsetFile + '.xml')
    width = image.getWidth()
    length = image.getLength()

    rawRangeOffsetFile = rangeOffsetFile + '.before_rdr_dem_affine'
    tmpRangeOffsetFile = rangeOffsetFile + '.rdr_dem_affine.tmp'

    for filename in [rawRangeOffsetFile, tmpRangeOffsetFile]:
        for ext in ['', '.xml', '.vrt']:
            if os.path.exists(filename + ext):
                os.remove(filename + ext)

    os.replace(rangeOffsetFile, rawRangeOffsetFile)
    for ext in ['.xml', '.vrt']:
        if os.path.exists(rangeOffsetFile + ext):
            os.remove(rangeOffsetFile + ext)

    _render_offset_xml(rawRangeOffsetFile, width, length)

    src = np.memmap(rawRangeOffsetFile, dtype=np.float64, mode='r', shape=(length, width))
    dst = np.memmap(tmpRangeOffsetFile, dtype=np.float64, mode='w+', shape=(length, width))

    m11, m12, m21, m22, t1, t2 = [float(val) for val in affineTransform]
    xCoord = np.arange(width, dtype=np.float64) + 1.0

    logger.info('Rectifying range offset with radar-DEM affine transform: {}'.format(
        affineTransform))

    for rowStart in range(0, length, chunkLength):
        rowStop = min(rowStart + chunkLength, length)
        yCoord = np.arange(rowStart, rowStop, dtype=np.float64)[:, None] + 1.0

        srcX = m11 * xCoord[None, :] + m12 * yCoord + t1 - 1.0
        srcY = m21 * xCoord[None, :] + m22 * yCoord + t2 - 1.0

        valid = ((srcX >= 0.0) & (srcX <= (width - 1)) &
                 (srcY >= 0.0) & (srcY <= (length - 1)))

        x0 = np.floor(srcX).astype(np.int64)
        y0 = np.floor(srcY).astype(np.int64)
        x0 = np.clip(x0, 0, width - 1)
        y0 = np.clip(y0, 0, length - 1)
        x1 = np.clip(x0 + 1, 0, width - 1)
        y1 = np.clip(y0 + 1, 0, length - 1)

        wx = srcX - x0
        wy = srcY - y0

        values = ((1.0 - wx) * (1.0 - wy) * src[y0, x0] +
                  wx * (1.0 - wy) * src[y0, x1] +
                  (1.0 - wx) * wy * src[y1, x0] +
                  wx * wy * src[y1, x1])
        values[~valid] = 0.0
        dst[rowStart:rowStop, :] = values

    dst.flush()
    del dst
    del src

    os.replace(tmpRangeOffsetFile, rangeOffsetFile)
    _render_offset_xml(rangeOffsetFile, width, length)
    logger.info('Rectified range offset written to {}; original saved as {}'.format(
        rangeOffsetFile, rawRangeOffsetFile))

def runGeo2rdr(self):
    from zerodop.geo2rdr import createGeo2rdr
    from isceobj.Planet.Planet import Planet

    logger.info("Running geo2rdr")

    info = self._insar.loadProduct( self._insar.secondarySlcCropProduct)

    offsetsDir = self.insar.offsetsDirname 
    os.makedirs(offsetsDir, exist_ok=True)

    grdr = createGeo2rdr()
    grdr.configure()

    planet = info.getInstrument().getPlatform().getPlanet()
    grdr.slantRangePixelSpacing = info.getInstrument().getRangePixelSize()
    grdr.prf = info.PRF #info.getInstrument().getPulseRepetitionFrequency()
    grdr.radarWavelength = info.getInstrument().getRadarWavelength()
    grdr.orbit = info.getOrbit()
    grdr.width = info.getImage().getWidth()
    grdr.length = info.getImage().getLength()

    grdr.wireInputPort(name='planet', object=planet)
    grdr.lookSide =  info.instrument.platform.pointingDirection

    grdr.orbitInterpolationMethod = 'LEGENDRE'

    grdr.setSensingStart(info.getSensingStart())
    grdr.rangeFirstSample = info.startingRange
    grdr.numberRangeLooks = 1
    grdr.numberAzimuthLooks = 1


    if self.insar.secondaryGeometrySystem.lower().startswith('native'):
        p = [x/info.PRF for x in info._dopplerVsPixel]
    else:
        p = [0.]

    grdr.dopplerCentroidCoeffs = p
    grdr.fmrateCoeffs = [0.]

    ###Input and output files
    grdr.rangeOffsetImageName = os.path.join(offsetsDir, self.insar.rangeOffsetFilename)
    grdr.azimuthOffsetImageName = os.path.join(offsetsDir, self.insar.azimuthOffsetFilename)

    latFilename = os.path.join(self.insar.geometryDirname, self.insar.latFilename + '.full')
    lonFilename = os.path.join(self.insar.geometryDirname, self.insar.lonFilename + '.full')
    heightFilename = os.path.join(self.insar.geometryDirname, self.insar.heightFilename + '.full')

    demImg = isceobj.createImage()
    demImg.load(heightFilename + '.xml')
    demImg.setAccessMode('READ')
    grdr.demImage = demImg

    latImg = isceobj.createImage()
    latImg.load(latFilename + '.xml')
    latImg.setAccessMode('READ')
    grdr.latImage = latImg

    lonImg = isceobj.createImage()
    lonImg.load(lonFilename + '.xml')
    lonImg.setAccessMode('READ')

    grdr.lonImage = lonImg
    grdr.outputPrecision = 'DOUBLE'
        
    grdr.geo2rdr()

    rectifyRangeOffsetWithAffine(
        grdr.rangeOffsetImageName,
        getattr(self.insar, 'radarDemAffineTransform', [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))

    return
