#!/usr/bin/env python3

import numpy as np 
import argparse
import os
import isce
import isceobj
import shelve
import datetime
from isceobj.Location.Offset import OffsetField
from mroipac.ampcor.Ampcor import Ampcor
import pickle
from math import comb as _math_comb


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
    parser.add_argument('--cr', dest='consistency_radius', type=float, default=None,
            help='Spatial consistency radius in pixels (default: auto = 3x median grid spacing)')
    parser.add_argument('--ct', dest='consistency_thresh', type=float, default=0.3,
            help='Spatial consistency offset deviation threshold in pixels')
    parser.add_argument('--mn', dest='min_neighbors', type=int, default=3,
            help='Minimum number of neighbours required for spatial consistency check')

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def estimateOffsetField(reference, secondary, azoffset=0, rgoffset=0):
    '''
    Estimate offset field between burst and simamp.
    '''


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
    objOffset.setSearchWindowSizeWidth(60)
    objOffset.setSearchWindowSizeHeight(60)
    margin = 2*objOffset.searchWindowSizeWidth + objOffset.windowSizeWidth

    objOffset.thresholdSNR = 0.01

    nAcross = 60
    nDown = 60

   
    offAc = max(101,-rgoffset)+margin
    offDn = max(101,-azoffset)+margin

    
    lastAc = int( min(width, sim.getWidth() - offAc) - margin)
    lastDn = int( min(length, sim.getLength() - offDn) - margin)

#    print('Across: ', offAc, lastAc, width, sim.getWidth(), margin)
#    print('Down: ', offDn, lastDn, length, sim.getLength(), margin)

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

    sar.finalizeImage()
    sim.finalizeImage()

    result = objOffset.getOffsetField()
    return result


def fitOffsets(field,azrgOrder=0,azazOrder=0,
        rgrgOrder=0,rgazOrder=0,snr=5.0,
        consistencyRadius=None, consistencyThresh=0.3, minNeighbors=3):
    '''
    Estimate constant range and azimuth shifts.
    '''

    print('%d input offset points before spatial consistency culling' %
          (len(field._offsets)))

    if consistencyRadius is None:
        inArr = np.array(field.unpackOffsets(), dtype=np.float64)
        coords = inArr[:, [0, 2]]
        spacings = []
        for col in range(2):
            vals = np.unique(coords[:, col])
            diffs = np.diff(np.sort(vals))
            diffs = diffs[diffs > 0.0]
            if len(diffs) > 0:
                spacings.append(np.median(diffs))

        if len(spacings) == 0:
            consistencyRadius = np.inf
        else:
            consistencyRadius = 3.0 * max(spacings)

    print('Applying spatial consistency culling: radius %.4f, threshold %.4f, min neighbours %d' %
          (consistencyRadius, consistencyThresh, minNeighbors))

    for passNumber in range(2):
        inArr = np.array(field.unpackOffsets(), dtype=np.float64)
        coords = inArr[:, [0, 2]]
        offsets = inArr[:, [1, 3]]

        keep = []
        removedSpatial = 0
        insufficientNeighbors = 0
        for ind in range(len(field._offsets)):
            distances = np.hypot(coords[:, 0] - coords[ind, 0],
                                 coords[:, 1] - coords[ind, 1])
            neighbors = np.where((distances > 0.0) & (distances <= consistencyRadius))[0]

            if len(neighbors) < minNeighbors:
                insufficientNeighbors += 1
                keep.append(ind)
                continue

            localOffset = np.median(offsets[neighbors], axis=0)
            residual = np.hypot(offsets[ind, 0] - localOffset[0],
                                offsets[ind, 1] - localOffset[1])

            if residual <= consistencyThresh:
                keep.append(ind)
            else:
                removedSpatial += 1

        field._offsets = [field._offsets[ind] for ind in keep]
        print('%d points left after spatial consistency culling pass %d (removed %d points, %d points had too few neighbours)' %
              (len(field._offsets), passNumber + 1, removedSpatial, insufficientNeighbors))

        if len(field._offsets) == 0:
            raise ValueError('No offsets left after spatial consistency culling pass %d' %
                             (passNumber + 1))

    if len(field._offsets) == 0:
        raise ValueError('No offsets left after spatial consistency culling')

    aa, dummy = field.getFitPolynomials(azimuthOrder=azazOrder, rangeOrder=azrgOrder, usenumpy=True)
    dummy, rr = field.getFitPolynomials(azimuthOrder=rgazOrder, rangeOrder=rgrgOrder, usenumpy=True)

    azshift = aa._coeffs[0][0]
    rgshift = rr._coeffs[0][0]
    print('Estimated az shift: ', azshift)
    print('Estimated rg shift: ', rgshift)

    return (aa, rr), field


def denormalizePoly(poly, field):
    '''
    Poly2D.polyfit() fits coefficients in normalized coordinate space but the
    normalization parameters (normAzimuth, normRange, meanAzimuth, meanRange)
    are not preserved by Python shelve/pickle due to ISCE2 Component framework
    initialization resetting them to defaults (1.0 / 0.0).

    This function bakes the normalization into the coefficients so the returned
    polynomial evaluates correctly with the default norm=1, mean=0.

    The normalization used by polyfit is re-derived from the same offset field
    that was passed to getFitPolynomials().
    '''
    from isceobj.Util.Poly2D import Poly2D

    inArr = np.array(field.unpackOffsets(), dtype=np.float64)
    if len(inArr) == 0:
        return poly

    # getFitPolynomials subtracts azmin from azimuth positions before calling
    # polyfit, so polyfit sees azimuth positions shifted to start at 0.
    # polyfit then computes: ymin=0, ynorm=max-min, meanAzimuth=0+azmin=azmin
    azmin  = float(np.min(inArr[:, 2]))
    aznorm = float(np.max(inArr[:, 2]) - azmin)

    # Range positions are NOT shifted before polyfit
    rgmin  = float(np.min(inArr[:, 0]))
    rgnorm = float(np.max(inArr[:, 0]) - rgmin)

    if aznorm == 0.0:
        aznorm = 1.0
    if rgnorm == 0.0:
        rgnorm = 1.0

    azOrd = poly._azimuthOrder
    rgOrd = poly._rangeOrder
    c = poly._coeffs

    # Transform coefficients from normalized space to raw pixel space.
    # Original: p(az, rng) = sum_ij c[i][j] * ((az-azmin)/aznorm)^i * ((rng-rgmin)/rgnorm)^j
    # Target:   p(az, rng) = sum_pq d[p][q] * az^p * rng^q
    #
    # By binomial expansion:
    # d[p][q] = sum_{i>=p, j>=q} c[i][j]
    #           * C(i,p) * (-azmin)^(i-p) / aznorm^i
    #           * C(j,q) * (-rgmin)^(j-q) / rgnorm^j
    new_c = [[0.0] * (rgOrd + 1) for _ in range(azOrd + 1)]

    for i in range(azOrd + 1):
        for j in range(rgOrd + 1):
            coef = c[i][j]
            if coef == 0.0:
                continue
            for p in range(i + 1):
                az_factor = _math_comb(i, p) * ((-azmin) ** (i - p)) / (aznorm ** i)
                for q in range(j + 1):
                    rg_factor = _math_comb(j, q) * ((-rgmin) ** (j - q)) / (rgnorm ** j)
                    new_c[p][q] += coef * az_factor * rg_factor

    new_poly = Poly2D()
    new_poly.initPoly(rangeOrder=rgOrd, azimuthOrder=azOrd, coeffs=new_c)
    # normAzimuth=1, normRange=1, meanAzimuth=0, meanRange=0 are correct defaults now
    return new_poly


def main(iargs=None):
    '''
    Generate offset fields burst by burst.
    '''

    inps = cmdLineParse(iargs)

    field = estimateOffsetField(inps.reference, inps.secondary,
            azoffset=inps.azoff, rgoffset=inps.rgoff)

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
            consistencyRadius=inps.consistency_radius,
            consistencyThresh=inps.consistency_thresh,
            minNeighbors=inps.min_neighbors)
    odb['cull_field'] = cull

    ####Scale by ratio
    for row in shifts[0]._coeffs:
        for ind, val in  enumerate(row):
            row[ind] = val * azratio

    for row in shifts[1]._coeffs:
        for ind, val in enumerate(row):
            row[ind] = val * rgratio
    

    odb['azpoly'] = denormalizePoly(shifts[0], cull)
    odb['rgpoly'] = denormalizePoly(shifts[1], cull)
    odb.close()

if __name__ == '__main__':
    main()



