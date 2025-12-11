#!/usr/bin/env python3

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import glob
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const
import os

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Unpack LT1 SLC data and store metadata in pickle file.')
    parser.add_argument('-i','--input', dest='h5dir', type=str,
            required=True, help='Input LT1 directory')
    parser.add_argument('-o', '--output', dest='slcdir', type=str,
            required=True, help='Output SLC directory')
    parser.add_argument('-p', '--polarization', dest='polarization', type=str,
            default='HH', help='polarization in case if quad or full pol data exists. Deafult: HH')
    parser.add_argument('--orbit', dest='orbitDir', type=str, default=None, required=False,
                        help='Optional: directory with the precise orbit file for LT1 SLC. Default: None')
    return parser.parse_args()


def unpack(hdf5, slcname, polarization='HH', orbitDir=None):
    '''
    Unpack HDF5 to binary SLC file.
    '''
    # Find all TIFF files and corresponding metadata files
    tiffnames = sorted(glob.glob(os.path.join(hdf5, '*.tiff')))
    xmlnames = sorted(glob.glob(os.path.join(hdf5, '*.meta.xml')))
    
    if not tiffnames:
        raise Exception(f"No TIFF files found in {hdf5}")
    
    if not xmlnames:
        raise Exception(f"No XML metadata files found in {hdf5}")
    
    # Ensure the output directory exists
    if not os.path.isdir(slcname):
        os.mkdir(slcname)
    
    date = os.path.basename(slcname)
    
    # Create Lutan1 object
    obj = createSensor('LUTAN1')
    obj.configure()
    
    # Set TIFF and XML file lists
    obj._tiffList = tiffnames
    obj._xmlFileList = xmlnames
    
    # If the orbit file is provided, find and set the orbit file list
    if orbitDir:
        try:
            orbnames = sorted(glob.glob(os.path.join(hdf5, '*ABSORBIT_SCIE.xml'), recursive=True))
            if not orbnames:
                orbnames = sorted(glob.glob(os.path.join(hdf5, '*.txt'), recursive=True))
        except IndexError:
            orbnames = []
        
        if orbnames:
            obj._orbitFileList = orbnames
            print("Found orbit files:", orbnames)
    
    # Set the output file path
    obj.output = os.path.join(slcname, date + '.slc')
    
    print("Processing files:")
    print("TIFF files:", obj._tiffList)
    print("XML files:", obj._xmlFileList)
    print("Output file:", obj.output)
    
    # Extract image data
    obj.extractImage()
    
    # Get the processed frame
    frame = obj.frame
    
    # Generate the image header file
    frame.getImage().renderHdr()
    
    # Get the Doppler coefficients
    coeffs = obj.doppler_coeff if hasattr(obj, 'doppler_coeff') else [0.0, 0.0, 0.0]
    
    # Set the Doppler and FM rate polynomials
    poly = Poly1D.Poly1D()
    poly.initPoly(order=1)
    poly.setCoeffs([0.0, 0.0])
    
    fpoly = Poly1D.Poly1D()
    fpoly.initPoly(order=1)
    fpoly.setCoeffs([0.0, 0.0])
    
    # Set the Doppler parameters
    frame._dopplerVsPixel = coeffs if coeffs else [0., 0., 0.]
    
    # Save the processing results
    pickName = os.path.join(slcname, 'data')
    with shelve.open(pickName) as db:
        db['frame'] = frame
        db['doppler'] = poly
        db['fmrate'] = fpoly
    
    print("Doppler coefficients:", poly._coeffs)
    print("FM rate coefficients:", fpoly._coeffs)
    
    return obj

if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    if inps.slcdir.endswith('/'):
        inps.slcdir = inps.slcdir[:-1]

    if inps.h5dir.endswith('/'):
        inps.h5dir = inps.h5dir[:-1]

    obj = unpack(inps.h5dir, inps.slcdir,
                 polarization=inps.polarization,
                 orbitDir=inps.orbitDir)

