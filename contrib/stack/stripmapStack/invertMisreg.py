#!/usr/bin/env python3

# Author: Heresh Fattahi

import os, sys, glob
import argparse
import configparser
import datetime
import time
import numpy as np
import shelve
import isce
import isceobj
from isceobj.Util.Poly2D import Poly2D

#################################################################
def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser( description='extracts the overlap geometry between reference bursts')
    parser.add_argument('-i', '--input', type=str, dest='input', required=True,
            help='Directory with the overlap directories that has calculated misregistration for each pair')
    parser.add_argument('-o', '--output', type=str, dest='output', required=True,
            help='output directory to save misregistration for each date with respect to the stack Reference date')
    parser.add_argument('-r', '--reference', type=str, dest='reference', default=None,
            help='Reference date (YYYYMMDD format). If not provided, the first date in sorted dateList will be used.')
    parser.add_argument('-f', '--misregFileName', type=str, dest='misregFileName', default='misreg.txt',
            help='misreg file name that contains the calculated misregistration for a pair')

    return parser

def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    inps = parser.parse_args(args=iargs)

    return inps


def date_list(pairDirs):
  dateList = []
  tbase = []
  for di in  pairDirs:
    #di = di.replace('.txt','')
    dates = os.path.basename(di).split('_')
    dates1 = os.path.basename(di).split('_')
    if not dates[0] in dateList: dateList.append(dates[0])
    if not dates[1] in dateList: dateList.append(dates[1])
   
  dateList.sort()
  d1 = datetime.datetime(*time.strptime(dateList[0],"%Y%m%d")[0:5])
  for ni in range(len(dateList)):
    d2 = datetime.datetime(*time.strptime(dateList[ni],"%Y%m%d")[0:5])
    diff = d2-d1
    tbase.append(diff.days)
  dateDict = {}
  for i in range(len(dateList)): dateDict[dateList[i]] = tbase[i]
  return tbase,dateList,dateDict

#####################################
def extract_offset(filename, verbose=False):
  if verbose:
    print(f'Reading offset from: {filename}')
  with shelve.open(os.path.join(filename,'misreg'),flag='r') as db:
       azpoly = db['azpoly']
       rgpoly = db['rgpoly']

  azCoefs = np.array(azpoly.getCoeffs())
  rgCoefs = np.array(rgpoly.getCoeffs())
  
  # Get constant offset (first coefficient) for display
  az_const = azCoefs.flatten()[0] if azCoefs.size > 0 else 0.0
  rg_const = rgCoefs.flatten()[0] if rgCoefs.size > 0 else 0.0
  
  if verbose:
    print(f'  Azimuth offset (constant): {az_const:.6f} pixels')
    print(f'  Range offset (constant): {rg_const:.6f} pixels')

  return azCoefs.flatten(), rgCoefs.flatten()

def getPolyInfo(filename):
  with shelve.open(os.path.join(filename,'misreg'),flag='r') as db:
       azpoly = db['azpoly']
       rgpoly = db['rgpoly']  
  azCoefs = azpoly.getCoeffs()
  rgCoefs = rgpoly.getCoeffs()
  info = {}
  info['sizeOfAzCoefs'] = np.size(azCoefs)
  info['sizeOfRgCoefs'] = np.size(rgCoefs)
  info['shapeOfAzCoefs'] = np.shape(azCoefs)
  info['shapeOfRgCoefs'] = np.shape(rgCoefs)
  info['azazOrder'] = azpoly.getAzimuthOrder()
  info['azrgOrder'] = azpoly.getRangeOrder()
  info['rgazOrder'] = rgpoly.getAzimuthOrder()
  info['rgrgOrder'] = rgpoly.getRangeOrder()

  return info
  #return np.size(azCoefs), np.size(rgCoefs), np.shape(azCoefs), np.shape(rgCoefs)

######################################
def design_matrix(pairDirs, referenceDate=None):
  '''Make the design matrix for the inversion.  '''
  tbase,dateList,dateDict = date_list(pairDirs)
  numDates = len(dateDict)
  numIfgrams = len(pairDirs)
  A = np.zeros((numIfgrams,numDates))
  B = np.zeros(np.shape(A))

  # numAzCoefs, numRgCoefs, azCoefsShape, rgCoefsShape = getPolyInfo(pairDirs[0])
  polyInfo = getPolyInfo(pairDirs[0])
  Laz = np.zeros((numIfgrams, polyInfo['sizeOfAzCoefs']))
  Lrg = np.zeros((numIfgrams, polyInfo['sizeOfRgCoefs']))
  daysList = []
  for day in tbase:
    daysList.append(day)
  tbase = np.array(tbase)
  t = np.zeros((numIfgrams,2))
  for ni in range(len(pairDirs)):
    date12 = os.path.basename(pairDirs[ni]).replace('.txt','')
    date = date12.split('_')
    ndxt1 = daysList.index(dateDict[date[0]])
    ndxt2 = daysList.index(dateDict[date[1]])
    A[ni,ndxt1] = -1
    A[ni,ndxt2] = 1
    B[ni,ndxt1:ndxt2] = tbase[ndxt1+1:ndxt2+1]-tbase[ndxt1:ndxt2]
    t[ni,:] = [dateDict[date[0]],dateDict[date[1]]]

  #  misreg_dict = extract_offset(os.path.join(overlapDirs[ni],misregName))
    azOff, rgOff = extract_offset(pairDirs[ni], verbose=False)
    Laz[ni,:] = azOff[:]
    Lrg[ni,:] = rgOff[:]

  # Find reference date index
  if referenceDate is None:
    # Default: use first date in sorted list (original behavior)
    refIdx = 0
  else:
    if referenceDate not in dateList:
      print(f'Warning: Reference date {referenceDate} not found in dateList. Using first date as reference.')
      refIdx = 0
    else:
      refIdx = dateList.index(referenceDate)
  
  # Remove reference date column
  A = np.delete(A, refIdx, axis=1)
  B = np.delete(B, refIdx, axis=1)
  
 # ind=~np.isnan(Laz)
 # return A[ind[:,0],:],B[ind[:,0],:],Laz[ind,:], Lrg[ind]
  return A, B, Laz, Lrg, refIdx
 
######################################
def main(iargs=None):

  inps = cmdLineParse(iargs)
  os.makedirs(inps.output, exist_ok=True)

  pairDirs = glob.glob(os.path.join(inps.input,'*'))
  polyInfo = getPolyInfo(pairDirs[0])

  tbase, dateList, dateDict = date_list(pairDirs)

  # Determine reference date
  referenceDate = inps.reference
  if referenceDate is None:
    # Default: use first date in sorted list (original behavior)
    referenceDate = dateList[0]
    print(f'No reference date specified. Using first date in sorted list: {referenceDate}')

  A, B, Laz, Lrg, refIdx = design_matrix(pairDirs, referenceDate=referenceDate)
  
  # Print observed offsets for each pair
  print('='*80)
  print('Observed offsets for each pair (before inversion):')
  print('='*80)
  print(f'{"Pair":<25} {"Azimuth offset (pixels)":<30} {"Range offset (pixels)":<30}')
  print('-'*80)
  for ni in range(len(pairDirs)):
    date12 = os.path.basename(pairDirs[ni]).replace('.txt','')
    # Get constant offset (first coefficient) for display
    az_const = Laz[ni, 0] if Laz.shape[1] > 0 else 0.0
    rg_const = Lrg[ni, 0] if Lrg.shape[1] > 0 else 0.0
    print(f'{date12:<25} {az_const:>28.6f} {rg_const:>28.6f}')
  print('='*80)
  print('')
  
  A1 = np.linalg.pinv(A)
  A1 = np.array(A1,np.float32)

  # Solve for offsets relative to reference date
  Saz = np.dot(A1, Laz)
  Srg = np.dot(A1, Lrg)

  residual_az = Laz-np.dot(A,Saz)
  residual_rg = Lrg-np.dot(A,Srg)
  RMSE_az = np.sqrt(np.sum(residual_az**2)/len(residual_az))
  RMSE_rg = np.sqrt(np.sum(residual_rg**2)/len(residual_rg))

  # Insert zero row at reference date position
  zero_row_az = np.zeros((1, Saz.shape[1]), dtype=np.float32)
  zero_row_rg = np.zeros((1, Srg.shape[1]), dtype=np.float32)
  Saz = np.vstack([Saz[:refIdx], zero_row_az, Saz[refIdx:]])
  Srg = np.vstack([Srg[:refIdx], zero_row_rg, Srg[refIdx:]])

  print('')
  print('Rank of design matrix: ' + str(np.linalg.matrix_rank(A)))
  if np.linalg.matrix_rank(A)==len(dateList)-1:
     print('Design matrix is full rank.')
  else:
     print('Design matrix is rank deficient. Network is disconnected.')
     print('Using a fully connected network is recommended.')
  print('RMSE in azimuth : '+str(RMSE_az)+' pixels')
  print('RMSE in range : '+str(RMSE_rg)+' pixels')
  print('')
  
  # Calculate and display residuals for each pair
  print('='*80)
  print('Residual analysis (observed - predicted) for each pair:')
  print('='*80)
  print(f'{"Pair":<25} {"Az residual":<20} {"Rg residual":<20} {"Total residual":<20}')
  print('-'*80)
  
  # Reconstruct full Saz and Srg for residual calculation
  Saz_full = Saz.copy()
  Srg_full = Srg.copy()
  
  # Calculate predicted offsets for each pair
  pair_residuals = []
  for ni in range(len(pairDirs)):
    date12 = os.path.basename(pairDirs[ni]).replace('.txt','')
    date = date12.split('_')
    idx1 = dateList.index(date[0])
    idx2 = dateList.index(date[1])
    
    # Predicted offset = S[idx2] - S[idx1]
    pred_az = Saz_full[idx2, 0] - Saz_full[idx1, 0]
    pred_rg = Srg_full[idx2, 0] - Srg_full[idx1, 0]
    
    # Observed offset
    obs_az = Laz[ni, 0]
    obs_rg = Lrg[ni, 0]
    
    # Residual
    res_az = obs_az - pred_az
    res_rg = obs_rg - pred_rg
    res_total = np.sqrt(res_az**2 + res_rg**2)
    
    pair_residuals.append({
      'pair': date12,
      'az_res': res_az,
      'rg_res': res_rg,
      'total_res': res_total,
      'obs_az': obs_az,
      'obs_rg': obs_rg,
      'pred_az': pred_az,
      'pred_rg': pred_rg
    })
    
    print(f'{date12:<25} {res_az:>18.6f} {res_rg:>18.6f} {res_total:>18.6f}')
  
  print('='*80)
  print('')
  
  # Identify problematic pairs (high residuals)
  threshold_az = 2.0 * RMSE_az  # 2-sigma threshold
  threshold_rg = 2.0 * RMSE_rg
  threshold_total = np.sqrt(threshold_az**2 + threshold_rg**2)
  
  problematic_pairs = []
  for pr in pair_residuals:
    if (abs(pr['az_res']) > threshold_az or 
        abs(pr['rg_res']) > threshold_rg or 
        pr['total_res'] > threshold_total):
      problematic_pairs.append(pr)
  
  if problematic_pairs:
    print('='*80)
    print('Potentially problematic pairs (residuals > 2*RMSE):')
    print('='*80)
    print(f'{"Pair":<25} {"Az residual":<20} {"Rg residual":<20} {"Total residual":<20}')
    print('-'*80)
    for pr in sorted(problematic_pairs, key=lambda x: x['total_res'], reverse=True):
      print(f'{pr["pair"]:<25} {pr["az_res"]:>18.6f} {pr["rg_res"]:>18.6f} {pr["total_res"]:>18.6f}')
    print('='*80)
    print('')
  else:
    print('No pairs with residuals significantly above threshold (2*RMSE).')
    print('')
  print('='*80)
  print('Estimated offsets with respect to the stack reference date:')
  print('='*80)
  print(f'{"Date":<15} {"Azimuth offset (pixels)":<30} {"Range offset (pixels)":<30}')
  print('-'*80)
  offset_dict={}
  for i in range(len(dateList)):
     # Get constant offset (first coefficient) for display
     az_const = Saz[i, 0] if Saz.shape[1] > 0 else 0.0
     rg_const = Srg[i, 0] if Srg.shape[1] > 0 else 0.0
     print(f'{dateList[i]:<15} {az_const:>28.6f} {rg_const:>28.6f}')
     offset_dict[dateList[i]]=Saz[i]
     
     # Create Poly2D objects and save to shelve
     azpoly = Poly2D()
     rgpoly = Poly2D()
     azCoefs = np.reshape(Saz[i,:],polyInfo['shapeOfAzCoefs']).tolist()
     rgCoefs = np.reshape(Srg[i,:],polyInfo['shapeOfRgCoefs']).tolist()
     azpoly.initPoly(rangeOrder=polyInfo['azrgOrder'], azimuthOrder=polyInfo['azazOrder'], coeffs=azCoefs)
     rgpoly.initPoly(rangeOrder=polyInfo['rgrgOrder'], azimuthOrder=polyInfo['rgazOrder'], coeffs=rgCoefs)

     os.makedirs(os.path.join(inps.output,dateList[i]), exist_ok=True)

     odb = shelve.open(os.path.join(inps.output,dateList[i]+'/misreg'))
     odb['azpoly'] = azpoly
     odb['rgpoly'] = rgpoly
     odb.close()
  
  print('='*80)
  print('')
  print('Note: Inverted offsets are cumulative and can be larger than individual pair offsets.')
  print('This is because offsets accumulate along the network path from reference date.')
  print('')
 
     #with open(os.path.join(inps.output,dateList[i]+'.txt'), 'w') as f:
     #   f.write(str(Saz[i]))

  print('')  
 
if __name__ == '__main__' :
  ''' 
  invert a network of the pair's mis-registrations to
  estimate the mis-registrations wrt the Reference date.
  '''

  main()