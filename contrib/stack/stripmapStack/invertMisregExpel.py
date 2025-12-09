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
def extract_offset(filename):
  print(filename)
  with shelve.open(os.path.join(filename,'misreg'),flag='r') as db:
       print(dir(db))
       azpoly = db['azpoly']
       rgpoly = db['rgpoly']

  azCoefs = np.array(azpoly.getCoeffs())
  rgCoefs = np.array(rgpoly.getCoeffs())

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
def design_matrix(pairDirs):
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
    azOff, rgOff = extract_offset(pairDirs[ni])
    Laz[ni,:] = azOff[:]
    Lrg[ni,:] = rgOff[:]

    def detect_outliers(offsets, threshold=1):
        """使用3-sigma法则检测异常值"""
        # 对每个系数分别计算
        valid = np.ones(offsets.shape[0], dtype=bool)  # 初始化所有点都是有效的
        for i in range(offsets.shape[1]):
            mean = np.mean(offsets[:, i])
            std = np.std(offsets[:, i])
            if std != 0:  # 避免除以零
                z_scores = np.abs((offsets[:, i] - mean) / std)
                valid &= (z_scores < threshold)
        return valid
  valid_az = detect_outliers(Laz)
  valid_rg = detect_outliers(Lrg)

  valid_pairs = valid_az & valid_rg
  for i in range(len(valid_pairs)):
      if not valid_pairs[i]:
          print(f"剔除配准对: {os.path.basename(pairDirs[i])}")
          print(f"方位向偏移: {Laz[i,:]}")
          print(f"距离向偏移: {Lrg[i,:]}")
  A = A[valid_pairs]
  B = B[valid_pairs]
  Laz = Laz[valid_pairs]
  Lrg = Lrg[valid_pairs]
  A = A[:,1:]
  B = B[:,:-1]
  
 # ind=~np.isnan(Laz)
 # return A[ind[:,0],:],B[ind[:,0],:],Laz[ind,:], Lrg[ind]
  return A, B, Laz, Lrg

def iterative_estimation(A, Laz, Lrg, max_iterations=3, residual_threshold=2.0):
    """迭代方式估计偏移量，逐步剔除异常值"""
    A_current = A.copy()
    Laz_current = Laz.copy()
    Lrg_current = Lrg.copy()
    
    for iteration in range(max_iterations):
        print(f"\n开始第{iteration+1}次迭代...")
        
        # 计算当前估计值
        A1 = np.linalg.pinv(A_current)
        Saz = np.dot(A1, Laz_current)
        Srg = np.dot(A1, Lrg_current)
        
        # 计算残差
        residual_az = np.abs(Laz_current - np.dot(A_current, Saz))
        residual_rg = np.abs(Lrg_current - np.dot(A_current, Srg))
        
        # 计算每个观测的总残差
        total_residual = np.sqrt(np.sum(residual_az**2, axis=1) + 
                               np.sum(residual_rg**2, axis=1))
        
        # 找出残差小于阈值的观测
        valid_obs = total_residual < residual_threshold
        
        # 打印剔除的观测信息
        n_removed = np.sum(~valid_obs)
        if n_removed > 0:
            print(f"剔除了{n_removed}个异常观测")
            print(f"最大残差: {np.max(total_residual):.3f} 像素")
        
        # 如果没有剔除的点，结束迭代
        if np.all(valid_obs):
            print("没有检测到异常值，迭代结束")
            break
            
        # 更新矩阵和观测值
        A_current = A_current[valid_obs]
        Laz_current = Laz_current[valid_obs]
        Lrg_current = Lrg_current[valid_obs]
        
        # 检查剩余观测是否足够
        if len(A_current) < A_current.shape[1]:
            print("警告：剩余观测数不足，停止迭代")
            break
    
    return Saz, Srg, A_current, Laz_current, Lrg_current

######################################
def main(iargs=None):

  inps = cmdLineParse(iargs)
  os.makedirs(inps.output, exist_ok=True)
    
  pairDirs = glob.glob(os.path.join(inps.input,'*'))
  polyInfo = getPolyInfo(pairDirs[0])

  tbase, dateList, dateDict = date_list(pairDirs)

  A, B, Laz, Lrg = design_matrix(pairDirs)
  # A1 = np.linalg.pinv(A)
  # A1 = np.array(A1,np.float32)

  # zero = np.array([0.],np.float32)
  # Saz = np.dot(A1, Laz)

  # Saz = np.dot(A1, Laz)
  # Srg = np.dot(A1, Lrg)

  # residual_az = Laz-np.dot(A,Saz)
  # residual_rg = Lrg-np.dot(A,Srg)
  # RMSE_az = np.sqrt(np.sum(residual_az**2)/len(residual_az))
  # RMSE_rg = np.sqrt(np.sum(residual_rg**2)/len(residual_rg))

  # 使用迭代方式估计
  Saz, Srg, A_clean, Laz_clean, Lrg_clean = iterative_estimation(A, Laz, Lrg)
  # 计算最终的RMSE
  residual_az = Laz_clean - np.dot(A_clean, Saz)
  residual_rg = Lrg_clean - np.dot(A_clean, Srg)
  RMSE_az = np.sqrt(np.sum(residual_az**2)/len(residual_az))
  RMSE_rg = np.sqrt(np.sum(residual_rg**2)/len(residual_rg))
    
  Saz = np.vstack((np.zeros((1,Saz.shape[1]), dtype=np.float32), Saz))
  Srg = np.vstack((np.zeros((1,Srg.shape[1]), dtype=np.float32), Srg))

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
  print('Estimated offsets with respect to the stack reference date')    
  print('')
  offset_dict={}
  for i in range(len(dateList)):
     print (dateList[i])
     offset_dict[dateList[i]]=Saz[i]
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
 
     with open(os.path.join(inps.output,dateList[i]+'.txt'), 'w') as f:
       f.write(str(Saz[i]))

  print('')  
 
if __name__ == '__main__' :
  ''' 
  invert a network of the pair's mis-registrations to
  estimate the mis-registrations wrt the Reference date.
  '''

  main()
