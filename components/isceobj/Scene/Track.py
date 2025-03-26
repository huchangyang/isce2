#!/usr/bin/env python3
#
#Copyright 2010, by the California Institute of Technology.
#ALL RIGHTS RESERVED.
#United States Government Sponsorship acknowledged.
#Any commercial use must be negotiated with the Office of
#Technology Transfer at the California Institute of Technology.
#
#This software may be subject to U.S. export control laws. By
#accepting this software, the user agrees to comply with all applicable
#U.S. export laws and regulations. User has the responsibility to obtain
#export licenses, or other export authority as may be required before
#exporting such information to foreign countries or providing access
#to foreign persons.
#
import isce
import sys
import os
from sys import float_info
import logging
import datetime
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import Orbit
from isceobj.Attitude.Attitude import Attitude
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from isceobj.Util.decorators import type_check, logged, pickled
import isceobj
import tempfile
import numpy as np
from operator import itemgetter
from isceobj import Constants as CN
from ctypes import cdll, c_char_p, c_int, c_ubyte,byref

@pickled
class Track(object):
    """A class to represent a collection of temporally continuous radar frame
    objects"""

    logging_name = "isce.Scene.Track"

    @logged
    def __init__(self):
        # These are attributes representing the starting time and stopping
        # time of the track
        # As well as the early and late times (range times) of the track
        self._startTime = datetime.datetime(year=datetime.MAXYEAR,month=1,day=1)
        self._stopTime = datetime.datetime(year=datetime.MINYEAR,month=1,day=1)
        # Hopefully this number is large
        # enough, Python doesn't appear to have a MAX_FLT variable
        self._nearRange = float_info.max
        self._farRange = 0.0
        self._frames = []
        self._frame = Frame()
        self._lastFile = ''
        return None

    def combineFrames(self, output, frames):
        attitudeOk = True
        for frame in frames:
            self.addFrame(frame)
            if hasattr(frame,'_attitude'):
                att = frame.getAttitude()
                if not att:
                    attitudeOk = False
        self.createInstrument()
        self.createTrack(output)
        self.createOrbit()
        if attitudeOk:
            self.createAttitude()
        # 添加重叠区域检查
        for i in range(len(frames)-1):
            current_frame = frames[i]
            next_frame = frames[i+1]
            
            # 计算重叠区域
            current_end = current_frame.getSensingStop()
            next_start = next_frame.getSensingStart()
            overlap_time = (next_start - current_end).total_seconds()
            
            self.logger.info(f"Frame {i} and {i+1} overlap:")
            self.logger.info(f"Frame {i} end: {current_end}")
            self.logger.info(f"Frame {i+1} start: {next_start}")
            self.logger.info(f"Overlap time: {overlap_time} seconds")
        return self._frame

    def createAuxFile(self, fileList, output):
        import struct
        from operator import itemgetter
        import os
        import array
        import copy
        dateIndx = []
        cnt = 0
        #first sort the files from earlier to latest. use the first element
        for name in fileList:
            with open(name,'rb') as fp: date = fp.read(16)
            day, musec = struct.unpack('<dd',date)
            dateIndx.append([day,musec,name])
            cnt += 1
        sortedDate = sorted(dateIndx, key=itemgetter(0,1))

        #we need to make sure that there are not duplicate points in the orbit since some frames overlap
        allL = array.array('d')
        allL1 = array.array('d')
        name = sortedDate[0][2]
        size = os.path.getsize(name)//8
        with open(name,'rb') as fp1: allL.fromfile(fp1,size)
        lastDay = allL[-2]
        lastMusec = allL[-1]
        for j in range(1, len(sortedDate)):
            name = sortedDate[j][2]
            size = os.path.getsize(name)//8
            with open(name,'rb') as fp1: allL1.fromfile(fp1, size)
            indxFound = None
            avgPRI = 0
            cnt = 0
            for i in range(len(allL1)//2):
                if i > 0:
                    avgPRI += allL1[2*i+1] - allL1[2*i-1]
                    cnt += 1
                if allL1[2*i] >= lastDay and allL1[2*i+1] > lastMusec:
                    avgPRI //= (cnt-1)
                    if (allL1[2*i+1] - lastMusec) > avgPRI/2:# make sure that the distance in pulse is atleast 1/2 PRI
                        indxFound = 2*i
                    else:#if not take the next
                        indxFound = 2*(i+1)
                        pass
                    break
            if not indxFound is None:
                allL.extend(allL1[indxFound:])
                lastDay = allL[-2]
                lastMusec = allL[-1]
                pass
            pass
        with open(output,'wb') as fp: allL.tofile(fp)
        return

    # Add an additional Frame object to the track
    @type_check(Frame)
    def addFrame(self, frame):
        self.logger.info("Adding Frame to Track")
        self._updateTrackTimes(frame)
        self._frames.append(frame)
        return None

    def createOrbit(self):
        orbitAll = Orbit()
        for i in range(len(self._frames)):
            orbit = self._frames[i].getOrbit()
            #remember that everything is by reference, so the changes applied to orbitAll will be made to the Orbit
            #object in self.frame
            for sv in orbit._stateVectors:
                orbitAll.addStateVector(sv)
            # sort the orbit state vecotrs according to time
            orbitAll._stateVectors.sort(key=lambda sv: sv.time)
        self.removeDuplicateVectors(orbitAll._stateVectors)
        self._frame.setOrbit(orbitAll)

    def removeDuplicateVectors(self,stateVectors):
        i1 = 0
        #remove duplicate state vectors
        while True:
            if i1 >= len(stateVectors) - 1:
                break
            if stateVectors[i1].time == stateVectors[i1+1].time:
                stateVectors.pop(i1+1)
            #since is sorted by time if is not equal we can pass to the next
            else:
                i1 += 1


    def createAttitude(self):
        attitudeAll = Attitude()
        for i in range(len(self._frames)):
            attitude = self._frames[i].getAttitude()
            #remember that everything is by reference, so the changes applied to attitudeAll will be made to the Attitude object in self.frame
            for sv in attitude._stateVectors:
                attitudeAll.addStateVector(sv)
            # sort the attitude state vecotrs according to time
            attitudeAll._stateVectors.sort(key=lambda sv: sv.time)
        self.removeDuplicateVectors(attitudeAll._stateVectors)
        self._frame.setAttitude(attitudeAll)

    def createInstrument(self):
        # the platform is already part of the instrument
        ins = self._frames[0].getInstrument()
        self._frame.setInstrument(ins)

    # sometime the startLine computed below from the sensingStart is not
    #precise and the image are missaligned.
    #for each pair do an exact mach by comparing the lines around lineStart
    #file1,2 input files, startLine1 is the estimated start line in the first file
    #line1 = last line used in the first file
    #width = width of the files
    #frameNum1,2 number of the frames in the sequence of frames to stitch
    #returns a more accurate line1
    def findOverlapLine(self, file1, file2, line1,width,frameNum1,frameNum2):
        import numpy as np
        import array
        fin2 = open(file2,'rb')
        arr2 = array.array('b')
        #read full line at the beginning of second file
        arr2.fromfile(fin2,width)
        buf2 = np.array(arr2,dtype = np.int8)
        numTries = 30
        # start around line1 and try numTries around line1
        # see searchlist to see which lines it searches
        searchNumLines = 2
        #make a sliding window that search for the searchSize samples inside buf2
        searchSize = 500
        max = 0
        indx = None
        fin1 = open(file1,'rb')
        for i in range(numTries):
            # example line1 = 0,searchNumLine = 2 and i = 0 search = [-2,-1,0,1], i = 1, serach =  [-4,-3,2,3]
            search = list(range(line1 - (i+1)*searchNumLines,line1 - i*searchNumLines))
            search.extend(list(range(line1 + i*searchNumLines,line1 + (i+1)*searchNumLines)))
            for k in search:
                arr1 = array.array('b')
                #seek to the line k and read +- searchSize/2 samples from the middle of the line
                fin1.seek(k*width + (width - searchSize)//2,0)
                arr1.fromfile(fin1,searchSize)
                buf1 = np.array(arr1,dtype = np.int8)
                found = False
                for i in np.arange(width-searchSize):
                    lenSame =len(np.nonzero(buf1 == buf2[i:i+searchSize])[0])
                    if  lenSame > max:
                        max = lenSame
                        indx = k
                        if(lenSame == searchSize):
                            found = True
                            break
                if(found):
                    break
            if(found):
                break
        if not found:
            self.logger.warning("Cannot find perfect overlap between frame %d and frame %d. Using acquisition time to find overlap position."%(frameNum1,frameNum2))
        fin1.close()
        fin2.close()
        print('Match found: ', indx)
        return indx

    def reAdjustStartLine(self, sortedList, width):
        """ Computed the adjusted starting lines based on matching in overlapping regions """
        from operator import itemgetter
        import os

        #first one always starts from zero
        startLine =  [sortedList[0][0]]
        outputs =  [sortedList[0][1]]
        for i in range(1,len(sortedList)):
            # endLine of the first file. we use all the lines of the first file up to endLine
            endLine = sortedList[i][0] - sortedList[i-1][0]
            indx = self.findOverlapLine(sortedList[i-1][1],sortedList[i][1],endLine,width,i-1,i)
            #if indx is not None than indx is the new start line
            #otherwise we use startLine  computed from acquisition time
            #no need to do this for ALOS; otherwise there will be problems when there are multiple prfs and the data are interpolated. C. Liang, 20-dec-2021
            if (self._frames[0].instrument.platform._mission != 'ALOS') and (indx is not None) and (indx + sortedList[i-1][0] != sortedList[i][0]):
                startLine.append(indx + sortedList[i-1][0])
                outputs.append(sortedList[i][1])
                self.logger.info("Changing starting line for frame %d from %d to %d"%(i,endLine,indx))
            else:
                startLine.append(sortedList[i][0])
                outputs.append(sortedList[i][1])

        return startLine,outputs



    # Create the actual Track data by concatenating data from
    # all of the Frames objects together
    def createTrack(self, output):
        """Create the actual Track data by concatenating data from all of the Frames objects together"""
        import os
        import numpy as np
        from operator import itemgetter
        from isceobj import Constants as CN
        
        # 检查第一帧的数据类型来决定处理方式
        is_slc = isinstance(self._frames[0].getImage(), isceobj.Image.SlcImage.SlcImage)
        
        if is_slc:
            # 使用Python处理SLC数据
            self.logger.info("Processing SLC data using Python")
            return self._createTrackSlc(output)
        else:
            # 原有的RAW数据处理方式
            self.logger.info("Processing RAW data using C")
            return self._createTrackRaw(output)

    def _createTrackSlc(self, output):
        """使用Python处理SLC数据"""
        # 计算总宽度和起始行
        totalWidth = max(frame.getNumberOfSamples() for frame in self._frames)
        prf = self._frames[0].getInstrument().getPulseRepetitionFrequency()
        
        # 收集帧信息
        frameInfos = []
        for frame in self._frames:
            startLine = int(round(DTU.timeDeltaToSeconds(frame.getSensingStart()-self._startTime)*prf))
            frameInfos.append({
                'frame': frame,
                'startLine': startLine,
                'filename': frame.getImage().getFilename(),
                'width': frame.getNumberOfSamples(),
                'lines': frame.getNumberOfLines()
            })
        
        # 按开始时间排序
        frameInfos.sort(key=lambda x: x['startLine'])
        
        # 计算总行数
        if len(self._frames) == 1:
            totalLines = self._frames[0].getNumberOfLines()
        else:
            totalLines = frameInfos[-1]['startLine'] + frameInfos[-1]['lines']
        
        self.logger.info(f"合并后的图像尺寸: {totalWidth} x {totalLines}")
        
        # 创建输出数组
        merged_data = np.zeros((totalLines, totalWidth), dtype=np.complex64)
        
        # 合并数据
        for i, frameInfo in enumerate(frameInfos):
            self.logger.info(f"处理第 {i+1}/{len(frameInfos)} 个帧...")
            
            try:
                # 读取帧数据
                frame_data = np.fromfile(frameInfo['filename'], dtype=np.complex64)
                frame_data = frame_data.reshape(frameInfo['lines'], frameInfo['width'])
                
                # 计算写入位置
                start_line = frameInfo['startLine']
                end_line = start_line + frameInfo['lines']
                
                # 如果不是第一帧，检查是否需要处理重叠区域
                if i > 0:
                    prev_end = frameInfos[i-1]['startLine'] + frameInfos[i-1]['lines']
                    if start_line < prev_end:
                        # 处理重叠区域
                        overlap = prev_end - start_line
                        self.logger.info(f"检测到与前一帧重叠 {overlap} 行")
                        # 使用渐变混合重叠区域
                        weights = np.linspace(0, 1, overlap)[:, np.newaxis]
                        overlap_region = merged_data[start_line:prev_end, :frameInfo['width']] * (1 - weights) + \
                                       frame_data[:overlap] * weights
                        merged_data[start_line:prev_end, :frameInfo['width']] = overlap_region
                        # 更新非重叠部分
                        merged_data[prev_end:end_line, :frameInfo['width']] = frame_data[overlap:]
                    else:
                        # 无重叠，直接写入
                        merged_data[start_line:end_line, :frameInfo['width']] = frame_data
                else:
                    # 第一帧直接写入
                    merged_data[start_line:end_line, :frameInfo['width']] = frame_data
                
            except Exception as e:
                self.logger.error(f"处理帧 {i+1} 时出错: {str(e)}")
                raise
        
        # 写入合并后的数据
        self.logger.info(f"写入合并数据到: {output}")
        merged_data.tofile(output)
        
        # 设置Frame属性
        self._frame.setOrbitNumber(self._frames[0].getOrbitNumber())
        self._frame.setSensingStart(self._startTime)
        self._frame.setSensingStop(self._stopTime)
        centerTime = DTU.timeDeltaToSeconds(self._stopTime-self._startTime)/2.0
        self._frame.setSensingMid(self._startTime + datetime.timedelta(microseconds=int(centerTime*1e6)))
        self._frame.setStartingRange(self._nearRange)
        self._frame.setFarRange(self._farRange)
        self._frame.setProcessingFacility(self._frames[0].getProcessingFacility())
        self._frame.setProcessingSystem(self._frames[0].getProcessingSystem())
        self._frame.setProcessingSoftwareVersion(self._frames[0].getProcessingSoftwareVersion())
        self._frame.setPolarization(self._frames[0].getPolarization())
        self._frame.setNumberOfLines(totalLines)
        self._frame.setNumberOfSamples(totalWidth)
        
        # 创建SLC图像对象
        slcImage = isceobj.createSlcImage()
        slcImage.setByteOrder('l')
        slcImage.setFilename(output)
        slcImage.setAccessMode('read')
        slcImage.setWidth(totalWidth)
        slcImage.setLength(totalLines)
        slcImage.setXmax(totalWidth)
        slcImage.setXmin(self._frames[0].getImage().getXmin())
        slcImage.setDataType('CFLOAT')
        slcImage.setImageType('slc')
        
        self._frame.setImage(slcImage)
        
        # 生成头文件和VRT文件
        slcImage.renderHdr()
        slcImage.renderVRT()
        
        return self._frame

    def _createTrackRaw(self, output):
        """原有的RAW数据处理方式"""
        # 把原来createTrack的代码移到这里
        ...

    # Extract the early, late, start and stop times from a Frame object
    # And use this information to update
    def _updateTrackTimes(self,frame):

        if (frame.getSensingStart() < self._startTime):
            self._startTime = frame.getSensingStart()
        if (frame.getSensingStop() > self._stopTime):
            self._stopTime = frame.getSensingStop()
        if (frame.getStartingRange() < self._nearRange):
            self._nearRange = frame.getStartingRange()
        if (frame.getFarRange() > self._farRange):
            self._farRange = frame.getFarRange()
            pass
        pass
    pass

def main():

    tr = Track()
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    line1 = 17731
    width = 21100
    indx = tr.findOverlapLine(file1, file2, line1,width,0,1)

if __name__ == '__main__':
    sys.exit(main())
