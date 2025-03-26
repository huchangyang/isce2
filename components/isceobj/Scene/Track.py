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
        
        # 记录第一帧的信息用于调试
        self.logger.info(f"First frame starts at line 0")
        self.logger.info(f"First frame has {self._frames[0].getNumberOfLines()} lines")
        
        for i in range(1,len(sortedList)):
            # 获取当前帧的行数
            current_frame_lines = self._frames[i].getNumberOfLines()
            prev_frame_lines = self._frames[i-1].getNumberOfLines()
            
            # 计算基于时间的预期重叠
            time_overlap = (self._frames[i-1].getSensingStop() - self._frames[i].getSensingStart()).total_seconds()
            prf = self._frames[i].getInstrument().getPulseRepetitionFrequency()
            expected_overlap_lines = int(abs(time_overlap * prf)) if time_overlap > 0 else 0
            
            # 限制最大重叠比例为20%
            max_overlap = min(int(0.2 * min(current_frame_lines, prev_frame_lines)), expected_overlap_lines)
            
            # 计算endLine（上一帧的结束位置）
            endLine = sortedList[i][0] - sortedList[i-1][0]
            
            # 使用findOverlapLine寻找实际重叠
            indx = self.findOverlapLine(sortedList[i-1][1], sortedList[i][1], endLine, width, i-1, i)
            
            # 记录调试信息
            self.logger.info(f"Frame {i} processing:")
            self.logger.info(f"Expected overlap based on time: {expected_overlap_lines} lines")
            self.logger.info(f"Maximum allowed overlap: {max_overlap} lines")
            self.logger.info(f"Original start line: {sortedList[i][0]}")
            
            if (self._frames[0].instrument.platform._mission != 'ALOS') and (indx is not None):
                # 计算实际重叠
                actual_overlap = endLine - indx if indx < endLine else 0
                
                # 如果重叠过大，调整到最大允许值
                if actual_overlap > max_overlap:
                    indx = endLine - max_overlap
                    self.logger.info(f"Overlap too large ({actual_overlap} lines), limiting to {max_overlap} lines")
                
                # 确保新的起始行不会导致负重叠
                new_start = indx + sortedList[i-1][0]
                if new_start < sortedList[i-1][0]:
                    new_start = sortedList[i-1][0] + prev_frame_lines - max_overlap
                
                startLine.append(new_start)
                outputs.append(sortedList[i][1])
                self.logger.info(f"Adjusted start line for frame {i} from {sortedList[i][0]} to {new_start}")
            else:
                # 如果是ALOS或没找到重叠，使用原始起始行，但仍要检查重叠
                new_start = sortedList[i][0]
                actual_overlap = prev_frame_lines - (new_start - sortedList[i-1][0])
                
                if actual_overlap > max_overlap:
                    new_start = sortedList[i-1][0] + prev_frame_lines - max_overlap
                    self.logger.info(f"Adjusting start line to limit overlap to {max_overlap} lines")
                
                startLine.append(new_start)
                outputs.append(sortedList[i][1])
                self.logger.info(f"Using original start line for frame {i}: {new_start}")

        return startLine,outputs



    # Create the actual Track data by concatenating data from
    # all of the Frames objects together
    def createTrack(self,output):
        import os
        from operator import itemgetter
        from isceobj import Constants as CN
        from ctypes import cdll, c_char_p, c_int, c_ubyte,byref
        import numpy as np
        lib = cdll.LoadLibrary(os.path.dirname(__file__)+'/concatenate.so')

        self.logger.info("Adjusting Sampling Window Start Times for all Frames")
        outputs = []
        totalWidth = 0
        auxList = []

        # 第一步：重采样和填充处理
        for frame in self._frames:
            thisNearRange = frame.getStartingRange()
            thisFarRange = frame.getFarRange()
            left_pad = int(round(
                (thisNearRange - self._nearRange)*
                frame.getInstrument().getRangeSamplingRate()/(CN.SPEED_OF_LIGHT/2.0)))*2
            right_pad = int(round((self._farRange - thisFarRange)*frame.getInstrument().getRangeSamplingRate()/(CN.SPEED_OF_LIGHT/2.0)))*2
            width = frame.getImage().getXmax()
            
            if width - int(width) != 0:
                raise ValueError("frame Xmax is not an integer")
            else:
                width = int(width)

            input = frame.getImage().getFilename()
            with tempfile.NamedTemporaryFile(dir='.', delete=False) as f:
                tempOutput = f.name

            pad_value = int(frame.getInstrument().getInPhaseValue())

            if totalWidth < left_pad + width + right_pad:
                totalWidth = left_pad + width + right_pad

            # 重采样处理
            input_c = c_char_p(bytes(input,'utf-8'))
            output_c = c_char_p(bytes(tempOutput,'utf-8'))
            width_c = c_int(width)
            left_pad_c = c_int(left_pad)
            right_pad_c = c_int(right_pad)
            pad_value_c = c_ubyte(pad_value)
            
            # 记录重采样参数
            self.logger.info(f"Frame resampling parameters:")
            self.logger.info(f"Input file: {input}")
            self.logger.info(f"Width: {width}, Left pad: {left_pad}, Right pad: {right_pad}")
            
            lib.swst_resample(input_c,output_c,byref(width_c),byref(left_pad_c),byref(right_pad_c),byref(pad_value_c))
            
            # 验证重采样结果
            if os.path.exists(tempOutput):
                file_size = os.path.getsize(tempOutput)
                expected_size = (width + left_pad + right_pad) * frame.getNumberOfLines() * 8  # complex64 = 8 bytes
                if file_size != expected_size:
                    self.logger.warning(f"Resampled file size mismatch: got {file_size}, expected {expected_size}")
            
            outputs.append(tempOutput)
            auxList.append(frame.auxFile)

        # 第二步：计算精确的帧起始位置
        prf = self._frames[0].getInstrument().getPulseRepetitionFrequency()
        lineSort = []
        
        # 计算每个帧的起始行
        for i, frame in enumerate(self._frames):
            startLine = int(round(DTU.timeDeltaToSeconds(frame.getSensingStart()-self._startTime)*prf))
            lineSort.append([startLine, outputs[i]])
            
            # 记录帧信息
            self.logger.info(f"Frame {i} information:")
            self.logger.info(f"Start time: {frame.getSensingStart()}")
            self.logger.info(f"Lines: {frame.getNumberOfLines()}")
            self.logger.info(f"Calculated start line: {startLine}")

        sortedList = sorted(lineSort, key=itemgetter(0))
        startLines, outputs = self.reAdjustStartLine(sortedList, totalWidth)

        # 第三步：计算总行数并验证
        if len(self._frames) == 1:
            totalLines = self._frames[0].getNumberOfLines()
        else:
            # 计算总行数，考虑重叠
            totalLines = 0
            for i in range(len(self._frames)):
                if i == 0:
                    # 第一帧全部使用
                    totalLines += self._frames[i].getNumberOfLines()
                else:
                    # 计算与前一帧的重叠
                    overlap = (self._frames[i-1].getSensingStop() - self._frames[i].getSensingStart()).total_seconds()
                    if overlap > 0:
                        # 重叠区域的行数
                        overlap_lines = int(round(overlap * self._frames[i].getInstrument().getPulseRepetitionFrequency()))
                        # 确保重叠行数不超过帧的行数
                        overlap_lines = min(overlap_lines, self._frames[i].getNumberOfLines())
                        # 添加非重叠部分
                        totalLines += self._frames[i].getNumberOfLines() - overlap_lines
                        self.logger.info(f"Frame {i} overlaps with previous frame by {overlap_lines} lines")
                    else:
                        # 无重叠，全部添加
                        totalLines += self._frames[i].getNumberOfLines()
                        self.logger.info(f"No overlap between frames {i-1} and {i}")
            
            # 验证计算结果
            sumFrameLines = sum(frame.getNumberOfLines() for frame in self._frames)
            maxReasonableLines = sumFrameLines  # 不应超过所有帧的总行数
            
            if totalLines > maxReasonableLines:
                self.logger.warning(f"Calculated total lines ({totalLines}) exceeds sum of frame lines ({maxReasonableLines})")
                totalLines = maxReasonableLines
                
            # 验证帧间距
            for i in range(len(startLines)-1):
                frame_gap = startLines[i+1] - (startLines[i] + self._frames[i].getNumberOfLines())
                if frame_gap > 100:  # 允许的最大间隔
                    self.logger.warning(f"Large gap detected between frames {i} and {i+1}: {frame_gap} lines")
                elif frame_gap < -self._frames[i].getNumberOfLines():
                    self.logger.warning(f"Excessive overlap detected between frames {i} and {i+1}")

        self.logger.info(f"Final total lines: {totalLines}")
        totalLines_c = c_int(totalLines)
        
        # 第四步：帧连接
        width_c = c_int(totalWidth)
        numberOfFrames_c = c_int(len(self._frames))
        inputs_c = (c_char_p * len(outputs))()
        for kk in range(len(outputs)):
            inputs_c[kk] = bytes(outputs[kk],'utf-8')
        output_c = c_char_p(bytes(output,'utf-8'))
        startLines_c = (c_int * len(startLines))()
        startLines_c[:] = startLines

        # 执行帧连接
        lib.frame_concatenate(output_c,byref(width_c),byref(totalLines_c),byref(numberOfFrames_c),inputs_c,startLines_c)

        # 验证输出文件
        if os.path.exists(output):
            output_size = os.path.getsize(output)
            expected_size = totalWidth * totalLines * 8  # complex64 = 8 bytes
            if output_size != expected_size:
                self.logger.error(f"Output file size mismatch: got {output_size}, expected {expected_size}")

        # 清理临时文件
        for file in outputs:
            if os.path.exists(file):
                os.unlink(file)

        # 设置帧属性
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

        # 创建输出图像
        firstImage = self._frames[0].getImage()
        imageType = firstImage.__class__.__name__
        
        if imageType == 'SlcImage':
            image = isceobj.createSlcImage()
            image.setDataType('CFLOAT')
            image.setImageType('slc')
        else:
            image = isceobj.createRawImage()
            
        image.setByteOrder('l')
        image.setFilename(output)
        image.setAccessMode('r')
        image.setWidth(totalWidth)
        image.setXmax(totalWidth)
        image.setXmin(self._frames[0].getImage().getXmin())
        self._frame.setImage(image)


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
