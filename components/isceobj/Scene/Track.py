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
from isceobj import Constants as CN
import isceobj
import numpy as np
import traceback

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
    def findOverlapLine(self, file1, file2, line1, width, frameNum1, frameNum2):
        import numpy as np
        import array
        
        # 增加搜索范围
        searchNumLines = 10  # 增加搜索行数
        searchSize = width  # 使用整行进行匹配
        numTries = 50      # 增加尝试次数
        
        # 读取第二帧的前几行作为参考
        with open(file2, 'rb') as fin2:
            arr2 = array.array('b')
            arr2.fromfile(fin2, width * searchNumLines)
            buf2 = np.array(arr2, dtype=np.int8).reshape(-1, width)
        
        max_correlation = 0
        best_offset = None
        
        with open(file1, 'rb') as fin1:
            # 在第一帧的末尾附近搜索
            for i in range(-numTries, numTries):
                test_line = line1 + i
                if test_line < 0:
                    continue
                    
                # 读取第一帧的对应行
                fin1.seek(test_line * width, 0)
                arr1 = array.array('b')
                try:
                    arr1.fromfile(fin1, width * searchNumLines)
                except EOFError:
                    continue
                    
                buf1 = np.array(arr1, dtype=np.int8).reshape(-1, width)
                
                # 计算相关性
                correlation = np.abs(np.corrcoef(buf1.flatten(), buf2.flatten())[0,1])
                
                if correlation > max_correlation:
                    max_correlation = correlation
                    best_offset = test_line
        
        if best_offset is None:
            self.logger.warning(f"Cannot find good overlap between frame {frameNum1} and frame {frameNum2}")
            return line1
        
        self.logger.info(f"Found best overlap at line {best_offset} with correlation {max_correlation}")
        return best_offset

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
            #otherwise we use startLine computed from acquisition time
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
        
        # check the data type of the first frame
        is_slc = isinstance(self._frames[0].getImage(), isceobj.Image.SlcImage.SlcImage)
        
        if is_slc:
            self.logger.info("Processing SLC data using Python")
            return self._createTrackSlc(output)
        else:
            self.logger.info("Processing RAW data using C")
            return self._createTrackRaw(output)

    def _createTrackSlc(self, output):
        """处理SLC数据的简化版本，直接通过成像时间确定起始行进行合并"""
        from isceobj import Constants as CN
        
        # 1. 按照成像时间排序所有帧
        sorted_frames = sorted(self._frames, key=lambda x: x.getSensingStart())
        width = sorted_frames[0].getNumberOfSamples()
        prf = sorted_frames[0].getInstrument().getPulseRepetitionFrequency()
        
        # 2. 计算每帧的起始行，使用更精确的方式处理小数
        start_lines = []
        actual_lines = []  # 记录实际使用的行数
        for i, frame in enumerate(sorted_frames):
            if i == 0:
                start_lines.append(0)
                actual_lines.append(frame.getNumberOfLines())
            else:
                # 使用高精度计算时间差对应的行数
                time_diff = (frame.getSensingStart() - sorted_frames[0].getSensingStart()).total_seconds()
                start_line_float = time_diff * prf
                
                # # 检查是否需要调整以避免1行的间隙
                # if i > 0:
                #     prev_end = start_lines[i-1] + actual_lines[i-1]
                #     gap = int(round(start_line_float)) - prev_end
                    
                #     if gap == 1:  # 如果有1行的间隙
                #         start_line = prev_end  # 直接接续前一帧
                #     elif gap == -1:  # 如果有1行的重叠
                #         start_line = prev_end - 1  # 略微重叠以保持连续性
                #     else:
                #         start_line = int(round(start_line_float))
                # else:
                start_line = int(start_line_float)
                
                start_lines.append(start_line)
                actual_lines.append(frame.getNumberOfLines())
                
                self.logger.info(f"Frame {i}: start_line_float={start_line_float:.2f}")
        
        # 3. 计算总行数
        total_lines = start_lines[-1] + sorted_frames[-1].getNumberOfLines()
        merged_data = np.zeros((total_lines, width), dtype=np.complex64)
        
        # 4. 逐帧合并数据
        for i, frame in enumerate(sorted_frames):
            # 读取当前帧数据
            with open(frame.image.getFilename(), 'rb') as f:
                data = np.fromfile(f, dtype=np.complex64)
                data = data.reshape(frame.getNumberOfLines(), width)
            
            # 将当前帧数据写入对应位置
            start_line = start_lines[i]
            merged_data[start_line:start_line + frame.getNumberOfLines()] = data
        
        # 5. 保存合并结果
        merged_data.tofile(output)
        
        # 6. 设置Frame属性
        self._frame.setNumberOfLines(total_lines)
        self._frame.setNumberOfSamples(width)
        
        # 设置基本属性
        self._frame.setOrbitNumber(sorted_frames[0].getOrbitNumber())
        self._frame.setSensingStart(self._startTime)
        self._frame.setSensingStop(self._stopTime)
        centerTime = DTU.timeDeltaToSeconds(self._stopTime-self._startTime)/2.0
        self._frame.setSensingMid(self._startTime + datetime.timedelta(microseconds=int(centerTime*1e6)))
        self._frame.setStartingRange(self._nearRange)
        self._frame.setFarRange(self._farRange)
        
        # 设置处理信息
        self._frame.setProcessingFacility(sorted_frames[0].getProcessingFacility())
        self._frame.setProcessingSystem(sorted_frames[0].getProcessingSystem())
        self._frame.setProcessingSoftwareVersion(sorted_frames[0].getProcessingSoftwareVersion())
        self._frame.setPolarization(sorted_frames[0].getPolarization())
        
        # 创建并设置SLC图像对象
        slcImage = isceobj.createSlcImage()
        slcImage.setFilename(output)
        slcImage.setAccessMode('read')
        slcImage.setWidth(width)
        slcImage.setLength(total_lines)
        slcImage.setXmin(0)
        slcImage.setXmax(width)
        slcImage.setDataType('CFLOAT')
        slcImage.scheme = 'BIP'
        slcImage.setByteOrder('l')
        slcImage.imageType = 'slc'
        
        # 设置图像到Frame
        self._frame.setImage(slcImage)
        
        # 生成头文件和VRT文件
        slcImage.renderHdr()
        slcImage.renderVRT()
        
        return self._frame

    def _createTrackRaw(self, output):
        """The original RAW data processing method"""
        import os
        import tempfile
        from operator import itemgetter
        from isceobj import Constants as CN
        from ctypes import cdll, c_char_p, c_int, c_ubyte, byref
        
        lib = cdll.LoadLibrary(os.path.dirname(__file__)+'/concatenate.so')
        # Perhaps we should check to see if Xmin is 0, if it is not, strip off the header
        self.logger.info("Adjusting Sampling Window Start Times for all Frames")
        # Iterate over each frame object, and calculate the number of samples with which to pad it on the left and right
        outputs = []
        totalWidth = 0
        auxList = []
        for frame in self._frames:
            # Calculate the amount of padding
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
            with tempfile.NamedTemporaryFile(dir='.') as f:
                tempOutput = f.name

            pad_value = int(frame.getInstrument().getInPhaseValue())

            if totalWidth < left_pad + width + right_pad:
                totalWidth = left_pad + width + right_pad
            # Resample this frame with swst_resample
            input_c = c_char_p(bytes(input,'utf-8'))
            output_c = c_char_p(bytes(tempOutput,'utf-8'))
            width_c = c_int(width)
            left_pad_c = c_int(left_pad)
            right_pad_c = c_int(right_pad)
            pad_value_c = c_ubyte(pad_value)
            lib.swst_resample(input_c,output_c,byref(width_c),byref(left_pad_c),byref(right_pad_c),byref(pad_value_c))
            outputs.append(tempOutput)
            auxList.append(frame.auxFile)

        #this step construct the aux file withe the pulsetime info for the all set of frames
        self.createAuxFile(auxList,output + '.aux')
        # This assumes that all of the frames to be concatenated are sampled at the same PRI
        prf = self._frames[0].getInstrument().getPulseRepetitionFrequency()
        # Calculate the starting output line of each scene
        i = 0
        lineSort = []
        # the listSort has 2 elements: a line start number which is the position of that specific frame
        # computed from acquisition time and the  corresponding file name
        for frame in self._frames:
            startLine = int(round(DTU.timeDeltaToSeconds(frame.getSensingStart()-self._startTime)*prf))
            lineSort.append([startLine,outputs[i]])
            i += 1

        sortedList = sorted(lineSort, key=itemgetter(0)) # sort by line number i.e. acquisition time
        startLines, outputs = self.reAdjustStartLine(sortedList,totalWidth)


        self.logger.info("Concatenating Frames along Track")
        # this is a hack since the length of the file could be actually different from the one computed using start and stop time. it only matters the last frame added
        import os

        fileSize = os.path.getsize(outputs[-1])

        numLines = fileSize//totalWidth + startLines[-1]
        totalLines_c = c_int(numLines)
        # Next, call frame_concatenate
        width_c = c_int(totalWidth) # Width of each frame (with the padding added in swst_resample)
        numberOfFrames_c = c_int(len(self._frames))
        inputs_c = (c_char_p * len(outputs))() # These are the inputs to frame_concatenate, but the outputs from swst_resample
        for kk in range(len(outputs)):
            inputs_c[kk] = bytes(outputs[kk],'utf-8')
        output_c = c_char_p(bytes(output,'utf-8'))
        startLines_c = (c_int * len(startLines))()
        startLines_c[:] = startLines
        lib.frame_concatenate(output_c,byref(width_c),byref(totalLines_c),byref(numberOfFrames_c),inputs_c,startLines_c)

        # Clean up the temporary output files from swst_resample
        for file in outputs:
            os.unlink(file)

        orbitNum = self._frames[0].getOrbitNumber()
        first_line_utc = self._startTime
        last_line_utc = self._stopTime
        centerTime = DTU.timeDeltaToSeconds(last_line_utc-first_line_utc)/2.0
        center_line_utc = first_line_utc + datetime.timedelta(microseconds=int(centerTime*1e6))
        procFac = self._frames[0].getProcessingFacility()
        procSys = self._frames[0].getProcessingSystem()
        procSoft = self._frames[0].getProcessingSoftwareVersion()
        pol = self._frames[0].getPolarization()
        xmin = self._frames[0].getImage().getXmin()


        self._frame.setOrbitNumber(orbitNum)
        self._frame.setSensingStart(first_line_utc)
        self._frame.setSensingMid(center_line_utc)
        self._frame.setSensingStop(last_line_utc)
        self._frame.setStartingRange(self._nearRange)
        self._frame.setFarRange(self._farRange)
        self._frame.setProcessingFacility(procFac)
        self._frame.setProcessingSystem(procSys)
        self._frame.setProcessingSoftwareVersion(procSoft)
        self._frame.setPolarization(pol)
        self._frame.setNumberOfLines(numLines)
        self._frame.setNumberOfSamples(width)
        # add image to frame
        rawImage = isceobj.createRawImage()
        rawImage.setByteOrder('l')
        rawImage.setFilename(output)
        rawImage.setAccessMode('r')
        rawImage.setWidth(totalWidth)
        rawImage.setXmax(totalWidth)
        rawImage.setXmin(xmin)
        self._frame.setImage(rawImage)
        
        return self._frame

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
