#!/usr/bin/python3

# Reader for Lutan-1 SLC data
# Used Sentinel1.py and ALOS.py as templates
# Author: Bryan Marfito, EOS-RS


import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
import datetime
import isce
import isceobj
from isceobj.Planet.Planet import Planet
from iscesys.Component.Component import Component
from isceobj.Sensor.Sensor import Sensor
from isceobj.Scene.Frame import Frame
from isceobj.Orbit.Orbit import StateVector, Orbit
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
from isceobj.Sensor import tkfunc


lookMap = { 'RIGHT' : -1,
            'LEFT' : 1}

# Antenna dimensions 9.8 x 3.4 m
antennaLength = 9.8

XML = Component.Parameter('_xmlList',
        public_name = 'XML',
        default = '',
        container=list,
        type=str,
        doc = 'List of names of Lutan-1 XML metadata files')


TIFF = Component.Parameter('_tiffList',
                            public_name ='TIFF',
                            default = '',
                            container=list,
                            type=str,
                            doc = 'List of names of Lutan-1 TIFF image files')

ORBIT_FILE = Component.Parameter('_orbitFileList',
                            public_name ='ORBITFILE',
                            default = '',
                            container=list,
                            type=str,
                            doc = 'List of names of orbit files')

XML_CONFIG = Component.Parameter('_xmlConfig',
                            public_name ='XML_CONFIG',
                            default = None,
                            type=str,
                            doc = 'XML configuration file with SLC and orbit directories')

SLC_DIR = Component.Parameter('_slcDir',
                            public_name ='SLC_DIR',
                            default = None,
                            type=str,
                            doc = 'Directory containing SLC TIFF files')

ORBIT_DIR = Component.Parameter('_orbitDir',
                            public_name ='ORBIT_DIR',
                            default = None,
                            type=str,
                            doc = 'Directory containing orbit files')

FILTER_METHOD = Component.Parameter('filterMethod',
                            public_name ='filterMethod',
                            default = 'combined_weighted_filter',
                            type=str,
                            doc = 'Orbit filter method (poly_filter, physics_filter, combined_weighted_filter)')


class Lutan1(Sensor):

    "Class for Lutan-1 SLC data"
    
    family = 'l1sm'
    logging_name = 'isce.sensor.Lutan1'

    parameter_list = (TIFF, ORBIT_FILE, FILTER_METHOD, XML_CONFIG, SLC_DIR, ORBIT_DIR) + Sensor.parameter_list

    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self.frameList = []
        self._imageFileList = []
        self._xmlFileList = []
        self.frame = None
        self.doppler_coeff = None
        self.filterMethod = 'combined_weighted_filter'
        self._tiff = None
        self._orbitFile = None
        self._xml = None

    def validateUserInputs(self):
        '''
        验证用户输入并自动查找文件
        参考Sentinel1.py中的实现
        '''
        # 如果提供了XML配置文件，从中读取
        if self._xmlConfig:
            self.loadFromXML()
            return

        # 如果提供了SLC目录，自动查找TIFF文件
        if self._slcDir and not self._tiffList:
            self.logger.info(f"Searching for TIFF files in {self._slcDir}")
            self._tiffList = sorted(glob.glob(os.path.join(self._slcDir, "*.tiff")))
            if not self._tiffList:
                self.logger.warning(f"No TIFF files found in {self._slcDir}")
                raise Exception(f"No TIFF files found in {self._slcDir}")
            else:
                self.logger.info(f"Found {len(self._tiffList)} TIFF files")
                for i, tiff in enumerate(self._tiffList):
                    self.logger.info(f"TIFF {i+1}: {tiff}")

        # 如果提供了轨道目录，自动查找轨道文件
        if self._orbitDir and not self._orbitFileList:
            self.logger.info(f"Searching for orbit files in {self._orbitDir}")
            self._orbitFileList = sorted(glob.glob(os.path.join(self._orbitDir, "*.xml")))
            if not self._orbitFileList:
                self.logger.warning(f"No orbit files found in {self._orbitDir}")
                self._orbitFileList = []
            else:
                self.logger.info(f"Found {len(self._orbitFileList)} orbit files")
                for i, orbit in enumerate(self._orbitFileList):
                    self.logger.info(f"Orbit {i+1}: {orbit}")

        # 检查是否提供了TIFF文件
        if not self._tiffList:
            raise Exception("No TIFF files provided. Use TIFF, SLC_DIR or XML_CONFIG parameter.")

        # 如果轨道文件数量与TIFF文件数量不匹配，尝试自动匹配
        if len(self._orbitFileList) > 0 and len(self._orbitFileList) != len(self._tiffList):
            self.logger.warning("Number of orbit files does not match number of TIFF files")
            self.logger.info("Attempting to match orbit files with TIFF files...")
            
            # 尝试根据日期匹配轨道文件
            # 对于同一日期的多景SLC，应该使用同一个轨道文件
            try:
                # 提取每个TIFF文件名中的日期信息
                tiff_dates = []
                for tiff in self._tiffList:
                    # 假设文件名格式包含日期，例如：LT1B_MONO_KRN_STRIP2_000643_E37.4_N37.1_20220411_SLC_HH_L1A_0000010037.tiff
                    # 提取日期部分（这里假设日期格式为YYYYMMDD，位于文件名的第8个下划线之后）
                    basename = os.path.basename(tiff)
                    parts = basename.split('_')
                    if len(parts) >= 8:
                        date_str = parts[7]  # 假设日期在第8个位置
                        if len(date_str) == 8 and date_str.isdigit():  # 确保是8位数字（YYYYMMDD）
                            tiff_dates.append(date_str)
                        else:
                            tiff_dates.append(None)
                    else:
                        tiff_dates.append(None)
                
                # 提取每个轨道文件名中的日期信息
                orbit_dates = []
                for orbit in self._orbitFileList:
                    # 假设轨道文件名格式包含日期，例如：LT1B_20230607102726409_V20220410T235500_20220412T000500_ABSORBIT_SCIE.xml
                    # 提取日期部分（这里假设日期格式为YYYYMMDD，位于文件名的第3个下划线之后）
                    basename = os.path.basename(orbit)
                    parts = basename.split('_')
                    if len(parts) >= 4:
                        date_str = parts[2][1:9]  # 假设日期在第3个位置，格式为VYYYYMMDD...
                        if len(date_str) == 8 and date_str.isdigit():  # 确保是8位数字（YYYYMMDD）
                            orbit_dates.append(date_str)
                        else:
                            orbit_dates.append(None)
                    else:
                        orbit_dates.append(None)
                
                # 为每个TIFF文件匹配对应的轨道文件
                matched_orbit_files = []
                for i, tiff_date in enumerate(tiff_dates):
                    if tiff_date is None:
                        matched_orbit_files.append(None)
                        continue
                    
                    # 查找日期匹配的轨道文件
                    matched = False
                    for j, orbit_date in enumerate(orbit_dates):
                        if orbit_date is not None and orbit_date == tiff_date:
                            matched_orbit_files.append(self._orbitFileList[j])
                            matched = True
                            break
                    
                    if not matched:
                        matched_orbit_files.append(None)
                
                # 如果所有TIFF文件都找到了匹配的轨道文件，使用匹配结果
                if all(orbit is not None for orbit in matched_orbit_files):
                    self.logger.info("Successfully matched orbit files based on date")
                    self._orbitFileList = matched_orbit_files
                else:
                    # 如果只有一个轨道文件，假设它适用于所有TIFF文件
                    if len(self._orbitFileList) == 1:
                        self.logger.info("Using the single orbit file for all TIFF files")
                        self._orbitFileList = [self._orbitFileList[0]] * len(self._tiffList)
                    else:
                        self.logger.warning("Could not match orbit files based on date, proceeding without orbit files")
                        self._orbitFileList = []
            except Exception as e:
                self.logger.warning(f"Error matching orbit files: {str(e)}")
                # 如果只有一个轨道文件，假设它适用于所有TIFF文件
                if len(self._orbitFileList) == 1:
                    self.logger.info("Using the single orbit file for all TIFF files")
                    self._orbitFileList = [self._orbitFileList[0]] * len(self._tiffList)
                else:
                    self.logger.warning("Proceeding without orbit files")
                    self._orbitFileList = []

    def parse(self):
        xmlFileName = self._tiff[:-4] + "meta.xml"
        self._xml = xmlFileName

        with open(self._xml, 'r') as fid:
            xmlstr = fid.read()
        
        self._xml_root = ET.fromstring(xmlstr)
        self.populateMetadata()
        fid.close()

        if self._orbitFile:
            # Check if orbit file exists or not
            if os.path.isfile(self._orbitFile):
                orb = self.extractOrbit()
                self.frame.orbit.setOrbitSource(os.path.basename(self._orbitFile))
            else:
                self.logger.warning("Orbit file not found: %s" % self._orbitFile)
                self.logger.info("Using createOrbit() to process orbit data")
                orb = self.createOrbit()
                self.frame.orbit.setOrbitSource('Annotation')
        else:
            self.logger.warning("No orbit file provided.")
            self.logger.info("Using createOrbit() to process orbit data")
            orb = self.createOrbit()
            self.frame.orbit.setOrbitSource('Annotation')

        for sv in orb:
            self.frame.orbit.addStateVector(sv)

    def validate_orbit(self, orbit):
        """验证轨道数据的有效性"""
        if not orbit._stateVectors:
            self.logger.error("No state vectors found in orbit")
            return False
        
        # 转换为列表进行操作
        state_vectors = list(orbit._stateVectors)
        if len(state_vectors) < 4:
            self.logger.error(f"Insufficient state vectors: {len(state_vectors)}")
            return False
        
        # 验证时间范围
        times = [sv.getTime() for sv in state_vectors]
        min_time = min(times)
        max_time = max(times)
        
        if self.frame.getSensingStart() < min_time or self.frame.getSensingStop() > max_time:
            self.logger.error("Frame sensing time outside orbit data range")
            self.logger.error(f"Orbit time range: {min_time} to {max_time}")
            self.logger.error(f"Frame time range: {self.frame.getSensingStart()} to {self.frame.getSensingStop()}")
            return False
        
        # 验证数据连续性
        times.sort()
        for i in range(1, len(times)):
            dt = (times[i] - times[i-1]).total_seconds()
            if dt > 2.0:  # 假设正常采样间隔为1秒
                self.logger.warning(f"Large gap ({dt} seconds) in orbit data between {times[i-1]} and {times[i]}")
        
        return True

    def convertToDateTime(self,string):
        dt = datetime.datetime.strptime(string,"%Y-%m-%dT%H:%M:%S.%f")
        return dt


    def grab_from_xml(self, path):
        '''
        从XML中获取指定路径的值
        参考Sentinel1.py中的实现
        '''
        try:
            res = self._xml_root.find(path).text
        except:
            raise Exception('Tag= %s not found'%(path))

        if res is None:
            raise Exception('Tag = %s not found'%(path))
        
        return res
    

    def populateMetadata(self):
        mission = self.grab_from_xml('generalHeader/mission')
        polarization = self.grab_from_xml('productInfo/acquisitionInfo/polarisationMode')
        frequency = float(self.grab_from_xml('instrument/radarParameters/centerFrequency'))
        passDirection = self.grab_from_xml('productInfo/missionInfo/orbitDirection')
        rangePixelSize = float(self.grab_from_xml('productInfo/imageDataInfo/imageRaster/columnSpacing'))
        azimuthPixelSize = float(self.grab_from_xml('productInfo/imageDataInfo/imageRaster/rowSpacing'))
        rangeSamplingRate = Const.c/(2.0*rangePixelSize)

        prf = float(self.grab_from_xml('instrument/settings/settingRecord/PRF'))
        lines = int(self.grab_from_xml('productInfo/imageDataInfo/imageRaster/numberOfRows'))
        samples = int(self.grab_from_xml('productInfo/imageDataInfo/imageRaster/numberOfColumns'))

        startingRange = float(self.grab_from_xml('productInfo/sceneInfo/rangeTime/firstPixel'))*Const.c/2.0
        #slantRange = float(self.grab_from_xml('productSpecific/complexImageInfo/'))
        incidenceAngle = float(self.grab_from_xml('productInfo/sceneInfo/sceneCenterCoord/incidenceAngle'))
        dataStartTime = self.convertToDateTime(self.grab_from_xml('productInfo/sceneInfo/start/timeUTC'))
        dataStopTime = self.convertToDateTime(self.grab_from_xml('productInfo/sceneInfo/stop/timeUTC'))
        pulseLength = float(self.grab_from_xml('processing/processingParameter/rangeCompression/chirps/referenceChirp/pulseLength'))
        pulseBandwidth = float(self.grab_from_xml('processing/processingParameter/rangeCompression/chirps/referenceChirp/pulseBandwidth'))
        chirpSlope = pulseBandwidth/pulseLength

        if self.grab_from_xml('processing/processingParameter/rangeCompression/chirps/referenceChirp/chirpSlope') == "DOWN":
            chirpSlope = -1.0 * chirpSlope
        else:
            pass

        # Check for satellite's look direction
        if self.grab_from_xml('productInfo/acquisitionInfo/lookDirection') == "LEFT":
            lookSide = lookMap['LEFT']
            print("Look direction: LEFT")
        else:
            lookSide = lookMap['RIGHT']
            print("Look direction: RIGHT")

        processingFacility = self.grab_from_xml('productInfo/generationInfo/level1ProcessingFacility')

        # Platform parameters
        platform = self.frame.getInstrument().getPlatform()
        platform.setPlanet(Planet(pname='Earth'))
        platform.setMission(mission)
        platform.setPointingDirection(lookSide)
        platform.setAntennaLength(antennaLength)

        # Instrument parameters
        instrument = self.frame.getInstrument()
        instrument.setRadarFrequency(frequency)
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setPulseLength(pulseLength)
        instrument.setChirpSlope(chirpSlope)
        instrument.setIncidenceAngle(incidenceAngle)
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setRangeSamplingRate(rangeSamplingRate)
        instrument.setPulseLength(pulseLength)
        
        # 设置 inPhaseValue 和 quadratureValue，参考 ALOS.py
        # 这些值用于 Track.py 中的 pad_value
        instrument.setInPhaseValue(63.5)
        instrument.setQuadratureValue(63.5)

        # Frame parameters
        self.frame.setSensingStart(dataStartTime)
        self.frame.setSensingStop(dataStopTime)
        self.frame.setProcessingFacility(processingFacility)

        # Two-way travel time 
        diffTime = DTUtil.timeDeltaToSeconds(dataStopTime - dataStartTime) / 2.0
        sensingMid = dataStartTime + datetime.timedelta(microseconds=int(diffTime*1e6))
        self.frame.setSensingMid(sensingMid)
        self.frame.setPassDirection(passDirection)
        self.frame.setPolarization(polarization)
        self.frame.setStartingRange(startingRange)
        self.frame.setFarRange(startingRange +  (samples - 1) * rangePixelSize)
        self.frame.setNumberOfLines(lines)
        self.frame.setNumberOfSamples(samples)

        return


    def extractOrbit(self):
        '''
        Extract orbit information from the orbit file
        '''
        orb = Orbit()
        orb.configure()

        # 设置时间范围，只收集成像时间段附近的状态向量
        margin = datetime.timedelta(minutes=30.0)  # 设置30分钟的时间余量
        tstart = self.frame.getSensingStart() - margin
        tend = self.frame.getSensingStop() + margin

        self.logger.info(f"Extracting orbit for time range: {tstart} to {tend}")
        self.logger.info(f"Orbit file: {self._orbitFile}")

        if not os.path.exists(self._orbitFile):
            self.logger.error(f"Orbit file does not exist: {self._orbitFile}")
            return None

        file_ext = os.path.splitext(self._orbitFile)[1].lower()

        if file_ext == '.xml':
            try:
                fp = open(self._orbitFile, 'r')
            except IOError as strerr:
                self.logger.error(f"IOError: {strerr}")
                return None
            
            _xml_root = ET.ElementTree(file=fp).getroot()
            node = _xml_root.find('Data_Block/List_of_OSVs')
            
            if node is None:
                self.logger.error("Could not find 'Data_Block/List_of_OSVs' in orbit file")
                fp.close()
                return None
            
            count = 0
            min_time = datetime.datetime(year=datetime.MAXYEAR, month=1, day=1)
            max_time = datetime.datetime(year=datetime.MINYEAR, month=1, day=1)
            
            # 只收集时间范围内的状态向量
            all_vectors = []
            for child in node:
                try:
                    timestamp_str = child.find('UTC').text
                    timestamp = self.convertToDateTime(timestamp_str)
                    
                    # 只处理时间范围内的状态向量
                    if timestamp < tstart or timestamp > tend:
                        continue
                    
                    # 更新最小和最大时间
                    if timestamp < min_time:
                        min_time = timestamp
                    if timestamp > max_time:
                        max_time = timestamp
                    
                    pos = []
                    vel = []
                    for tag in ['VX', 'VY', 'VZ']:
                        vel.append(float(child.find(tag).text))

                    for tag in ['X', 'Y', 'Z']:
                        pos.append(float(child.find(tag).text))

                    vec = StateVector()
                    vec.setTime(timestamp)
                    vec.setPosition(pos)
                    vec.setVelocity(vel)
                    all_vectors.append(vec)
                    count += 1
                except Exception as e:
                    self.logger.warning(f"Error parsing orbit state vector: {e}")
                    continue

            fp.close()
            
            self.logger.info(f"Extracted {count} orbit state vectors")
            self.logger.info(f"Orbit time range: {min_time} to {max_time}")
            self.logger.info(f"SLC sensing time range: {self.frame.getSensingStart()} to {self.frame.getSensingStop()}")
            
            if count == 0:
                self.logger.error("No orbit state vectors found in the specified time range")
                self.logger.error(f"Orbit file time range: {min_time} to {max_time}")
                self.logger.error(f"Required time range: {tstart} to {tend}")
                return None

            # 按时间排序
            all_vectors.sort(key=lambda x: x.getTime())
            
            # 移除重复的状态向量
            unique_vectors = []
            prev_time = None
            for sv in all_vectors:
                curr_time = sv.getTime()
                if prev_time is None or curr_time != prev_time:
                    unique_vectors.append(sv)
                    prev_time = curr_time
            
            # 确保至少有4个状态向量
            if len(unique_vectors) < 4:
                self.logger.warning(f"Only {len(unique_vectors)} unique state vectors found, attempting to extend time range")
                
                # 获取第一个和最后一个向量的时间
                t_start = unique_vectors[0].getTime()
                t_end = unique_vectors[-1].getTime()
                
                # 扩展时间范围
                margin = datetime.timedelta(minutes=30)
                t_start_extended = t_start - margin
                t_end_extended = t_end + margin
                
                # 重新收集扩展时间范围内的状态向量
                extended_vectors = []
                for child in node:
                    try:
                        timestamp_str = child.find('UTC').text
                        timestamp = self.convertToDateTime(timestamp_str)
                        
                        if t_start_extended <= timestamp <= t_end_extended:
                            pos = []
                            vel = []
                            for tag in ['VX', 'VY', 'VZ']:
                                vel.append(float(child.find(tag).text))
                            for tag in ['X', 'Y', 'Z']:
                                pos.append(float(child.find(tag).text))
                            
                            vec = StateVector()
                            vec.setTime(timestamp)
                            vec.setPosition(pos)
                            vec.setVelocity(vel)
                            extended_vectors.append(vec)
                    except Exception as e:
                        self.logger.warning(f"Error parsing orbit state vector: {e}")
                        continue
                
                # 重新排序和去重
                extended_vectors.sort(key=lambda x: x.getTime())
                unique_vectors = []
                prev_time = None
                for sv in extended_vectors:
                    curr_time = sv.getTime()
                    if prev_time is None or curr_time != prev_time:
                        unique_vectors.append(sv)
                        prev_time = curr_time
            
            if len(unique_vectors) < 4:
                self.logger.error(f"Still only found {len(unique_vectors)} unique state vectors after extending time range")
                return None
            
            # 添加到轨道对象
            for sv in unique_vectors:
                orb.addStateVector(sv)
            
            self.logger.info(f"Created orbit with {len(unique_vectors)} state vectors")
            self.logger.info(f"Orbit time range: {unique_vectors[0].getTime()} to {unique_vectors[-1].getTime()}")
            
            return orb

        elif file_ext == '.txt':
            with open(self._orbitFile, 'r') as fid:
                for line in fid:
                    if not line.startswith('#'):
                        break
                
                count = 0
                min_time = datetime.datetime(year=datetime.MAXYEAR, month=1, day=1)
                max_time = datetime.datetime(year=datetime.MINYEAR, month=1, day=1)
                
                # 只收集时间范围内的状态向量
                all_vectors = []
                for line in fid:
                    try:
                        fields = line.split()
                        if len(fields) >= 13:
                            year = int(fields[0])
                            month = int(fields[1])
                            day = int(fields[2])
                            hour = int(fields[3])
                            minute = int(fields[4])
                            second = float(fields[5])
                            
                            int_second = int(second)
                            microsecond = int((second - int_second) * 1e6)
                            # Convert to datetime   
                            timestamp = datetime.datetime(year, month, day, hour, minute, int_second, microsecond)
                            
                            # 只处理时间范围内的状态向量
                            if timestamp < tstart or timestamp > tend:
                                continue
                            
                            # 更新最小和最大时间
                            if timestamp < min_time:
                                min_time = timestamp
                            if timestamp > max_time:
                                max_time = timestamp
                            
                            pos = [float(fields[6]), float(fields[7]), float(fields[8])]
                            vel = [float(fields[9]), float(fields[10]), float(fields[11])]
                            vec = StateVector()
                            vec.setTime(timestamp)
                            vec.setPosition(pos)
                            vec.setVelocity(vel)
                            all_vectors.append(vec)
                            count += 1
                    except Exception as e:
                        self.logger.warning(f"Error parsing orbit state vector: {e}")
                        continue
                
                self.logger.info(f"Extracted {count} orbit state vectors")
                self.logger.info(f"Orbit time range: {min_time} to {max_time}")
                self.logger.info(f"SLC sensing time range: {self.frame.getSensingStart()} to {self.frame.getSensingStop()}")
                
                if count == 0:
                    self.logger.error("No orbit state vectors found in the specified time range")
                    self.logger.error(f"Orbit file time range: {min_time} to {max_time}")
                    self.logger.error(f"Required time range: {tstart} to {tend}")
                    return None

                # 按时间排序
                all_vectors.sort(key=lambda x: x.getTime())
                
                # 移除重复的状态向量
                unique_vectors = []
                prev_time = None
                for sv in all_vectors:
                    curr_time = sv.getTime()
                    if prev_time is None or curr_time != prev_time:
                        unique_vectors.append(sv)
                        prev_time = curr_time
                
                # 确保至少有4个状态向量
                if len(unique_vectors) < 4:
                    self.logger.warning(f"Only {len(unique_vectors)} unique state vectors found, attempting to extend time range")
                    
                    # 获取第一个和最后一个向量的时间
                    t_start = unique_vectors[0].getTime()
                    t_end = unique_vectors[-1].getTime()
                    
                    # 扩展时间范围
                    margin = datetime.timedelta(minutes=30)
                    t_start_extended = t_start - margin
                    t_end_extended = t_end + margin
                    
                    # 重新收集扩展时间范围内的状态向量
                    fid.seek(0)  # 重置文件指针
                    for line in fid:
                        if line.startswith('#'):
                            continue
                        try:
                            fields = line.split()
                            if len(fields) >= 13:
                                year = int(fields[0])
                                month = int(fields[1])
                                day = int(fields[2])
                                hour = int(fields[3])
                                minute = int(fields[4])
                                second = float(fields[5])
                                
                                int_second = int(second)
                                microsecond = int((second - int_second) * 1e6)
                                timestamp = datetime.datetime(year, month, day, hour, minute, int_second, microsecond)
                                
                                if t_start_extended <= timestamp <= t_end_extended:
                                    pos = [float(fields[6]), float(fields[7]), float(fields[8])]
                                    vel = [float(fields[9]), float(fields[10]), float(fields[11])]
                                    vec = StateVector()
                                    vec.setTime(timestamp)
                                    vec.setPosition(pos)
                                    vec.setVelocity(vel)
                                    extended_vectors.append(vec)
                        except Exception as e:
                            self.logger.warning(f"Error parsing orbit state vector: {e}")
                            continue
                    
                    # 重新排序和去重
                    extended_vectors.sort(key=lambda x: x.getTime())
                    unique_vectors = []
                    prev_time = None
                    for sv in extended_vectors:
                        curr_time = sv.getTime()
                        if prev_time is None or curr_time != prev_time:
                            unique_vectors.append(sv)
                            prev_time = curr_time
                
                if len(unique_vectors) < 4:
                    self.logger.error(f"Still only found {len(unique_vectors)} unique state vectors after extending time range")
                    return None
                
                # 添加到轨道对象
                for sv in unique_vectors:
                    orb.addStateVector(sv)
                
                self.logger.info(f"Created orbit with {len(unique_vectors)} state vectors")
                self.logger.info(f"Orbit time range: {unique_vectors[0].getTime()} to {unique_vectors[-1].getTime()}")
                
                return orb
        else:
            self.logger.error(f"Unsupported orbit file extension: {file_ext}")
            return None

    def filter_orbit(self, times, positions, velocities):
        """使用标准多项式拟合滤波轨道数据"""
        t0 = times[0]
        seconds = np.array([(t - t0).total_seconds() for t in times])
        
        filtered_pos = np.zeros_like(positions)
        filtered_vel = np.zeros_like(velocities)
        
        for i in range(3):  # X, Y, Z
            pos_coef = np.polyfit(seconds, positions[:,i], 4)
            filtered_pos[:,i] = np.polyval(pos_coef, seconds)
            
            vel_coef = np.polyfit(seconds, velocities[:,i], 4)
            filtered_vel[:,i] = np.polyval(vel_coef, seconds)
        
        return filtered_pos, filtered_vel

    def physics_constrained_filter(self, time_data, positions, velocities, img_start_sec=None, img_stop_sec=None):
        """Filter based on physics constraints
        
        Args:
        time_data: time sequence (seconds)
        positions: position data (N x 3 numpy array)
        velocities: velocity data (N x 3 numpy array)
        img_start_sec: imaging start time (seconds)
        img_stop_sec: imaging stop time (seconds)
        """
        # Detect and handle time discontinuities
        time_diffs = np.diff(time_data)

        large_gaps = np.where(time_diffs > 2.0)[0]

        full_time = np.arange(time_data[0], time_data[-1] + 1, 1.0)

        interp_positions = np.zeros((len(full_time), 3))
        interp_velocities = np.zeros((len(full_time), 3))

        original_data_mask = np.zeros(len(full_time), dtype=bool)

        for i, t in enumerate(time_data):
            idx = int(round(t - full_time[0]))
            if 0 <= idx < len(full_time):
                original_data_mask[idx] = True
                interp_positions[idx] = positions[i]
                interp_velocities[idx] = velocities[i]
        
        # Interpolate missing points
        for i in range(3):
            # Interpolate positions
            valid_pos = original_data_mask
            if np.sum(valid_pos) > 1:  
                interp_positions[:, i] = np.interp(full_time, 
                                                full_time[valid_pos],
                                                interp_positions[valid_pos, i])  
            # Interpolate velocities
            valid_vel = original_data_mask
            if np.sum(valid_vel) > 1:
                interp_velocities[:, i] = np.interp(full_time,
                                                full_time[valid_vel],
                                                interp_velocities[valid_vel, i])
        
        # Fit positions with polynomial segments
        fitted_positions = np.zeros_like(interp_positions)
        for i in range(3):
            if len(large_gaps) == 0:
                # If there are no large gaps, fit the whole time sequence
                order = min(4, len(full_time)-1)
                coeffs = np.polyfit(full_time, interp_positions[:, i], order)
                fitted_positions[:, i] = np.polyval(coeffs, full_time)
            else:
                # Fit with polynomial segments
                start_idx = 0
                for gap_idx in large_gaps:
                    if gap_idx - start_idx > 5:
                        seg_time = full_time[start_idx:gap_idx+1]
                        seg_pos = interp_positions[start_idx:gap_idx+1, i]
                        order = min(4, len(seg_time)-1)
                        coeffs = np.polyfit(seg_time, seg_pos, order)
                        fitted_positions[start_idx:gap_idx+1, i] = np.polyval(coeffs, seg_time)
                    start_idx = gap_idx + 1
                # Process the last segment
                if len(full_time) - start_idx > 5:
                    seg_time = full_time[start_idx:]
                    seg_pos = interp_positions[start_idx:, i]
                    order = min(4, len(seg_time)-1)
                    coeffs = np.polyfit(seg_time, seg_pos, order)
                    fitted_positions[start_idx:, i] = np.polyval(coeffs, seg_time)
        
        # Calculate theoretical velocities based on positions
        theoretical_velocities = np.zeros_like(interp_velocities)
        
        # Calculate theoretical velocities based on positions
        for i in range(1, len(full_time)-1):
            dt = full_time[i+1] - full_time[i-1]
            theoretical_velocities[i] = (fitted_positions[i+1] - fitted_positions[i-1]) / dt
        
        # Set the theoretical velocities at boundary points to be the same as the interpolated velocities
        theoretical_velocities[0] = interp_velocities[0]
        theoretical_velocities[-1] = interp_velocities[-1]
        
        # Calculate weights
        weights = np.ones(len(full_time))
        
        # Process boundary points
        boundary_mask = np.zeros_like(full_time, dtype=bool)
        boundary_mask[0] = True  # Start point
        boundary_mask[-1] = True  # End point
        if len(large_gaps) > 0:
            # The start and end points of the segments are also considered as boundary points
            for gap_idx in large_gaps:
                if gap_idx > 0:
                    boundary_mask[gap_idx] = True
                if gap_idx + 1 < len(boundary_mask):
                    boundary_mask[gap_idx + 1] = True
        
        # Set the weights of boundary points to a smaller value
        weights[boundary_mask] = 0.01
        
        # Enhance the weights of the imaging period
        if img_start_sec is not None and img_stop_sec is not None:
            img_mask = (full_time >= img_start_sec) & (full_time <= img_stop_sec)
            weights[img_mask] *= 5.0  # Enhance the weights of the imaging period
        
        # Time decay weights
        time_weights = np.exp(-np.abs(full_time - full_time[len(full_time)//2]) / 100)
        weights *= time_weights
        
        # Merge velocities
        filtered_velocities = np.zeros_like(theoretical_velocities)
        for i in range(3):
            # Calculate weighted average
            non_boundary = ~boundary_mask
            filtered_velocities[non_boundary, i] = (
                theoretical_velocities[non_boundary, i] * weights[non_boundary] +
                interp_velocities[non_boundary, i] * (1 - weights[non_boundary])
            )
            
            # Use interpolated velocities at boundary points
            filtered_velocities[boundary_mask, i] = interp_velocities[boundary_mask, i]
        
        # Return the data at the original time points
        output_positions = fitted_positions[original_data_mask]
        output_velocities = filtered_velocities[original_data_mask]
        
        return output_positions, output_velocities

    def combined_weighted_filter(self, time_data, positions, velocities, img_start_sec=None, img_stop_sec=None):
        """Filter based on combined weighted method
        
        Args:
        time_data: time sequence (seconds)
        positions: position data (N x 3 numpy array)
        velocities: velocity data (N x 3 numpy array)
        img_start_sec: imaging start time (seconds)
        img_stop_sec: imaging stop time (seconds)
        """
        # Directly specify fixed weights
        weights_img = {'original': 0.222, 'fitted': 0.522, 'theory': 0.256}
        weights_non_img = {'original': 0.222, 'fitted': 0.522, 'theory': 0.256}
        # Special weights for boundary points: don't use theoretical velocity
        weights_boundary = {'original': 0.3, 'fitted': 0.7, 'theory': 0.0}
        
        # 1. Detect and handle time discontinuities
        time_diffs = np.diff(time_data)
        
        large_gaps = np.where(time_diffs > 2.0)[0]
        
        full_time = np.arange(time_data[0], time_data[-1] + 1, 1.0)
        
        interp_positions = np.zeros((len(full_time), 3))
        interp_velocities = np.zeros((len(full_time), 3))
        
        original_data_mask = np.zeros(len(full_time), dtype=bool)
        
        for i, t in enumerate(time_data):
            idx = int(round(t - full_time[0]))
            if 0 <= idx < len(full_time):
                original_data_mask[idx] = True
                interp_positions[idx] = positions[i]
                interp_velocities[idx] = velocities[i]
        
        # Interpolate missing points
        for i in range(3):
            # Interpolate positions
            valid_pos = original_data_mask
            if np.sum(valid_pos) > 1:  # Ensure enough points for interpolation
                interp_positions[:, i] = np.interp(full_time, 
                                                 full_time[valid_pos],
                                                 interp_positions[valid_pos, i])
            
            # Interpolate velocities
            valid_vel = original_data_mask
            if np.sum(valid_vel) > 1:
                interp_velocities[:, i] = np.interp(full_time,
                                                  full_time[valid_vel],
                                                  interp_velocities[valid_vel, i])
        
        # 2. Polynomial fitting for velocities, using a fixed 4th order polynomial
        fitted_velocities = np.zeros_like(interp_velocities)
        for i in range(3):
            if len(large_gaps) == 0:
                # If no large gaps, fit the entire dataset
                try:
                    coeffs = np.polyfit(full_time, interp_velocities[:, i], 4)
                    fitted_velocities[:, i] = np.polyval(coeffs, full_time)
                except:
                    # If fitting fails, use interpolated results
                    fitted_velocities[:, i] = interp_velocities[:, i]
            else:
                # Fit with segments when there are large gaps
                start_idx = 0
                for gap_idx in large_gaps:
                    if gap_idx - start_idx > 4:  # Ensure segment is long enough
                        seg_time = full_time[start_idx:gap_idx+1]
                        seg_vel = interp_velocities[start_idx:gap_idx+1, i]
                        try:
                            coeffs = np.polyfit(seg_time, seg_vel, 4)
                            fitted_velocities[start_idx:gap_idx+1, i] = np.polyval(coeffs, seg_time)
                        except:
                            # If fitting fails, use interpolated results
                            fitted_velocities[start_idx:gap_idx+1, i] = seg_vel
                    else:
                        # Segment too short, use interpolated results
                        fitted_velocities[start_idx:gap_idx+1, i] = interp_velocities[start_idx:gap_idx+1, i]
                    start_idx = gap_idx + 1
                
                if len(full_time) - start_idx > 4:
                    seg_time = full_time[start_idx:]
                    seg_vel = interp_velocities[start_idx:, i]
                    try:
                        coeffs = np.polyfit(seg_time, seg_vel, 4)
                        fitted_velocities[start_idx:, i] = np.polyval(coeffs, seg_time)
                    except:
                        # If fitting fails, use interpolated results
                        fitted_velocities[start_idx:, i] = seg_vel
                else:
                    # Segment too short, use interpolated results
                    fitted_velocities[start_idx:, i] = interp_velocities[start_idx:, i]
        
        # 3. Polynomial fitting for positions, using a fixed 4th order polynomial
        fitted_positions = np.zeros_like(interp_positions)
        for i in range(3):
            # Check if segmentation is needed
            if len(large_gaps) == 0:
                # If no large gaps, fit the entire dataset
                try:
                    coeffs = np.polyfit(full_time, interp_positions[:, i], 4)
                    fitted_positions[:, i] = np.polyval(coeffs, full_time)
                except:
                    # If fitting fails, use interpolated results
                    fitted_positions[:, i] = interp_positions[:, i]
            else:
                # Fit with segments when there are large gaps
                start_idx = 0
                for gap_idx in large_gaps:
                    if gap_idx - start_idx > 4:  # Ensure segment is long enough
                        seg_time = full_time[start_idx:gap_idx+1]
                        seg_pos = interp_positions[start_idx:gap_idx+1, i]
                        try:
                            coeffs = np.polyfit(seg_time, seg_pos, 4)
                            fitted_positions[start_idx:gap_idx+1, i] = np.polyval(coeffs, seg_time)
                        except:
                            # If fitting fails, use interpolated results
                            fitted_positions[start_idx:gap_idx+1, i] = seg_pos
                    else:
                        # Segment too short, use interpolated results
                        fitted_positions[start_idx:gap_idx+1, i] = interp_positions[start_idx:gap_idx+1, i]
                    start_idx = gap_idx + 1
                
                if len(full_time) - start_idx > 4:
                    seg_time = full_time[start_idx:]
                    seg_pos = interp_positions[start_idx:, i]
                    try:
                        coeffs = np.polyfit(seg_time, seg_pos, 4)
                        fitted_positions[start_idx:, i] = np.polyval(coeffs, seg_time)
                    except:
                        # If fitting fails, use interpolated results
                        fitted_positions[start_idx:, i] = seg_pos
                else:
                    # Segment too short, use interpolated results
                    fitted_positions[start_idx:, i] = interp_positions[start_idx:, i]
        
        # 4. Calculate theoretical velocities based on positions
        theoretical_velocities = np.zeros_like(fitted_velocities)
        
        # Use central difference to calculate theoretical velocities (non-boundary points)
        for i in range(1, len(full_time)-1):
            dt = full_time[i+1] - full_time[i-1]
            theoretical_velocities[i] = (fitted_positions[i+1] - fitted_positions[i-1]) / dt
        
        # Set theoretical velocities at boundary points to be the same as fitted velocities
        theoretical_velocities[0] = fitted_velocities[0]
        theoretical_velocities[-1] = fitted_velocities[-1]
        
        # 5. Special processing for the imaging time period
        if img_start_sec is not None and img_stop_sec is not None:
            img_mask = (full_time >= img_start_sec) & (full_time <= img_stop_sec)
            if np.any(img_mask):
                # For non-original data points in the imaging period, use fitted results
                img_non_original = np.where(img_mask & ~original_data_mask)[0]
                if len(img_non_original) > 0:
                    theoretical_velocities[img_non_original] = fitted_velocities[img_non_original]
        
        # 6. Merge fitted velocities and theoretical velocities
        filtered_velocities = np.zeros_like(theoretical_velocities)
        for i in range(3):
            if img_start_sec is not None and img_stop_sec is not None:
                img_mask = (full_time >= img_start_sec) & (full_time <= img_stop_sec)
                
                # Create boundary points mask
                boundary_mask = np.zeros_like(full_time, dtype=bool)
                boundary_mask[0] = True  # Start point
                boundary_mask[-1] = True  # End point
                if len(large_gaps) > 0:
                    # The start and end points of segments are also considered boundary points
                    for gap_idx in large_gaps:
                        if gap_idx > 0:
                            boundary_mask[gap_idx] = True
                        if gap_idx + 1 < len(boundary_mask):
                            boundary_mask[gap_idx + 1] = True
                
                # Use img weights for imaging period
                non_boundary_img = img_mask & ~boundary_mask
                filtered_velocities[non_boundary_img, i] = (
                    interp_velocities[non_boundary_img, i] * weights_img['original'] +
                    fitted_velocities[non_boundary_img, i] * weights_img['fitted'] +
                    theoretical_velocities[non_boundary_img, i] * weights_img['theory']
                )
                
                # Use non_img weights for non-imaging period
                non_boundary_non_img = ~img_mask & ~boundary_mask
                filtered_velocities[non_boundary_non_img, i] = (
                    interp_velocities[non_boundary_non_img, i] * weights_non_img['original'] +
                    fitted_velocities[non_boundary_non_img, i] * weights_non_img['fitted'] +
                    theoretical_velocities[non_boundary_non_img, i] * weights_non_img['theory']
                )
                
                # Use special weights for boundary points
                filtered_velocities[boundary_mask, i] = (
                    interp_velocities[boundary_mask, i] * weights_boundary['original'] +
                    fitted_velocities[boundary_mask, i] * weights_boundary['fitted'] +
                    theoretical_velocities[boundary_mask, i] * weights_boundary['theory']
                )
            else:
                # If no imaging time information, use non_img weights for non-boundary points
                boundary_mask = np.zeros_like(full_time, dtype=bool)
                boundary_mask[0] = True
                boundary_mask[-1] = True
                if len(large_gaps) > 0:
                    for gap_idx in large_gaps:
                        if gap_idx > 0:
                            boundary_mask[gap_idx] = True
                        if gap_idx + 1 < len(boundary_mask):
                            boundary_mask[gap_idx + 1] = True
                
                # Non-boundary points
                non_boundary = ~boundary_mask
                filtered_velocities[non_boundary, i] = (
                    interp_velocities[non_boundary, i] * weights_non_img['original'] +
                    fitted_velocities[non_boundary, i] * weights_non_img['fitted'] +
                    theoretical_velocities[non_boundary, i] * weights_non_img['theory']
                )
                
                # Boundary points
                filtered_velocities[boundary_mask, i] = (
                    interp_velocities[boundary_mask, i] * weights_boundary['original'] +
                    fitted_velocities[boundary_mask, i] * weights_boundary['fitted'] +
                    theoretical_velocities[boundary_mask, i] * weights_boundary['theory']
                )
        
        # Return the data at the original time points
        output_positions = fitted_positions[original_data_mask]
        output_velocities = filtered_velocities[original_data_mask]
        
        return output_positions, output_velocities

    def createOrbit(self):
        """
        Create orbit from multiple frames using filtering techniques.
        
        This method collects state vectors from all frames, filters them 
        and creates a consistent orbit for the entire data stack.
        """
        from isceobj.Orbit.Orbit import Orbit, StateVector
        
        # Check if we have a frameList to work with
        if not hasattr(self, 'frameList') or not self.frameList:
            self.logger.warning("No frameList found, extracting orbit from annotation file")
            return self.extractOrbitFromAnnotation()
        
        # Create new orbit object
        orbit = Orbit()
        orbit.configure()
        
        # Collect orbit state vectors from all frames
        all_vectors = []
        for frame in self.frameList:
            if frame.orbit and frame.orbit._stateVectors:
                all_vectors.extend(frame.orbit._stateVectors)
        
        # If no state vectors found in frameList, try to extract from annotation
        if not all_vectors:
            self.logger.warning("No orbit state vectors found in frameList, extracting from annotation")
            return self.extractOrbitFromAnnotation()
            
        # Sort by time
        all_vectors.sort(key=lambda x: x.getTime())
        
        # Remove duplicate state vectors
        unique_vectors = []
        prev_time = None
        for sv in all_vectors:
            curr_time = sv.getTime()
            if prev_time is None or curr_time != prev_time:
                unique_vectors.append(sv)
                prev_time = curr_time
        
        # If still not enough vectors, use annotation file
        if len(unique_vectors) < 4:
            self.logger.warning(f"Only found {len(unique_vectors)} unique state vectors. Minimum required is 4.")
            return self.extractOrbitFromAnnotation()
        
        # Extract timestamps, positions, and velocities
        timestamps = []
        positions = []
        velocities = []
        
        for sv in unique_vectors:
            timestamps.append(sv.getTime())
            positions.append(sv.getPosition())
            velocities.append(sv.getVelocity())
        
        # Convert to numpy arrays for filtering
        positions = np.array(positions)
        velocities = np.array(velocities)
        
        # Determine sensing start and stop times by finding min/max from all frames
        sensing_start = min([frame.getSensingStart() for frame in self.frameList])
        sensing_stop = max([frame.getSensingStop() for frame in self.frameList])
        
        # Calculate relative time in seconds for filtering
        t0 = timestamps[0]
        time_seconds = np.array([(t - t0).total_seconds() for t in timestamps])
        
        # Get imaging time in seconds relative to t0
        img_start_sec = (sensing_start - t0).total_seconds()
        img_stop_sec = (sensing_stop - t0).total_seconds()
        
        # Apply orbit filtering based on selected method
        try:
            if self.filterMethod == 'poly_filter':
                self.logger.info("Applying polynomial filter to orbit data")
                filtered_pos, filtered_vel = self.poly_filter(timestamps, positions, velocities)
            elif self.filterMethod == 'physics_filter':
                self.logger.info("Applying physics-constrained filter to orbit data")
                filtered_pos, filtered_vel = self.physics_constrained_filter(
                    time_seconds, positions, velocities,
                    img_start_sec, img_stop_sec
                )
            else:  # default to combined_weighted_filter
                self.logger.info("Applying combined weighted filter to orbit data")
                filtered_pos, filtered_vel = self.combined_weighted_filter(
                    time_seconds, positions, velocities,
                    img_start_sec, img_stop_sec
                )
        except Exception as e:
            self.logger.error(f"Error during orbit filtering: {str(e)}")
            self.logger.warning("Falling back to annotation file orbit")
            return self.extractOrbitFromAnnotation()
        
        # Create orbit state vectors
        for i, timestamp in enumerate(timestamps):
            vec = StateVector()
            vec.setTime(timestamp)
            vec.setPosition(filtered_pos[i].tolist())
            vec.setVelocity(filtered_vel[i].tolist())
            orbit.addStateVector(vec)
        
        self.logger.info(f"Created orbit with {len(timestamps)} state vectors")
        self.logger.info(f"Orbit time range: {timestamps[0]} to {timestamps[-1]}")
        
        return orbit

    def extractImage(self):
        """提取图像数据"""
        # 验证用户输入
        self.validateUserInputs()
        
        # 确保_xmlFileList和_imageFileList被正确设置
        if not self._xmlFileList:
            self._xmlFileList = [tiff[:-4] + "meta.xml" for tiff in self._tiffList]
            self.logger.info(f"Generated XML file list: {self._xmlFileList}")
        
        if not self._imageFileList:
            self._imageFileList = self._tiffList
            self.logger.info(f"Using TIFF file list as image file list: {self._imageFileList}")
        
        # 检查文件是否存在
        for xml_file, tiff_file in zip(self._xmlFileList, self._imageFileList):
            if not os.path.exists(xml_file):
                self.logger.error(f"XML file not found: {xml_file}")
                raise RuntimeError(f"XML file not found: {xml_file}")
            if not os.path.exists(tiff_file):
                self.logger.error(f"TIFF file not found: {tiff_file}")
                raise RuntimeError(f"TIFF file not found: {tiff_file}")
        
        # 清空frameList
        self.frameList = []
        
        # 处理每个frame
        for i, (xml_file, tiff_file) in enumerate(zip(self._xmlFileList, self._imageFileList)):
            self.logger.info(f"Processing frame {i+1}/{len(self._xmlFileList)}")
            self.logger.info(f"XML file: {xml_file}")
            self.logger.info(f"TIFF file: {tiff_file}")
            
            # 创建新的frame
            frame = Frame()
            frame.configure()
            
            # 设置当前frame和文件
            self.frame = frame
            self._tiff = tiff_file
            if self._orbitFileList and i < len(self._orbitFileList):
                self._orbitFile = self._orbitFileList[i]
            else:
                self._orbitFile = None
            
            try:
                # 解析元数据和轨道信息
                self.parse()
                
                # 提取图像数据
                outputNow = self.output + "_" + str(i) if len(self._xmlFileList) > 1 else self.output
                self.extractFrameImage(tiff_file, outputNow)
                
                # 添加到frameList
                self.frameList.append(frame)
                self.logger.info(f"Successfully processed frame {i+1}")
                
            except Exception as e:
                self.logger.error(f"Error processing frame {i+1}: {str(e)}")
                raise
        
        # 确保frameList不为空
        if not self.frameList:
            raise RuntimeError("No frames were processed")
        
        self.logger.info(f"Successfully processed {len(self.frameList)} frames")
        
        # 使用tkfunc处理多frame
        merged_frame = tkfunc(self)
        
        # 处理轨道信息
        if len(self.frameList) > 1:
            # 如果有多个帧，合并轨道
            merged_orbit = self.mergeOrbits([frame.orbit for frame in self.frameList])
            if merged_orbit:
                self.logger.info("Successfully merged orbits from all frames")
        else:
            # 如果只有一个帧，直接使用其轨道
            merged_orbit = self.frameList[0].orbit if self.frameList else None
            
        # 确保frame和frameList被正确设置
        if merged_frame:
            self.frame = merged_frame
            if merged_orbit:
                self.frame.orbit = merged_orbit
            self.frameList = [merged_frame]
            
            # 确保图像被正确设置
            if not self.frame.image:
                # 创建SLC图像对象
                slcImage = isceobj.createSlcImage()
                slcImage.setByteOrder('l')
                slcImage.setFilename(self.output)
                slcImage.setAccessMode('read')
                slcImage.setWidth(self.frame.getNumberOfSamples())
                slcImage.setLength(self.frame.getNumberOfLines())
                slcImage.setXmin(0)
                slcImage.setXmax(self.frame.getNumberOfSamples())
                slcImage.setDataType('CFLOAT')
                slcImage.setImageType('slc')
                
                # 设置图像
                self.frame.setImage(slcImage)
                
                # 生成头文件和VRT文件
                slcImage.renderHdr()
                slcImage.renderVRT()
                
                # 保存轨道信息到单独文件
                orbit_file = self.output + '.orb'
                self.saveOrbitToFile(self.frame.orbit, orbit_file)
        else:
            # 如果没有合并的frame，使用第一个frame并设置合并的轨道
            self.frame = self.frameList[0]
            self.frameList = [self.frame]
            if len(self.frameList) > 1 and merged_orbit:
                self.frame.setOrbit(merged_orbit)
                # 保存合并后的轨道信息
                orbit_file = self.output + '.orb'
                self.saveOrbitToFile(merged_orbit, orbit_file)
        
        return self.frame

    def extractFrameImage(self, tiff_file, output):
        """
        提取单个帧的图像数据，正确处理复数SLC数据
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise Exception('GDAL python bindings not found.')
        
        # 打开TIFF文件
        src = gdal.Open(tiff_file.strip(), gdal.GA_ReadOnly)
        if src is None:
            raise Exception(f"Failed to open TIFF file: {tiff_file}")
        
        try:
            # 获取图像信息
            width = self.frame.getNumberOfSamples()
            length = self.frame.getNumberOfLines()
            
            # 确保TIFF文件有两个波段（实部和虚部）
            if src.RasterCount != 2:
                raise Exception(f"Expected 2 bands for complex data, found {src.RasterCount} bands")
            
            # 获取实部和虚部波段
            band1 = src.GetRasterBand(1)  # 实部
            band2 = src.GetRasterBand(2)  # 虚部
            
            self.logger.info(f"Reading complex data from {tiff_file}")
            self.logger.info(f"Image dimensions: {width} x {length}")
            
            # 一次性读取所有数据
            real = band1.ReadAsArray(0, 0, width, length).astype(np.float32)
            imag = band2.ReadAsArray(0, 0, width, length).astype(np.float32)
            
            # 验证数据形状
            if real.shape != (length, width) or imag.shape != (length, width):
                raise ValueError(f"数据形状不匹配: real {real.shape}, imag {imag.shape}, 期望 ({length}, {width})")
            
            # 创建复数数组
            data = np.empty((length, width), dtype=np.complex64)
            data.real = real
            data.imag = imag
            
            # 检查数据有效性
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                self.logger.warning("检测到无效值（NaN或Inf）在复数数据中")
            
            # 记录数据统计信息
            self.logger.info(f"数据统计: 最小幅度={np.min(np.abs(data)):.2f}, "
                            f"最大幅度={np.max(np.abs(data)):.2f}, "
                            f"均值幅度={np.mean(np.abs(data)):.2f}")
            
            # 写入SLC文件
            self.logger.info(f"Writing complex data to {output}")
            with open(output, 'wb') as fid:
                data.tofile(fid)
            
            # 验证输出文件大小
            expected_size = length * width * 8  # complex64 = 8 bytes
            actual_size = os.path.getsize(output)
            if actual_size != expected_size:
                raise ValueError(f"输出文件大小不匹配: 实际={actual_size}, 期望={expected_size}")
            
            # 创建SLC图像对象
            slcImage = isceobj.createSlcImage()
            slcImage.setByteOrder('l')
            slcImage.setFilename(output)
            slcImage.setAccessMode('read')
            slcImage.setWidth(width)
            slcImage.setLength(length)
            slcImage.setXmin(0)
            slcImage.setXmax(width)
            slcImage.setDataType('CFLOAT')
            slcImage.setImageType('slc')
            
            # 设置图像
            self.frame.setImage(slcImage)
            
            # 生成头文件和VRT文件
            slcImage.renderHdr()
            slcImage.renderVRT()
            
            # 清理资源
            del real, imag, data
            src = None
            band1 = None
            band2 = None
            
            self.logger.info(f"Successfully extracted frame image to {output}")
            
        except Exception as e:
            self.logger.error(f"Error extracting frame image: {str(e)}")
            raise
        finally:
            # 确保GDAL资源被释放
            if src is not None:
                src = None


    def mergeFrames(self):
        """
        合并多个帧，同时处理frame和slc数据
        """
        if not self.frameList:
            return None
        
        # 按时间排序帧
        sorted_frames = sorted(self.frameList, key=lambda x: x.getSensingStart())
        
        # 计算总行数和处理重叠
        total_lines = 0
        overlap_info = []  # 存储重叠信息
        
        for i, frame in enumerate(sorted_frames):
            if i > 0:
                prev_frame = sorted_frames[i-1]
                overlap = (prev_frame.getSensingStop() - frame.getSensingStart()).total_seconds()
                if overlap > 0:
                    overlap_lines = int(overlap * frame.getInstrument().getPulseRepetitionFrequency())
                    overlap_lines = min(overlap_lines, frame.getNumberOfLines())
                    total_lines += frame.getNumberOfLines() - overlap_lines
                    overlap_info.append((i, overlap_lines))
                    self.logger.info(f"Frame {i} overlaps with frame {i-1} by {overlap_lines} lines")
                else:
                    total_lines += frame.getNumberOfLines()
                    overlap_info.append((i, 0))
            else:
                total_lines += frame.getNumberOfLines()
                overlap_info.append((i, 0))
        
        # 创建合并后的frame
        merged_frame = Frame()
        merged_frame.configure()
        
        # 设置基本属性
        merged_frame.setSensingStart(sorted_frames[0].getSensingStart())
        merged_frame.setSensingStop(sorted_frames[-1].getSensingStop())
        merged_frame.setStartingRange(sorted_frames[0].getStartingRange())
        merged_frame.setFarRange(sorted_frames[-1].getFarRange())
        merged_frame.setNumberOfLines(total_lines)
        merged_frame.setNumberOfSamples(sorted_frames[0].getNumberOfSamples())
        
        # 复制第一帧的仪器参数
        merged_frame.setInstrument(sorted_frames[0].getInstrument().copy())
        
        # 合并轨道信息
        merged_orbit = self.mergeOrbits([frame.orbit for frame in sorted_frames])
        merged_frame.setOrbit(merged_orbit)
        
        # 处理SLC数据
        width = sorted_frames[0].getNumberOfSamples()
        merged_data = np.zeros((total_lines, width), dtype=np.complex64)
        current_line = 0
        
        for i, frame in enumerate(sorted_frames):
            # 读取当前帧数据
            with open(frame.image.getFilename(), 'rb') as f:
                frame_data = np.fromfile(f, dtype=np.complex64).reshape(frame.getNumberOfLines(), width)
            
            if i == 0:
                # 第一帧直接复制
                merged_data[current_line:current_line+frame.getNumberOfLines()] = frame_data
                current_line += frame.getNumberOfLines()
            else:
                # 处理重叠区域
                overlap_lines = overlap_info[i][1]
                if overlap_lines > 0:
                    # 计算非重叠部分
                    non_overlap_data = frame_data[overlap_lines:]
                    non_overlap_lines = len(non_overlap_data)
                    
                    # 复制非重叠部分
                    merged_data[current_line:current_line+non_overlap_lines] = non_overlap_data
                    current_line += non_overlap_lines
                else:
                    # 无重叠,直接复制
                    merged_data[current_line:current_line+frame.getNumberOfLines()] = frame_data
                    current_line += frame.getNumberOfLines()
        
        # 写入合并后的SLC数据
        output_file = self.output
        with open(output_file, 'wb') as f:
            merged_data.tofile(f)
        
        # 设置合并后的图像
        slcImage = isceobj.createSlcImage()
        slcImage.setByteOrder('l')
        slcImage.setFilename(output_file)
        slcImage.setAccessMode('read')
        slcImage.setWidth(width)
        slcImage.setLength(total_lines)
        slcImage.setXmin(0)
        slcImage.setXmax(width)
        slcImage.setDataType('CFLOAT')
        slcImage.setImageType('slc')
        
        merged_frame.setImage(slcImage)
        
        # 生成头文件和VRT文件
        slcImage.renderHdr()
        slcImage.renderVRT()
        
        # 生成辅助文件
        self.makeFakeAux(output_file)
        
        return merged_frame

    def mergeOrbits(self, orbits):
        """合并多个轨道的状态向量"""
        from isceobj.Orbit.Orbit import Orbit, StateVector
        
        merged_orbit = Orbit()
        merged_orbit.configure()
        
        # 收集所有状态向量
        all_vectors = []
        for orbit in orbits:
            if orbit and orbit._stateVectors:
                # 转换为列表后再扩展
                all_vectors.extend(list(orbit._stateVectors))
        
        if not all_vectors:
            self.logger.warning("No orbit state vectors found in any frames")
            return None
        
        # 按时间排序
        all_vectors.sort(key=lambda x: x.getTime())
        
        # 移除重复的状态向量
        unique_vectors = []
        prev_time = None
        for sv in all_vectors:
            curr_time = sv.getTime()
            if prev_time is None or curr_time != prev_time:
                unique_vectors.append(sv)
                prev_time = curr_time
        
        # 添加到合并后的轨道
        for sv in unique_vectors:
            merged_orbit.addStateVector(sv)
        
        return merged_orbit

    def extractDoppler(self):
        '''
        Set doppler values to zero since the metadata doppler values are unreliable.
        Also, the SLC images are zero doppler.
        '''
        dop = [0., 0., 0.]

        ####For insarApp
        quadratic = {}
        quadratic['a'] = dop[0] / self.frame.getInstrument().getPulseRepetitionFrequency()
        quadratic['b'] = 0.
        quadratic['c'] = 0.

        print("Average doppler: ", dop)
        self.frame._dopplerVsPixel = dop

        return quadratic

    def makeFakeAux(self, outputNow):
        '''
        Generate an aux file based on sensing start and prf.
        '''
        import math, array

        prf = self.frame.getInstrument().getPulseRepetitionFrequency()
        senStart = self.frame.getSensingStart()
        numPulses = self.frame.numberOfLines
        # the aux files has two entries per line. day of the year and microseconds in the day
        musec0 = (senStart.hour*3600 + senStart.minute*60 + senStart.second)*10**6 + senStart.microsecond
        maxMusec = (24*3600)*10**6  # use it to check if we went across a day. very rare
        day0 = (datetime.datetime(senStart.year,senStart.month,senStart.day) - datetime.datetime(senStart.year,1,1)).days + 1
        outputArray = array.array('d',[0]*2*numPulses)
        self.frame.auxFile = outputNow + '.aux'
        fp = open(self.frame.auxFile,'wb')
        j = -1
        for i1 in range(numPulses):
            j += 1
            musec = round((j/prf)*10**6) + musec0
            if musec >= maxMusec:
                day0 += 1
                musec0 = musec%maxMusec
                musec = musec0
                j = 0
            outputArray[2*i1] = day0
            outputArray[2*i1+1] = musec

        outputArray.tofile(fp)
        fp.close()

    def loadFromXML(self):
        """
        从XML配置文件中读取SLC和轨道文件路径，并自动检索所有文件
        支持多种XML格式
        """
        if self._xmlConfig is None:
            self.logger.error("XML configuration file not provided")
            return False
            
        try:
            # 处理可能的zip文件路径
            if self._xmlConfig.startswith('/vsizip'):
                import zipfile
                parts = self._xmlConfig.split(os.path.sep)
                if parts[2] == '':
                    parts[2] = os.path.sep
                zipname = os.path.join(*(parts[2:-1]))
                fname = parts[-1]
                
                zf = zipfile.ZipFile(zipname, 'r')
                xmlstr = zf.read(fname)
                zf.close()
                root = ET.fromstring(xmlstr)
            else:
                tree = ET.parse(self._xmlConfig)
                root = tree.getroot()
            
            # 尝试不同的XML格式
            # 格式1: <property name="SLC_DIR"><value>./SLC</value></property>
            slc_dir_element = root.find(".//property[@name='SLC_DIR']/value")
            if slc_dir_element is not None:
                slc_dir = slc_dir_element.text
                # 检索所有TIFF文件
                self._imageFileList = sorted(glob.glob(os.path.join(slc_dir, "*.tiff")))
                if not self._imageFileList:
                    self.logger.warning(f"No TIFF files found in {slc_dir}")
            else:
                # 格式2: <tiff>./SLC/file.tiff</tiff>
                tiff_element = root.find(".//tiff")
                if tiff_element is not None:
                    self._imageFileList = [tiff_element.text]
                else:
                    self.logger.error("No SLC_DIR or tiff element found in XML configuration")
                    return False
                
            # 获取轨道文件目录（可选）
            orbit_dir_element = root.find(".//property[@name='ORBIT_DIR']/value")
            if orbit_dir_element is not None:
                orbit_dir = orbit_dir_element.text
                # 检索所有XML轨道文件
                self._xmlFileList = sorted(glob.glob(os.path.join(orbit_dir, "*.xml")))
                if not self._xmlFileList:
                    self.logger.warning(f"No orbit XML files found in {orbit_dir}")
                    self._xmlFileList = []
            else:
                # 格式2: <orbitFile>./orbits/file.xml</orbitFile> 或 <ORBITFILE>./orbits/file.xml</ORBITFILE>
                # 尝试不同大小写的轨道文件标签
                orbit_element = root.find(".//orbitFile") or root.find(".//ORBITFILE")
                
                # 也尝试 property 格式的轨道文件
                if orbit_element is None:
                    orbit_property = root.find(".//property[@name='orbitFile']/value") or root.find(".//property[@name='ORBITFILE']/value")
                    if orbit_element is not None:
                        orbit_element = orbit_property
                
                if orbit_element is not None and orbit_element.text:
                    self._xmlFileList = [orbit_element.text]
                    self.logger.info(f"Using orbit file: {orbit_element.text}")
                else:
                    self.logger.warning("No ORBIT_DIR or orbitFile element found in XML configuration, proceeding without orbit files")
                    self._xmlFileList = []
                
            # 获取输出文件名
            output_element = root.find(".//property[@name='OUTPUT']/value")
            if output_element is not None:
                self.output = output_element.text
            else:
                # 格式2: <OUTPUT>reference</OUTPUT>
                output_element = root.find(".//OUTPUT")
                if output_element is not None:
                    self.output = output_element.text
                else:
                    self.logger.error("OUTPUT not found in XML configuration")
                    return False
                
            # 打印找到的文件信息
            self.logger.info(f"Found {len(self._imageFileList)} TIFF files and {len(self._xmlFileList)} orbit files")
            for i, tiff in enumerate(self._imageFileList):
                self.logger.info(f"TIFF {i+1}: {tiff}")
            for i, orbit in enumerate(self._xmlFileList):
                self.logger.info(f"Orbit {i+1}: {orbit}")
                
            # 如果轨道文件数量与TIFF文件数量不匹配，尝试自动匹配
            if len(self._xmlFileList) > 0 and len(self._xmlFileList) != len(self._imageFileList):
                self.logger.warning("Number of orbit files does not match number of TIFF files")
                self.logger.info("Attempting to match orbit files with TIFF files...")
                
                # 如果只有一个轨道文件，假设它适用于所有TIFF文件
                if len(self._xmlFileList) == 1:
                    self.logger.info("Using the single orbit file for all TIFF files")
                    self._xmlFileList = [self._xmlFileList[0]] * len(self._imageFileList)
                else:
                    # 尝试根据日期匹配轨道文件
                    try:
                        # 提取每个TIFF文件名中的日期信息
                        tiff_dates = []
                        for tiff in self._imageFileList:
                            # 假设文件名格式包含日期，例如：LT1B_MONO_KRN_STRIP2_000643_E37.4_N37.1_20220411_SLC_HH_L1A_0000010037.tiff
                            basename = os.path.basename(tiff)
                            parts = basename.split('_')
                            if len(parts) >= 8:
                                date_str = parts[7]  # 假设日期在第8个位置
                                if len(date_str) == 8 and date_str.isdigit():  # 确保是8位数字（YYYYMMDD）
                                    tiff_dates.append(date_str)
                                else:
                                    tiff_dates.append(None)
                            else:
                                tiff_dates.append(None)
                        
                        # 提取每个轨道文件名中的日期信息
                        orbit_dates = []
                        for orbit in self._xmlFileList:
                            # 假设轨道文件名格式包含日期，例如：LT1B_20230607102726409_V20220410T235500_20220412T000500_ABSORBIT_SCIE.xml
                            basename = os.path.basename(orbit)
                            parts = basename.split('_')
                            if len(parts) >= 4:
                                date_str = parts[2][1:9]  # 假设日期在第3个位置，格式为VYYYYMMDD...
                                if len(date_str) == 8 and date_str.isdigit():  # 确保是8位数字（YYYYMMDD）
                                    orbit_dates.append(date_str)
                                else:
                                    orbit_dates.append(None)
                            else:
                                orbit_dates.append(None)
                        
                        # 为每个TIFF文件匹配对应的轨道文件
                        matched_orbit_files = []
                        for i, tiff_date in enumerate(tiff_dates):
                            if tiff_date is None:
                                matched_orbit_files.append(None)
                                continue
                            
                            # 查找日期匹配的轨道文件
                            matched = False
                            for j, orbit_date in enumerate(orbit_dates):
                                if orbit_date is not None and orbit_date == tiff_date:
                                    matched_orbit_files.append(self._xmlFileList[j])
                                    matched = True
                                    break
                            
                            if not matched:
                                matched_orbit_files.append(None)
                        
                        # 如果所有TIFF文件都找到了匹配的轨道文件，使用匹配结果
                        if all(orbit is not None for orbit in matched_orbit_files):
                            self.logger.info("Successfully matched orbit files based on date")
                            self._xmlFileList = matched_orbit_files
                        else:
                            self.logger.warning("Could not match all orbit files based on date, proceeding without orbit files")
                            self._xmlFileList = []
                    except Exception as e:
                        self.logger.warning(f"Error matching orbit files: {str(e)}")
                        self.logger.warning("Proceeding without orbit files")
                        self._xmlFileList = []
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading XML configuration: {str(e)}")
            return False

    def saveOrbitToFile(self, orbit, filename):
        """将轨道信息保存到单独的文件中"""
        import pickle
        state_vectors_data = []
        for sv in orbit._stateVectors:
            state_vectors_data.append({
                'time': sv.time,
                'position': sv.position,
                'velocity': sv.velocity
            })
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'state_vectors': state_vectors_data,
                'orbit_quality': orbit._orbitQuality,
                'orbit_source': orbit._orbitSource,
                'reference_frame': orbit._referenceFrame
            }, f)

    def loadOrbitFromFile(self, filename):
        """从文件中恢复轨道信息"""
        import pickle
        from isceobj.Orbit.Orbit import Orbit, StateVector
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        orbit = Orbit()
        orbit.configure()
        
        # 恢复轨道属性
        orbit.setOrbitQuality(data['orbit_quality'])
        orbit.setOrbitSource(data['orbit_source'])
        orbit.setReferenceFrame(data['reference_frame'])
        
        # 恢复状态向量
        for sv_data in data['state_vectors']:
            sv = StateVector()
            sv.setTime(sv_data['time'])
            sv.setPosition(sv_data['position'])
            sv.setVelocity(sv_data['velocity'])
            orbit.addStateVector(sv)
        
        return orbit

    def extractOrbitFromAnnotation(self):
        '''Extract orbit information from xml annotation and apply filtering'''
        try:
            fp = open(self._xml, 'r')
        except IOError as strerr:
            print("IOError: %s" % strerr)
    
        _xml_root = ET.ElementTree(file=fp).getroot()
        node = _xml_root.find('platform/orbit')
        countNode = len(list(_xml_root.find('platform/orbit')))
    
        frameOrbit = Orbit()
        frameOrbit.setOrbitSource('Header')
        margin = datetime.timedelta(minutes=10.0)
        tstart = self.frame.getSensingStart() - margin
        tend = self.frame.getSensingStop() + margin
        
        timestamps = []
        positions = []
        velocities = []
        
        for k in range(1,countNode):
            timestamp = self.convertToDateTime(node.find('stateVec[{}]/timeUTC'.format(k)).text)
            if (timestamp >= tstart) and (timestamp <= tend):
                pos = [float(node.find('stateVec[{}]/posX'.format(k)).text), 
                      float(node.find('stateVec[{}]/posY'.format(k)).text), 
                      float(node.find('stateVec[{}]/posZ'.format(k)).text)]
                vel = [float(node.find('stateVec[{}]/velX'.format(k)).text), 
                      float(node.find('stateVec[{}]/velY'.format(k)).text), 
                      float(node.find('stateVec[{}]/velZ'.format(k)).text)]
                
                timestamps.append(timestamp)
                positions.append(pos)
                velocities.append(vel)
        
        fp.close()

        # 检查是否有足够的数据
        if len(timestamps) < 4:
            self.logger.warning(f"Only {len(timestamps)} state vectors found in annotation file. Minimum required is 4.")
            if hasattr(self, 'frame') and hasattr(self.frame, 'getSensingStart') and hasattr(self.frame, 'getSensingStop'):
                # 尝试扩大时间范围
                margin = datetime.timedelta(minutes=30.0)
                tstart = self.frame.getSensingStart() - margin
                tend = self.frame.getSensingStop() + margin
                
                self.logger.info(f"Extending time range to {tstart} - {tend}")
                
                # 重新收集数据
                timestamps = []
                positions = []
                velocities = []
                
                try:
                    fp = open(self._xml, 'r')
                    _xml_root = ET.ElementTree(file=fp).getroot()
                    node = _xml_root.find('platform/orbit')
                    countNode = len(list(_xml_root.find('platform/orbit')))
                    
                    for k in range(1,countNode):
                        timestamp = self.convertToDateTime(node.find('stateVec[{}]/timeUTC'.format(k)).text)
                        if (timestamp >= tstart) and (timestamp <= tend):
                            pos = [float(node.find('stateVec[{}]/posX'.format(k)).text), 
                                  float(node.find('stateVec[{}]/posY'.format(k)).text), 
                                  float(node.find('stateVec[{}]/posZ'.format(k)).text)]
                            vel = [float(node.find('stateVec[{}]/velX'.format(k)).text), 
                                  float(node.find('stateVec[{}]/velY'.format(k)).text), 
                                  float(node.find('stateVec[{}]/velZ'.format(k)).text)]
                            
                            timestamps.append(timestamp)
                            positions.append(pos)
                            velocities.append(vel)
                    
                    fp.close()
                except Exception as e:
                    self.logger.error(f"Error while extending time range: {str(e)}")
        
        if len(timestamps) < 4:
            self.logger.error(f"Still only found {len(timestamps)} state vectors after extending time range")
            raise RuntimeError("Insufficient orbit state vectors for interpolation (minimum 4 required)")

        positions = np.array(positions)
        velocities = np.array(velocities)
        
        t0 = timestamps[0]
        time_seconds = np.array([(t - t0).total_seconds() for t in timestamps])
        
        img_start_sec = (self.frame.getSensingStart() - t0).total_seconds()
        img_stop_sec = (self.frame.getSensingStop() - t0).total_seconds()
        
        if self.filterMethod == 'poly_filter':
            self.logger.info("Applying polynomial filter to orbit data")
            filtered_pos, filtered_vel = self.poly_filter(timestamps, positions, velocities)
        elif self.filterMethod == 'physics_filter':
            self.logger.info("Applying physics-constrained filter to orbit data")
            filtered_pos, filtered_vel = self.physics_constrained_filter(
                time_seconds, positions, velocities,
                img_start_sec, img_stop_sec
            )
        else:  # default to combined_weighted_filter
            self.logger.info("Applying combined weighted filter to orbit data")
            filtered_pos, filtered_vel = self.combined_weighted_filter(
                time_seconds, positions, velocities,
                img_start_sec, img_stop_sec
            )
        
        # Create orbit state vectors
        for i, timestamp in enumerate(timestamps):
            vec = StateVector()
            vec.setTime(timestamp)
            vec.setPosition(filtered_pos[i].tolist())
            vec.setVelocity(filtered_vel[i].tolist())
            frameOrbit.addStateVector(vec)
        
        return frameOrbit