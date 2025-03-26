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
from isceobj.Orbit.OrbitExtender import OrbitExtender
from osgeo import gdal
import warnings
from scipy.interpolate import UnivariateSpline
from isceobj.Sensor import tkfunc
from isceobj.Image.SlcImage import SlcImage


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


class Lutan1(Sensor):

    "Class for Lutan-1 SLC data"
    
    family = 'l1sm'
    logging_name = 'isce.sensor.Lutan1'

    parameter_list = (TIFF, ORBIT_FILE, XML_CONFIG, SLC_DIR, ORBIT_DIR) + Sensor.parameter_list

    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self.frameList = []
        self._imageFileList = []
        self._xmlFileList = []
        self.frame = None
        self.doppler_coeff = None
        self.filterMethod = 'weighted'
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

        # 初始化 orb 变量，确保在所有情况下都有值
        orb = None

        if self._orbitFile:
            # Check if orbit file exists or not
            if os.path.isfile(self._orbitFile) == True:
                orb = self.extractOrbit()
                if orb is None:
                    self.logger.error("Failed to extract orbit from file")
                    raise RuntimeError("Failed to extract orbit from file")
                self.frame.orbit.setOrbitSource(os.path.basename(self._orbitFile))
            else:
                self.logger.warning(f"Orbit file {self._orbitFile} not found. Using orbit from annotation file.")
                orb = self.createOrbit()
                self.frame.orbit.setOrbitSource(os.path.basename(self._xml))
                self.frame.orbit.setOrbitSource('Annotation')
        else:
            self.logger.warning("No orbit file provided. Using orbit from annotation file.")
            orb = self.createOrbit()
            self.frame.orbit.setOrbitSource(os.path.basename(self._xml))
            self.frame.orbit.setOrbitSource('Annotation')

        # 确保 orb 不为 None
        if orb is None:
            self.logger.error("Failed to extract orbit information.")
            raise RuntimeError("Failed to extract orbit information.")
        
        # 添加状态向量到轨道
        for sv in orb._stateVectors:
            self.frame.orbit.addStateVector(sv)
        
        # 设置轨道的时间范围
        if orb._stateVectors:
            self.frame.orbit.setTimeRange(orb._stateVectors[0].getTime(), orb._stateVectors[-1].getTime())
            self.logger.info(f"Set orbit time range: {orb._stateVectors[0].getTime()} to {orb._stateVectors[-1].getTime()}")
            self.logger.info(f"Frame sensing time range: {self.frame.getSensingStart()} to {self.frame.getSensingStop()}")
        else:
            self.logger.error("No state vectors in orbit")
            raise RuntimeError("No state vectors in orbit")

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
            
            # 设置轨道的时间范围
            orb.setTimeRange(unique_vectors[0].getTime(), unique_vectors[-1].getTime())
            
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
                
                # 设置轨道的时间范围
                orb.setTimeRange(unique_vectors[0].getTime(), unique_vectors[-1].getTime())
                
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
            
            vel_coef = np.polyfit(seconds, velocities[:,i], 5)
            filtered_vel[:,i] = np.polyval(vel_coef, seconds)
        
        return filtered_pos, filtered_vel

    def filter_orbit_sliding(self, times, positions, velocities, window_size=20):
        """使用滑动窗口的多项式拟合"""
        t0 = times[0]
        seconds = np.array([(t - t0).total_seconds() for t in times])
        
        filtered_pos = np.zeros_like(positions)
        filtered_vel = np.zeros_like(velocities)
        
        for i in range(len(times)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(times), i + window_size//2)
            
            window_seconds = seconds[start_idx:end_idx]
            window_pos = positions[start_idx:end_idx]
            window_vel = velocities[start_idx:end_idx]
            
            for j in range(3):
                pos_coef = np.polyfit(window_seconds - seconds[i], window_pos[:,j], 4)
                filtered_pos[i,j] = np.polyval(pos_coef, 0)
                
                vel_coef = np.polyfit(window_seconds - seconds[i], window_vel[:,j], 5)
                filtered_vel[i,j] = np.polyval(vel_coef, 0)
                
        return filtered_pos, filtered_vel

    def filter_orbit_spline(self, times, positions, velocities, k=4):
        """使用样条拟合轨道数据"""
        t0 = times[0]
        seconds = np.array([(t - t0).total_seconds() for t in times])
        
        filtered_pos = np.zeros_like(positions)
        filtered_vel = np.zeros_like(velocities)
        
        for i in range(3):
            spline_pos = UnivariateSpline(seconds, positions[:,i], k=k, s=len(seconds))
            filtered_pos[:,i] = spline_pos(seconds)
            
            spline_vel = UnivariateSpline(seconds, velocities[:,i], k=k, s=len(seconds))
            filtered_vel[:,i] = spline_vel(seconds)
        
        return filtered_pos, filtered_vel

    def filter_orbit_combined(self, times, positions, velocities, window_size=10):
        """先滑动窗口，再多项式拟合的组合方法"""
        sliding_pos, sliding_vel = self.filter_orbit_sliding(times, positions, velocities, window_size)
        
        t0 = times[0]
        seconds = np.array([(t - t0).total_seconds() for t in times])
        
        filtered_pos = np.zeros_like(positions)
        filtered_vel = np.zeros_like(velocities)
        
        for i in range(3):
            pos_coef = np.polyfit(seconds, sliding_pos[:,i], 4)
            filtered_pos[:,i] = np.polyval(pos_coef, seconds)
            
            vel_coef = np.polyfit(seconds, sliding_vel[:,i], 5)
            filtered_vel[:,i] = np.polyval(vel_coef, seconds)
        
        return filtered_pos, filtered_vel

    def filter_orbit_combined_spline(self, times, positions, velocities, window_size=10, k=4):
        """先滑动窗口，再样条拟合的组合方法"""
        # 1. 首先进行滑动窗口滤波
        sliding_pos, sliding_vel = self.filter_orbit_sliding(times, positions, velocities, window_size)
        
        # 2. 对滑动窗口结果进行样条拟合
        t0 = times[0]
        seconds = np.array([(t - t0).total_seconds() for t in times])
        
        filtered_pos = np.zeros_like(positions)
        filtered_vel = np.zeros_like(velocities)
        
        # 对每个坐标分量进行拟合
        for i in range(3):
            # 位置样条拟合
            spline_pos = UnivariateSpline(seconds, sliding_pos[:,i], k=k, s=len(seconds))
            filtered_pos[:,i] = spline_pos(seconds)
            
            # 速度样条拟合
            spline_vel = UnivariateSpline(seconds, sliding_vel[:,i], k=k, s=len(seconds))
            filtered_vel[:,i] = spline_vel(seconds)
        
        return filtered_pos, filtered_vel
        
    def filter_orbit_weighted(self, times, positions, velocities):
        """对成像时间段赋予更高权重"""
        t0 = times[0]
        seconds = np.array([(t - t0).total_seconds() for t in times])
        
        # 获取成像时间段
        data_start_time = self.frame.getSensingStart()
        data_stop_time = self.frame.getSensingStop()
        
        weights = np.ones(len(times))
        scene_indices = [i for i, t in enumerate(times) 
                        if data_start_time <= t <= data_stop_time]
        weights[scene_indices] = 2.0  # 成像段更高权重
        
        filtered_pos = np.zeros_like(positions)
        filtered_vel = np.zeros_like(velocities)
        
        for i in range(3):
            # 位置加权拟合
            pos_coef = np.polyfit(seconds, positions[:,i], 4, w=weights)
            filtered_pos[:,i] = np.polyval(pos_coef, seconds)
            
            # 速度加权拟合
            vel_coef = np.polyfit(seconds, velocities[:,i], 5, w=weights)
            filtered_vel[:,i] = np.polyval(vel_coef, seconds)
        
        return filtered_pos, filtered_vel

    def createOrbit(self):
        """
        Create orbit from multiple frames.
        """
        from isceobj.Orbit.Orbit import Orbit, StateVector
        
        # 创建新的轨道对象
        orbit = Orbit()
        orbit.configure()
        
        # 收集所有帧的轨道状态向量
        all_vectors = []
        for frame in self.frameList:
            if frame.orbit and frame.orbit._stateVectors:
                all_vectors.extend(frame.orbit._stateVectors)
        
        if not all_vectors:
            self.logger.error("No orbit state vectors found in any frames")
            raise RuntimeError("No orbit state vectors found in any frames")
            
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
            margin = datetime.timedelta(minutes=60)
            t_start_extended = t_start - margin
            t_end_extended = t_end + margin
            
            # 重新收集扩展时间范围内的状态向量
            extended_vectors = []
            for frame in self.frameList:
                if frame.orbit and frame.orbit._stateVectors:
                    for sv in frame.orbit._stateVectors:
                        t = sv.getTime()
                        if t_start_extended <= t <= t_end_extended:
                            extended_vectors.append(sv)
            
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
            raise RuntimeError("Insufficient orbit state vectors for interpolation")
        
        # 添加到轨道对象
        for sv in unique_vectors:
            orbit.addStateVector(sv)
        
        self.logger.info(f"Created orbit with {len(unique_vectors)} state vectors")
        self.logger.info(f"Orbit time range: {unique_vectors[0].getTime()} to {unique_vectors[-1].getTime()}")
        
        return orbit

    def extractImage(self):
        """
        提取图像数据
        """
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
            
            # 设置当前frame
            self.frame = frame
            
            try:
                # 解析XML文件
                self._xml_root = ET.parse(xml_file).getroot()
                
                # 解析元数据
                self.populateMetadata()
                
                # 提取图像数据
                outputNow = self.output + "_" + str(i) if len(self._xmlFileList) > 1 else self.output
                self.extractFrameImage(tiff_file, outputNow)
                
                # 处理轨道信息
                if self._orbitFileList and i < len(self._orbitFileList):
                    self._orbitFile = self._orbitFileList[i]
                    self.logger.info(f"Processing orbit file for frame {i+1}: {self._orbitFile}")
                    orb = self.extractOrbit()
                    if orb:
                        frame.setOrbit(orb)
                        self.logger.info(f"Successfully set orbit for frame {i+1}")
                    else:
                        self.logger.warning(f"Failed to extract orbit for frame {i+1}")
                        # 尝试从XML创建轨道
                        orb = self.createOrbit()
                        if orb:
                            frame.setOrbit(orb)
                            self.logger.info(f"Successfully created orbit from XML for frame {i+1}")
                        else:
                            self.logger.error(f"Failed to create orbit from XML for frame {i+1}")
                            raise RuntimeError(f"Failed to create orbit for frame {i+1}")
                else:
                    self.logger.warning(f"No orbit file provided for frame {i+1}, creating from XML")
                    orb = self.createOrbit()
                    if orb:
                        frame.setOrbit(orb)
                        self.logger.info(f"Successfully created orbit from XML for frame {i+1}")
                    else:
                        self.logger.error(f"Failed to create orbit from XML for frame {i+1}")
                        raise RuntimeError(f"Failed to create orbit for frame {i+1}")
                
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
        
        # 确保frame和frameList被正确设置
        if merged_frame:
            self.frame = merged_frame
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
        else:
            # 如果没有合并的frame，使用第一个frame
            self.frame = self.frameList[0]
            self.frameList = [self.frame]
        
        return self.frame

    def extractFrameImage(self, tiff_file, output):
        """
        提取单个帧的图像数据
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise Exception('GDAL python bindings not found.')
        
        # 打开TIFF文件
        src = gdal.Open(tiff_file.strip(), gdal.GA_ReadOnly)
        if src is None:
            raise Exception(f"Failed to open TIFF file: {tiff_file}")
        
        # 获取图像信息
        width = self.frame.getNumberOfSamples()
        length = self.frame.getNumberOfLines()
        
        # 读取数据并写入输出文件
        band = src.GetRasterBand(1)
        with open(output, 'wb') as fid:
            for ii in range(length):
                data = band.ReadAsArray(0, ii, width, 1)
                data.tofile(fid)
        
        # 清理资源
        src = None
        band = None
        
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
        
        # 生成辅助文件
        self.makeFakeAux(output)

    def mergeFrames(self):
        """
        合并多个帧，处理重叠区域
        """
        if not self.frameList:
            return None
        
        # 按时间排序帧
        sorted_frames = sorted(self.frameList, key=lambda x: x.getSensingStart())
        
        # 计算总行数
        total_lines = 0
        for i, frame in enumerate(sorted_frames):
            if i > 0:
                # 计算与前一帧的重叠
                prev_frame = sorted_frames[i-1]
                overlap = (prev_frame.getSensingStop() - frame.getSensingStart()).total_seconds()
                if overlap > 0:
                    # 重叠区域的行数
                    overlap_lines = int(overlap * frame.getInstrument().getPulseRepetitionFrequency())
                    # 确保重叠行数不超过帧的行数
                    overlap_lines = min(overlap_lines, frame.getNumberOfLines())
                    total_lines += frame.getNumberOfLines() - overlap_lines
                else:
                    total_lines += frame.getNumberOfLines()
        
        # 创建输出文件
        output_file = self.output
        width = sorted_frames[0].getNumberOfSamples()
        
        # 使用numpy进行数据处理
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
                # 处理与前一帧的重叠
                prev_frame = sorted_frames[i-1]
                overlap = (prev_frame.getSensingStop() - frame.getSensingStart()).total_seconds()
                if overlap > 0:
                    # 重叠区域的行数
                    overlap_lines = int(overlap * frame.getInstrument().getPulseRepetitionFrequency())
                    # 确保重叠行数不超过帧的行数
                    overlap_lines = min(overlap_lines, frame.getNumberOfLines())
                    
                    # 直接复制非重叠区域到当前位置
                    if overlap_lines < frame.getNumberOfLines():
                        copy_lines = frame.getNumberOfLines() - overlap_lines
                        merged_data[current_line:current_line+copy_lines] = frame_data[overlap_lines:]
                        current_line += copy_lines
                else:
                    # 无重叠,直接复制整个帧
                    merged_data[current_line:current_line+frame.getNumberOfLines()] = frame_data
                    current_line += frame.getNumberOfLines()
        
        # 写入合并后的数据
        with open(output_file, 'wb') as f:
            merged_data.tofile(f)
        
        # 创建合并后的帧
        merged_frame = Frame()
        merged_frame.configure()
        
        # 设置基本属性
        merged_frame.setSensingStart(sorted_frames[0].getSensingStart())
        merged_frame.setSensingStop(sorted_frames[-1].getSensingStop())
        merged_frame.setStartingRange(sorted_frames[0].getStartingRange())
        merged_frame.setFarRange(sorted_frames[-1].getFarRange())
        merged_frame.setNumberOfLines(total_lines)
        merged_frame.setNumberOfSamples(width)
        
        # 设置仪器参数
        merged_frame.setInstrument(sorted_frames[0].getInstrument())
        
        # 设置图像
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
        
        # 处理轨道信息
        if self._orbitFile:
            # 如果提供了轨道文件，使用完整的轨道文件
            self.logger.info(f"Using provided orbit file: {self._orbitFile}")
            orb = self.extractOrbit()
            if orb:
                merged_frame.setOrbit(orb)
                self.logger.info("Successfully set orbit from file")
            else:
                self.logger.warning("Failed to extract orbit from file")
        else:
            # 如果没有提供轨道文件，合并各个帧的轨道
            merged_orbit = self.mergeOrbits([frame.orbit for frame in sorted_frames])
            if merged_orbit:
                merged_frame.setOrbit(merged_orbit)
                self.logger.info("Successfully merged orbits from frames")
            else:
                self.logger.warning("Failed to merge orbits from frames")
        
        return merged_frame

    def mergeOrbits(self, orbits):
        """
        Merge multiple orbits into one.
        """
        if not orbits:
            return None
        
        # 获取所有轨道的状态向量
        all_vectors = []
        for orb in orbits:
            all_vectors.extend(orb._stateVectors)
        
        # 按时间排序
        all_vectors.sort(key=lambda x: x.getTime())
        
        # 创建新的轨道对象
        merged_orbit = Orbit()
        merged_orbit.configure()
        
        # 添加所有状态向量
        for vec in all_vectors:
            merged_orbit.addStateVector(vec)
        
        # 设置轨道的时间范围
        if all_vectors:
            merged_orbit.setTimeRange(all_vectors[0].getTime(), all_vectors[-1].getTime())
            self.logger.info(f"Set merged orbit time range: {all_vectors[0].getTime()} to {all_vectors[-1].getTime()}")
        
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
        """
        将轨道信息保存到单独的文件中
        """
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
        """
        从文件中恢复轨道信息
        """
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
