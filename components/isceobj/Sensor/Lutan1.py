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

    def __init__(self, name = ''):
        super(Lutan1,self).__init__(self.__class__.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        self._xml_root = None
        self.doppler_coeff = None
        self.filterMethod = 'weighted'
        self.frameList = []
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
                self.frame.orbit.setOrbitSource(os.path.basename(self._orbitFile))
            else:
                self.logger.warning(f"Orbit file {self._orbitFile} not found. Using orbit from annotation file.")
                orb = self.extractOrbitFromAnnotation()
                self.frame.orbit.setOrbitSource(os.path.basename(self._xml))
                self.frame.orbit.setOrbitSource('Annotation')
        else:
            self.logger.warning("No orbit file provided. Using orbit from annotation file.")
            orb = self.extractOrbitFromAnnotation()
            self.frame.orbit.setOrbitSource(os.path.basename(self._xml))
            self.frame.orbit.setOrbitSource('Annotation')

        # 确保 orb 不为 None
        if orb is None:
            self.logger.error("Failed to extract orbit information.")
            raise RuntimeError("Failed to extract orbit information.")

        for sv in orb:
            self.frame.orbit.addStateVector(sv)

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

        # I based the margin on the data that I have.
        # Lutan-1 position and velocity sampling frequency is 1 Hz
        margin = datetime.timedelta(minutes=30.0)
        tstart = self.frame.getSensingStart() - margin
        tend = self.frame.getSensingStop() + margin

        file_ext = os.path.splitext(self._orbitFile)[1].lower()

        if file_ext == '.xml':
            try:
                fp = open(self._orbitFile, 'r')
            except IOError as strerr:
                print("IOError: %s" % strerr)
            
            _xml_root = ET.ElementTree(file=fp).getroot()
            node = _xml_root.find('Data_Block/List_of_OSVs')
            
            for child in node:
                timestamp = self.convertToDateTime(child.find('UTC').text)
                if (timestamp >= tstart) and (timestamp <= tend):
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
                    orb.addStateVector(vec)

            fp.close()

        elif file_ext == '.txt':
            with open(self._orbitFile, 'r') as fid:
                for line in fid:
                    if not line.startswith('#'):
                        break
                
                for line in fid:
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
                        
                        if (timestamp >= tstart) and (timestamp <= tend):
                            pos = [float(fields[6]), float(fields[7]), float(fields[8])]
                            vel = [float(fields[9]), float(fields[10]), float(fields[11])]
                            vec = StateVector()
                            vec.setTime(timestamp)
                            vec.setPosition(pos)
                            vec.setVelocity(vel)
                            orb.addStateVector(vec)
        else:
            raise Exception("Unsupported orbit file extension: %s" % file_ext)
        return orb
        
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

    def extractOrbitFromAnnotation(self):
        '''
        从xml注释中提取轨道信息并进行滤波
        如果没有精轨数据，使用此方法
        '''
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
        
        # 收集所有轨道数据点
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
        
        # 计算相对时间（秒）
        time_seconds = [(t - timestamps[0]).total_seconds() for t in timestamps]
        self.filterMethod = 'weighted'
        # 根据选择的方法进行滤波
        if self.filterMethod == 'standard':
            filtered_pos, filtered_vel = self.filter_orbit(timestamps, np.array(positions), np.array(velocities))
        elif self.filterMethod == 'sliding':
            filtered_pos, filtered_vel = self.filter_orbit_sliding(timestamps, np.array(positions), np.array(velocities))
        elif self.filterMethod == 'spline':
            filtered_pos, filtered_vel = self.filter_orbit_spline(timestamps, np.array(positions), np.array(velocities))
        elif self.filterMethod == 'combined':
            filtered_pos, filtered_vel = self.filter_orbit_combined(timestamps, np.array(positions), np.array(velocities))
        elif self.filterMethod == 'combined_spline':
            filtered_pos, filtered_vel = self.filter_orbit_combined_spline(timestamps, np.array(positions), np.array(velocities))
        elif self.filterMethod == 'weighted':
            filtered_pos, filtered_vel = self.filter_orbit_weighted(timestamps, np.array(positions), np.array(velocities))
        else:
            print(f"Warning: Unknown filter method {self.filterMethod}, using standard method")
            filtered_pos, filtered_vel = self.filter_orbit(timestamps, np.array(positions), np.array(velocities))
        
        # 将滤波后的结果转换为轨道状态向量
        for i, timestamp in enumerate(timestamps):
            vec = StateVector()
            vec.setTime(timestamp)
            vec.setPosition(filtered_pos[i].tolist())
            vec.setVelocity(filtered_vel[i].tolist())
            frameOrbit.addStateVector(vec)
        
        return frameOrbit
    
    def extractImage(self):
        # 验证用户输入并自动查找文件
        self.validateUserInputs()
        
        # 如果提供了XML配置文件，先从中加载文件列表
        if self._xmlConfig is not None:
            if not self.loadFromXML():
                self.logger.error("Failed to load from XML configuration file")
                return None
        
        if(len(self._tiffList) != len(self._orbitFileList) and len(self._orbitFileList) > 0):
            self.logger.error(
                "Number of orbit files different from number of image files.")
            raise RuntimeError
        
        self.frameList = []
        for i in range(len(self._tiffList)):
            appendStr = "_" + str(i)
            # 如果只有一个文件，不改变输出文件名
            if(len(self._tiffList) == 1):
                appendStr = ''

            self.frame = Frame()
            self.frame.configure()

            self._tiff = self._tiffList[i]
            # 检查是否提供了轨道文件
            if len(self._orbitFileList) > 0:
                self._orbitFile = self._orbitFileList[i]
            else:
                self._orbitFile = None
            
            # 解析元数据并提取图像
            try:
                self.parse()
                outputNow = self.output + appendStr
                
                # 提取图像数据
                width = self.frame.getNumberOfSamples()
                lgth = self.frame.getNumberOfLines()
                src = gdal.Open(self._tiff.strip(), gdal.GA_ReadOnly)

                # Band 1 as real and band 2 as imaginary numbers
                band1 = src.GetRasterBand(1)
                band2 = src.GetRasterBand(2)
                cJ = np.complex64(1.0j)

                fid = open(outputNow, 'wb')
                for ii in range(lgth):
                    # Combine the real and imaginary to make
                    # them in to complex numbers
                    real = band1.ReadAsArray(0,ii,width,1)
                    imag = band2.ReadAsArray(0,ii,width,1)
                    # Data becomes np.complex128 after combining them
                    data = real + (cJ * imag)
                    data.tofile(fid)

                fid.close()
                real = None
                imag = None
                src = None
                band1 = None
                band2 = None

                # 设置图像属性
                slcImage = isceobj.createSlcImage()
                slcImage.setByteOrder('l')
                slcImage.setFilename(outputNow)
                slcImage.setAccessMode('read')
                slcImage.setWidth(self.frame.getNumberOfSamples())
                slcImage.setLength(self.frame.getNumberOfLines())
                slcImage.setXmin(0)
                slcImage.setXmax(self.frame.getNumberOfSamples())
                self.frame.setImage(slcImage)
                
                # 生成辅助文件
                self.makeFakeAux(outputNow)
                
                # 将当前帧添加到帧列表
                self.frameList.append(self.frame)
            except IOError as e:
                self.logger.error(f"Error processing file {self._tiff}: {str(e)}")
                continue
        
        # 如果处理了多个文件，返回帧列表
        if len(self.frameList) > 1:
            # 为了兼容 tkfunc 函数，临时设置 _imageFileList 属性
            self._imageFileList = self._tiffList
            
            # 特殊处理：如果是 Lutan1 数据，使用自定义的合并方法而不是 Track.py 中的 findOverlapLine
            # 这是为了避免 EOFError: read() didn't return enough bytes 错误
            from isceobj.Scene.Track import Track
            tk = Track()
            
            # 添加所有帧到 Track 对象
            for frame in self.frameList:
                tk.addFrame(frame)
                
            # 创建合并后的帧
            tk.createInstrument()
            
            # 自定义合并方法：直接使用基于时间的合并，跳过 findOverlapLine
            # 修改 Track.py 中的 reAdjustStartLine 方法
            original_reAdjustStartLine = tk.reAdjustStartLine
            
            def custom_reAdjustStartLine(sortedList, width):
                """自定义的 reAdjustStartLine 方法，跳过 findOverlapLine"""
                startLine = [sortedList[0][0]]
                outputs = [sortedList[0][1]]
                for i in range(1, len(sortedList)):
                    startLine.append(sortedList[i][0])
                    outputs.append(sortedList[i][1])
                return startLine, outputs
            
            # 替换方法
            tk.reAdjustStartLine = custom_reAdjustStartLine
            
            # 继续合并过程
            tk.createTrack(self.output)
            tk.createOrbit()
            
            # 恢复原始方法
            tk.reAdjustStartLine = original_reAdjustStartLine
            
            # 获取合并后的帧
            result = tk._frame
            
            # 使用完后删除临时属性
            delattr(self, '_imageFileList')
            return result
        
        # 如果只处理了一个文件，返回单个帧
        return self.frame

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
                self._tiffList = sorted(glob.glob(os.path.join(slc_dir, "*.tiff")))
                if not self._tiffList:
                    self.logger.warning(f"No TIFF files found in {slc_dir}")
            else:
                # 格式2: <tiff>./SLC/file.tiff</tiff>
                tiff_element = root.find(".//tiff")
                if tiff_element is not None:
                    self._tiffList = [tiff_element.text]
                else:
                    self.logger.error("No SLC_DIR or tiff element found in XML configuration")
                    return False
                
            # 获取轨道文件目录（可选）
            orbit_dir_element = root.find(".//property[@name='ORBIT_DIR']/value")
            if orbit_dir_element is not None:
                orbit_dir = orbit_dir_element.text
                # 检索所有XML轨道文件
                self._orbitFileList = sorted(glob.glob(os.path.join(orbit_dir, "*.xml")))
                if not self._orbitFileList:
                    self.logger.warning(f"No orbit XML files found in {orbit_dir}")
                    self._orbitFileList = []
            else:
                # 格式2: <orbitFile>./orbits/file.xml</orbitFile> 或 <ORBITFILE>./orbits/file.xml</ORBITFILE>
                # 尝试不同大小写的轨道文件标签
                orbit_element = root.find(".//orbitFile") or root.find(".//ORBITFILE")
                
                # 也尝试 property 格式的轨道文件
                if orbit_element is None:
                    orbit_property = root.find(".//property[@name='orbitFile']/value") or root.find(".//property[@name='ORBITFILE']/value")
                    if orbit_property is not None:
                        orbit_element = orbit_property
                
                if orbit_element is not None and orbit_element.text:
                    self._orbitFileList = [orbit_element.text]
                    self.logger.info(f"Using orbit file: {orbit_element.text}")
                else:
                    self.logger.warning("No ORBIT_DIR or orbitFile element found in XML configuration, proceeding without orbit files")
                    self._orbitFileList = []
                
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
            self.logger.info(f"Found {len(self._tiffList)} TIFF files and {len(self._orbitFileList)} orbit files")
            for i, tiff in enumerate(self._tiffList):
                self.logger.info(f"TIFF {i+1}: {tiff}")
            for i, orbit in enumerate(self._orbitFileList):
                self.logger.info(f"Orbit {i+1}: {orbit}")
                
            # 如果轨道文件数量与TIFF文件数量不匹配，尝试自动匹配
            if len(self._orbitFileList) > 0 and len(self._orbitFileList) != len(self._tiffList):
                self.logger.warning("Number of orbit files does not match number of TIFF files")
                self.logger.info("Attempting to match orbit files with TIFF files...")
                
                # 如果只有一个轨道文件，假设它适用于所有TIFF文件
                if len(self._orbitFileList) == 1:
                    self.logger.info("Using the single orbit file for all TIFF files")
                    self._orbitFileList = [self._orbitFileList[0]] * len(self._tiffList)
                else:
                    # 尝试根据日期匹配轨道文件
                    try:
                        # 提取每个TIFF文件名中的日期信息
                        tiff_dates = []
                        for tiff in self._tiffList:
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
                        for orbit in self._orbitFileList:
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
                            self.logger.warning("Could not match all orbit files based on date, proceeding without orbit files")
                            self._orbitFileList = []
                    except Exception as e:
                        self.logger.warning(f"Error matching orbit files: {str(e)}")
                        self.logger.warning("Proceeding without orbit files")
                        self._orbitFileList = []
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading XML configuration: {str(e)}")
            return False
