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


lookMap = { 'RIGHT' : -1,
            'LEFT' : 1}

# Antenna dimensions 9.8 x 3.4 m
antennaLength = 9.8

XML = Component.Parameter('xml',
        public_name = 'xml',
        default = None,
        type = str,
        doc = 'Input XML file')


TIFF = Component.Parameter('tiff',
                            public_name ='tiff',
                            default = None,
                            type=str,
                            doc = 'Input image file')

ORBIT_FILE = Component.Parameter('orbitFile',
                            public_name ='orbitFile',
                            default = None,
                            type=str,
                            doc = 'Orbit file')


class Lutan1(Sensor):

    "Class for Lutan-1 SLC data"
    
    family = 'l1sm'
    logging_name = 'isce.sensor.Lutan1'

    parameter_list = (TIFF, ORBIT_FILE) + Sensor.parameter_list

    def __init__(self, name = ''):
        super(Lutan1,self).__init__(self.__class__.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        self._xml_root = None
        self.doppler_coeff = None
        self.filterMethod = 'weighted'

    def parse(self):
        xmlFileName = self.tiff[:-4] + "meta.xml"
        self.xml = xmlFileName

        with open(self.xml, 'r') as fid:
            xmlstr = fid.read()
        
        self._xml_root = ET.fromstring(xmlstr)
        self.populateMetadata()
        fid.close()

        if self.orbitFile:
            # Check if orbit file exists or not
            if os.path.isfile(self.orbitFile) == True:
                orb = self.extractOrbit()
                self.frame.orbit.setOrbitSource(os.path.basename(self.orbitFile))
            else:
                pass
        else:
            warnings.warn("WARNING! No orbit file found. Orbit information from the annotation file is used for processing.")
            orb = self.extractOrbitFromAnnotation()
            self.frame.orbit.setOrbitSource(os.path.basename(self.xml))
            self.frame.orbit.setOrbitSource('Annotation')

        for sv in orb:
            self.frame.orbit.addStateVector(sv)

    def convertToDateTime(self,string):
        dt = datetime.datetime.strptime(string,"%Y-%m-%dT%H:%M:%S.%f")
        return dt


    def grab_from_xml(self, path):
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

        file_ext = os.path.splitext(self.orbitFile)[1].lower()

        if file_ext == '.xml':
            try:
                fp = open(self.orbitFile, 'r')
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
            with open(self.orbitFile, 'r') as fid:
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

    def physics_constrained_filter(self, time_data, positions, velocities, 
                                 img_start_sec=None, img_stop_sec=None):
        """基于物理约束的轨道滤波方法
        
        参数:
        time_data: 时间序列（秒）
        positions: 位置数据 (N x 3的numpy数组)
        velocities: 速度数据 (N x 3的numpy数组)
        img_start_sec: 成像开始时间（秒）
        img_stop_sec: 成像结束时间（秒）
        """
        # 直接指定固定权重
        weights_img = {'original': 0.222, 'fitted': 0.522, 'theory': 0.256}
        weights_non_img = {'original': 0.222, 'fitted': 0.522, 'theory': 0.256}
        
        # 1. 检测并处理异常点
        time_diffs = np.diff(time_data)
        median_dt = np.median(time_diffs)
        dt_threshold = 3 * median_dt  # 降低阈值，使检测更敏感
        
        # 标记时间异常点
        time_anomaly_mask = np.zeros(len(time_data), dtype=bool)
        time_anomaly_mask[1:] = time_diffs > dt_threshold
        time_anomaly_mask[:-1] |= time_diffs > dt_threshold
        
        # 检测位置和速度异常值
        pos_anomaly_mask = np.zeros_like(time_anomaly_mask)
        vel_anomaly_mask = np.zeros_like(time_anomaly_mask)
        
        for i in range(3):  # 对每个维度分别处理
            # 位置异常检测
            pos_median = np.median(positions[:, i])
            pos_mad = np.median(np.abs(positions[:, i] - pos_median))
            pos_anomaly_mask |= np.abs(positions[:, i] - pos_median) > 3.0 * pos_mad
            
            # 速度异常检测
            vel_median = np.median(velocities[:, i])
            vel_mad = np.median(np.abs(velocities[:, i] - vel_median))
            vel_anomaly_mask |= np.abs(velocities[:, i] - vel_median) > 3.0 * vel_mad
        
        # 合并所有异常标记
        anomaly_mask = time_anomaly_mask | pos_anomaly_mask | vel_anomaly_mask
        
        # 2. 对速度进行多项式拟合
        fitted_velocities = np.zeros_like(velocities)
        for i in range(3):
            valid_indices = ~anomaly_mask
            if np.sum(valid_indices) > 5:
                coeffs = np.polyfit(time_data[valid_indices], velocities[valid_indices, i], 5)
                fitted_velocities[:, i] = np.polyval(coeffs, time_data)
            else:
                coeffs = np.polyfit(time_data, velocities[:, i], 5)
                fitted_velocities[:, i] = np.polyval(coeffs, time_data)
        
        # 3. 对位置进行多项式拟合
        fitted_positions = np.zeros_like(positions)
        for i in range(3):
            valid_indices = ~anomaly_mask
            if np.sum(valid_indices) > 5:
                coeffs = np.polyfit(time_data[valid_indices], positions[valid_indices, i], 4)
                fitted_positions[:, i] = np.polyval(coeffs, time_data)
            else:
                coeffs = np.polyfit(time_data, positions[:, i], 4)
                fitted_positions[:, i] = np.polyval(coeffs, time_data)
        
        # 4. 计算基于位置的理论速度
        theoretical_velocities = np.zeros_like(velocities)
        dt = median_dt
        
        # 使用中心差分计算速度
        for i in range(1, len(time_data)-1):
            if not anomaly_mask[i]:
                prev_idx = i - 1
                next_idx = i + 1
                
                while prev_idx > 0 and anomaly_mask[prev_idx]:
                    prev_idx -= 1
                while next_idx < len(time_data)-1 and anomaly_mask[next_idx]:
                    next_idx += 1
                
                if not (anomaly_mask[prev_idx] or anomaly_mask[next_idx]):
                    dt_local = time_data[next_idx] - time_data[prev_idx]
                    if dt_local > 0:
                        theoretical_velocities[i] = (fitted_positions[next_idx] - fitted_positions[prev_idx]) / dt_local
        
        # 处理边界点和异常点
        # 向前填充
        last_valid = None
        for i in range(len(time_data)):
            if not anomaly_mask[i] and np.any(theoretical_velocities[i]):
                last_valid = theoretical_velocities[i].copy()
            elif last_valid is not None:
                theoretical_velocities[i] = last_valid
        
        # 向后填充未处理的点
        last_valid = None
        for i in range(len(time_data)-1, -1, -1):
            if not anomaly_mask[i] and np.any(theoretical_velocities[i]):
                last_valid = theoretical_velocities[i].copy()
            elif last_valid is not None:
                theoretical_velocities[i] = last_valid
        
        # 5. 在成像时间段内特殊处理
        if img_start_sec is not None and img_stop_sec is not None:
            img_mask = (time_data >= img_start_sec) & (time_data <= img_stop_sec)
            if np.any(img_mask):
                valid_img_mask = img_mask & ~anomaly_mask
                if np.any(valid_img_mask):
                    img_indices = np.where(img_mask)[0]
                    for i in img_indices:
                        if anomaly_mask[i]:
                            theoretical_velocities[i] = fitted_velocities[i]
        
        # 6. 融合拟合速度和理论速度
        filtered_velocities = np.zeros_like(velocities)
        for i in range(3):
            if img_start_sec is not None and img_stop_sec is not None:
                img_mask = (time_data >= img_start_sec) & (time_data <= img_stop_sec)
                
                # 成像时间段使用img权重
                filtered_velocities[img_mask, i] = (
                    velocities[img_mask, i] * weights_img['original'] +
                    fitted_velocities[img_mask, i] * weights_img['fitted'] +
                    theoretical_velocities[img_mask, i] * weights_img['theory']
                )
                
                # 非成像时间段使用non_img权重
                filtered_velocities[~img_mask, i] = (
                    velocities[~img_mask, i] * weights_non_img['original'] +
                    fitted_velocities[~img_mask, i] * weights_non_img['fitted'] +
                    theoretical_velocities[~img_mask, i] * weights_non_img['theory']
                )
            else:
                # 如果没有成像时间信息，使用non_img权重
                filtered_velocities[:, i] = (
                    velocities[:, i] * weights_non_img['original'] +
                    fitted_velocities[:, i] * weights_non_img['fitted'] +
                    theoretical_velocities[:, i] * weights_non_img['theory']
                )
        
        return fitted_positions, filtered_velocities

    def extractOrbitFromAnnotation(self):
        '''从xml注释中提取轨道信息并进行滤波'''
        try:
            fp = open(self.xml, 'r')
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
        
        # 收集轨道数据
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

        # 转换为numpy数组
        positions = np.array(positions)
        velocities = np.array(velocities)
        
        # 计算相对时间（秒）
        t0 = timestamps[0]
        time_seconds = np.array([(t - t0).total_seconds() for t in timestamps])
        
        # 获取成像时间的相对秒数
        img_start_sec = (self.frame.getSensingStart() - t0).total_seconds()
        img_stop_sec = (self.frame.getSensingStop() - t0).total_seconds()
        
        # 应用物理约束滤波
        filtered_pos, filtered_vel = self.physics_constrained_filter(
            time_seconds, positions, velocities,
            img_start_sec, img_stop_sec
        )
        
        # 创建轨道状态向量
        for i, timestamp in enumerate(timestamps):
            vec = StateVector()
            vec.setTime(timestamp)
            vec.setPosition(filtered_pos[i].tolist())
            vec.setVelocity(filtered_vel[i].tolist())
            frameOrbit.addStateVector(vec)
        
        return frameOrbit
    
    def extractImage(self):
        self.parse()
        width = self.frame.getNumberOfSamples()
        lgth = self.frame.getNumberOfLines()
        src = gdal.Open(self.tiff.strip(), gdal.GA_ReadOnly)

        # Band 1 as real and band 2 as imaginary numbers
        # Confirmed by Zhang Yunjun
        band1 = src.GetRasterBand(1)
        band2 = src.GetRasterBand(2)
        cJ = np.complex64(1.0j)

        fid = open(self.output, 'wb')
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

        ####
        slcImage = isceobj.createSlcImage()
        slcImage.setByteOrder('l')
        slcImage.setFilename(self.output)
        slcImage.setAccessMode('read')
        slcImage.setWidth(self.frame.getNumberOfSamples())
        slcImage.setLength(self.frame.getNumberOfLines())
        slcImage.setXmin(0)
        slcImage.setXmax(self.frame.getNumberOfSamples())
        self.frame.setImage(slcImage)

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
