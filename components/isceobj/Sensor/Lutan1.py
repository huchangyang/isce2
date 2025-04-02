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

FILTER_METHOD = Component.Parameter('filterMethod',
                            public_name ='filterMethod',
                            default = 'combined_weighted_filter',
                            type=str,
                            doc = 'Orbit filter method (poly_filter, physics_filter, combined_weighted_filter)')


class Lutan1(Sensor):

    "Class for Lutan-1 SLC data"
    
    family = 'l1sm'
    logging_name = 'isce.sensor.Lutan1'

    parameter_list = (TIFF, ORBIT_FILE, FILTER_METHOD) + Sensor.parameter_list

    def __init__(self, name = ''):
        super(Lutan1,self).__init__(self.__class__.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        self._xml_root = None
        self.doppler_coeff = None
        self.filterMethod = 'combined_weighted_filter'

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
        
    def poly_filter(self, times, positions, velocities):
        """Use polynomial fitting to filter orbit data"""
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

    def extractOrbitFromAnnotation(self):
        '''Extract orbit information from xml annotation and apply filtering'''
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

        positions = np.array(positions)
        velocities = np.array(velocities)
        
        t0 = timestamps[0]
        time_seconds = np.array([(t - t0).total_seconds() for t in timestamps])

        img_start_sec = (self.frame.getSensingStart() - t0).total_seconds()
        img_stop_sec = (self.frame.getSensingStop() - t0).total_seconds()
        
        if self.filterMethod == 'poly_filter':
            filtered_pos, filtered_vel = self.poly_filter(timestamps, positions, velocities)
        elif self.filterMethod == 'physics_filter':
            filtered_pos, filtered_vel = self.physics_constrained_filter(
                time_seconds, positions, velocities,
                img_start_sec, img_stop_sec
            )
        else:  # default to combined_weighted_filter
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
