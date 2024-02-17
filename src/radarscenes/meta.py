"""Compute Lever Arm.

This module computes the distance between the rear-axle and each radar.

Ex.:
    Sensor 1: x_len = 3.6493 m, y_len = -0.8737 m
    Sensor 2: x_len = 3.8615 m, y_len = -0.6958 m
    Sensor 3: x_len = 3.8649 m, y_len = 0.6900 m
    Sensor 4: x_len = 3.6498 m, y_len = 0.8730 m
"""

import os
import json
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot


# Path to a sequence
path = os.path.join('/mnt/sda/paulo/delta/data/radarscenes/data/sequence_10')
sensors_id = [1,     2,  3,  4]
angles     = [-85, -25, 25, 85]

# Reading scene data
with open(os.path.join(path, 'scenes.json')) as json_file:
    parsed = json.load(json_file)
    scenes_data = parsed['scenes']

radar_data  = h5py.File(os.path.join(path, 'radar_data.h5'), 'r')
odometry    = radar_data['odometry']
radar_scans = radar_data['radar_data']

# Looping through sequence to recover the lever arm
for i, sensor_id in enumerate(sensors_id):
    x_len = []
    y_len = []
    for timestamp, info in scenes_data.items():
        if info['sensor_id'] == sensor_id:
            speed = odometry[info['odometry_index']][4]
            init, ennd = info['radar_indices']
            scans = np.array([[range_sc, azimuth_sc, x_cc, y_cc]
                    for timestamp, sensor_id, range_sc, azimuth_sc, rcs, vr, vr_compensated, x_cc, y_cc, x_seq, y_seq, uuid, track_id, label_id in radar_scans[init:ennd]])
            # Computing x_rad and y_rad
            x_rad = scans[:,0] * np.cos(scans[:,1])
            y_rad = scans[:,0] * np.sin(scans[:,1])
            scans[:,0] = x_rad
            scans[:,1] = y_rad
            R = Rot.from_euler('z', [angles[i]], degrees=True)
            r = R.as_matrix().squeeze()[:2,:2]
            scans[:,[0,1]] = np.dot(scans[:,[0,1]], r.T)
            # Computing lever arm
            x_len.append(np.mean(scans[:,2] - scans[:,0]))
            y_len.append(np.mean(scans[:,3] - scans[:,1]))
    print(f'Sensor {sensor_id}: x_len = {np.mean(x_len):.4f} m, y_len = {np.mean(y_len):.4f} m')
