""" Dataset classes for using with Pytorch.
"""

import os
import json
import h5py
import random
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import Dataset


class RadarDataset(Dataset):
    
    def __init__(self, path, map_params, augment_params, other_params):
        super(RadarDataset, self).__init__()
        
        self.path = path
        self.map_params = map_params
        self.augment_params = augment_params
        self.other_params = other_params
        self.speeds = []
        self.timestamps = []

        # Getting sequences according to the split
        with open(os.path.join(self.path,'my_sequences.json')) as json_file:
            parsed = json.load(json_file)
            sequences = parsed['sequences']  
        selection = []
        for k in sequences.keys():
            info = sequences[k]
            if info['category'] == other_params['split']:
                selection.append(k)

        # Creating unified file to optimize RAM
        split_path = os.path.join(self.path, 'Processed', other_params['split'])
        os.makedirs(split_path, exist_ok=True)
        _, _, files = next(os.walk(split_path))
        files_nums = [int(f.split('.')[0].split('_')[-1]) for f in files]
        id = max(files_nums) + 1 if files_nums else 1
        self.offset_dict = {}
        self.length = 0
        self.unified_file = os.path.join(split_path, f'unified_{id}.txt')

        # Writing the data
        with open(self.unified_file, 'w') as u:

            # Reading data for each selected sequence
            pbar = tqdm(total=len(selection), desc=f"Creating {other_params['split']} set")
            for sequence in selection:
                sequence = os.path.join(self.path, sequence)
                with open(os.path.join(sequence, 'scenes.json')) as json_file:
                    parsed = json.load(json_file)
                    scenes_data = parsed['scenes']
                radar_data  = h5py.File(os.path.join(sequence, 'radar_data.h5'), 'r')
                odometry    = radar_data['odometry']
                radar_scans = radar_data['radar_data']

                # Getting data from the selected sensor
                for timestamp, info in scenes_data.items():
                    if info['sensor_id'] == other_params['sensor_id']:
                        speed = odometry[info['odometry_index']][4]
                        init, ennd = info['radar_indices']
                        scans = np.array([[range_sc, azimuth_sc, vr, rcs]
                                for timestamp, sensor_id, range_sc, azimuth_sc, rcs, vr, vr_compensated, x_cc, y_cc, x_seq, y_seq, uuid, track_id, label_id in radar_scans[init:ennd]])
                        # Computing x_rad and y_rad
                        x_rad = scans[:,0] * np.cos(scans[:,1])
                        y_rad = scans[:,0] * np.sin(scans[:,1])
                        scans[:,0] = x_rad
                        scans[:,1] = y_rad
                        # Creating unified list where scans = [x_rad, y_rad, vr, rcs]
                        data = [timestamp, scans.tolist(), speed]
                        self.offset_dict[self.length] = u.tell()
                        u.write(str(data)+'\n')
                        self.speeds.append(speed)
                        self.timestamps.append(timestamp)
                        self.length += 1

                pbar.update(1)
            pbar.close()

        print(f"Initialization of the {other_params['split']} set complete. A total of {len(self.offset_dict)} synchronized frames available.")
            
        
    def __len__(self):
        return len(self.offset_dict)
    

    def __getitem__(self, id):
        # Reading data on demand
        offset = self.offset_dict[id]
        with open(self.unified_file, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            # Processing recovered data
            curr = eval(line)
            timestamp = torch.tensor([int(curr[0])], dtype=torch.int64)
            data = np.array(curr[1])
            speed_ref = torch.tensor([curr[2]], dtype=torch.float)          
            # Augmenting data
            if self.augment_params['augment']:
                # Computing random angles to be used in the sequence
                delta_z = random.uniform(-self.augment_params['delta_z'], self.augment_params['delta_z']) * np.pi/180.
                data = self._rotate(data, delta_z)
            # Dropping out data
            if self.other_params['split'] == 'train':
                # Early dropout
                if self.other_params['epochs'] < self.other_params['threshold']:
                    indices = np.random.binomial(1, self.other_params['dropout'], data.shape[0])
                    data[np.bool_(indices), 2:] = 0.
            # Preparing tensors
            I = self._map(data)
            I = torch.tensor(I, dtype=torch.float)
        return I, speed_ref, timestamp
    
    
    def _rotate(self, data, delta_z):
        ''' data has columns in the following order: x, y, vr, rcs '''
        # Rotating points
        R = np.array([[np.cos(delta_z), -np.sin(delta_z)],
                      [np.sin(delta_z), np.cos(delta_z)]])
        data[:,:2] = np.dot(data[:,:2], R)
        return data
        
    
    def _map(self, data):
        ''' data has columns in the following order: x, y, vr, rcs '''
        # Adjusting coordinates from radar to image
        px = self.map_params['px']
        py = self.map_params['py']
        rx = self.map_params['rx']
        ry = self.map_params['ry']
        in_channels = self.other_params['in_channels']
        I = np.zeros((in_channels, px, py), dtype=np.float32)
        T = np.zeros((data.shape[0], 2), dtype=np.uint16)
        # Computing mapped positions
        T[:,0] = np.uint16(np.maximum(np.minimum((px/rx) * data[:,0], px-1), 0))    
        T[:,1] = np.uint16(np.maximum(np.minimum((py/ry) * data[:,1] + (py/2), py-1), 0))
        # Mapping channels
        for channel in range(in_channels):
            I[channel, T[:,0], T[:,1]] = np.squeeze(data[:, 2 + channel])
        return I
        

    def clear(self):
        if os.path.exists(self.unified_file):
            os.remove(self.unified_file)
            print(f"Unified file for {self.other_params['split']} set deleted successfully.")
        else:
            print("The file does not exist.") 


class SequenceRadarDataset(Dataset):
    
    def __init__(self, path, map_params, augment_params, other_params):
        super(SequenceRadarDataset, self).__init__()
        self.path = path
        self.map_params = map_params
        self.augment_params = augment_params
        self.other_params = other_params
        self.speeds = []
        self.timestamps = []

        # Creating unified file to optimize RAM
        my_path = os.path.join(self.path, 'Processed', other_params['sequence'])
        os.makedirs(my_path, exist_ok=True)
        _, _, files = next(os.walk(my_path))
        files_nums = [int(f.split('.')[0].split('_')[-1]) for f in files]
        id = max(files_nums) + 1 if files_nums else 1
        self.offset_dict = {}
        self.length = 0
        self.unified_file = os.path.join(my_path, f'unified_{id}.txt')

        # Writing the data
        with open(self.unified_file, 'w') as u:
            # Reading data for selected sequence
            sequence_path = os.path.join(self.path, other_params['sequence'])
            with open(os.path.join(sequence_path, 'scenes.json')) as json_file:
                parsed = json.load(json_file)
                scenes_data = parsed['scenes']
            radar_data  = h5py.File(os.path.join(sequence_path, 'radar_data.h5'), 'r')
            odometry    = radar_data['odometry']
            radar_scans = radar_data['radar_data']

            # Getting data from the selected sensor
            with tqdm(total=len(scenes_data.items()), desc=f"Preparing '{other_params['sequence']}'") as pbar:
                # Getting data from the selected sensor
                for timestamp, info in scenes_data.items():
                    if info['sensor_id'] == other_params['sensor_id']:
                        speed = odometry[info['odometry_index']][4]
                        init, ennd = info['radar_indices']
                        scans = np.array([[range_sc, azimuth_sc, vr, rcs]
                                for timestamp, sensor_id, range_sc, azimuth_sc, rcs, vr, vr_compensated, x_cc, y_cc, x_seq, y_seq, uuid, track_id, label_id in radar_scans[init:ennd]])
                        # Computing x_rad and y_rad
                        x_rad = scans[:,0] * np.cos(scans[:,1])
                        y_rad = scans[:,0] * np.sin(scans[:,1])
                        scans[:,0] = x_rad
                        scans[:,1] = y_rad
                        # Creating unified list where scans = [x_rad, y_rad, vr, rcs]
                        data = [timestamp, scans.tolist(), speed]
                        self.offset_dict[self.length] = u.tell()
                        u.write(str(data)+'\n')
                        self.speeds.append(speed)
                        self.timestamps.append(timestamp)
                        self.length += 1
                    pbar.update(1)
        print(f"Initialization of the {other_params['sequence']} complete. A total of {len(self.offset_dict)} synchronized frames available.")
            
    def __len__(self):
        return len(self.offset_dict)
    
    def __getitem__(self, id):
        # Reading data on demand
        offset = self.offset_dict[id]
        with open(self.unified_file, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            # Processing recovered data
            curr = eval(line)
            timestamp = torch.tensor([int(curr[0])], dtype=torch.int64)
            data = np.array(curr[1])
            speed_ref = torch.tensor([curr[2]], dtype=torch.float)          
            # Augmenting data
            if self.augment_params['augment']:
                # Computing random angles to be used in the sequence
                delta_z = random.uniform(-self.augment_params['delta_z'], self.augment_params['delta_z']) * np.pi/180.
                data = self._rotate(data, delta_z)
            # Preparing tensors
            I = self._map(data)
            I = torch.tensor(I, dtype=torch.float)
        return I, speed_ref, timestamp
    
    def _rotate(self, data, delta_z):
        ''' data has columns in the following order: x, y, vr, rcs '''
        # Rotating points
        R = np.array([[np.cos(delta_z), -np.sin(delta_z)],
                      [np.sin(delta_z), np.cos(delta_z)]])
        data[:,:2] = np.dot(data[:,:2], R)
        return data
        
    def _map(self, data):
        ''' data has columns in the following order: x, y, vr, rcs '''
        # Adjusting coordinates from radar to image
        px = self.map_params['px']
        py = self.map_params['py']
        rx = self.map_params['rx']
        ry = self.map_params['ry']
        in_channels = self.other_params['in_channels']
        I = np.zeros((in_channels, px, py), dtype=np.float32)
        T = np.zeros((data.shape[0], 2), dtype=np.uint16)
        # Computing mapped positions
        T[:,0] = np.uint16(np.maximum(np.minimum((px/rx) * data[:,0], px-1), 0))    
        T[:,1] = np.uint16(np.maximum(np.minimum((py/ry) * data[:,1] + (py/2), py-1), 0))
        # Mapping channels
        for channel in range(in_channels):
            I[channel, T[:,0], T[:,1]] = np.squeeze(data[:, 2 + channel])
        return I

    def clear(self):
        if os.path.exists(self.unified_file):
            os.remove(self.unified_file)
            print(f"Unified file for {self.other_params['sequence']} set deleted successfully.")
        else:
            print("The file does not exist.") 


class SimpleRadarDataset(Dataset):
    
    def __init__(self, path, other_params):
        super(SimpleRadarDataset, self).__init__()
        self.path = path
        self.other_params = other_params
        self.speeds = []
        self.timestamps = []
        # Creating unified file to optimize RAM
        my_path = os.path.join(self.path, 'Processed', other_params['sequence'])
        os.makedirs(my_path, exist_ok=True)
        _, _, files = next(os.walk(my_path))
        files_nums = [int(f.split('.')[0].split('_')[-1]) for f in files]
        id = max(files_nums) + 1 if files_nums else 1
        self.offset_dict = {}
        self.length = 0
        self.unified_file = os.path.join(my_path, f'unified_{id}.txt')

        # Writing the data
        with open(self.unified_file, 'w') as u:
            # Reading data for selected sequence
            sequence_path = os.path.join(self.path, other_params['sequence'])
            with open(os.path.join(sequence_path, 'scenes.json')) as json_file:
                parsed = json.load(json_file)
                scenes_data = parsed['scenes']
            radar_data  = h5py.File(os.path.join(sequence_path, 'radar_data.h5'), 'r')
            odometry    = radar_data['odometry']
            radar_scans = radar_data['radar_data']
            # Getting data from the selected sensor
            with tqdm(total=len(scenes_data.items()), desc=f"Preparing '{other_params['sequence']}'") as pbar:
                for timestamp, info in scenes_data.items():
                    if info['sensor_id'] == other_params['sensor_id']:
                        speed = odometry[info['odometry_index']][4]
                        init, ennd = info['radar_indices']
                        scans = np.array([[azimuth_sc, vr]
                                for timestamp, sensor_id, range_sc, azimuth_sc, rcs, vr, vr_compensated, x_cc, y_cc, x_seq, y_seq, uuid, track_id, label_id in radar_scans[init:ennd]])
                        data = [timestamp, scans.tolist(), speed]
                        self.offset_dict[self.length] = u.tell()
                        u.write(str(data)+'\n')
                        self.speeds.append(speed)
                        self.timestamps.append(timestamp)
                        self.length += 1
                    pbar.update(1)
        print(f"Initialization of the {other_params['sequence']} complete. A total of {len(self.offset_dict)} synchronized frames available.")
        
    def __len__(self):
        return len(self.offset_dict)
    
    def __getitem__(self, id):
        # Reading data on demand
        offset = self.offset_dict[id]
        with open(self.unified_file, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            # Processing recovered data
            curr = eval(line)
            timestamp = int(int(curr[0]))
            data = np.array(curr[1])
            speed_ref = float(curr[2])          
        return timestamp, data, speed_ref 

    def clear(self):
        if os.path.exists(self.unified_file):
            os.remove(self.unified_file)
            print(f"Unified file for {self.other_params['sequence']} set deleted successfully.")
        else:
            print("The file does not exist.") 