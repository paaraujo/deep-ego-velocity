""" Dataset classes for using with Pytorch.
"""

import os
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
        selection = []
        split_path = os.path.join(self.path, other_params['split'])
        files = os.listdir(split_path)
        for f in files:
            if other_params['sensor_id'] in f:
                selection.append(f)

        # Creating unified file to optimize RAM
        processed_path = os.path.join(self.path, 'Processed', other_params['split'])
        os.makedirs(processed_path, exist_ok=True)
        _, _, files = next(os.walk(processed_path))
        files_nums = [int(f.split('.')[0].split('_')[-1]) for f in files]
        id = max(files_nums) + 1 if files_nums else 1
        self.offset_dict = {}
        self.length = 0
        self.unified_file = os.path.join(processed_path, f'unified_{id}.txt')

        # Writing the data
        with open(self.unified_file, 'w') as u:

            # Reading data for each selected sequence
            pbar = tqdm(total=len(selection), desc=f"Creating {other_params['split']} set")
            for sequence in selection:
                sequence = os.path.join(split_path, sequence)
                with open(sequence) as f:
                    # Data is a list(timestamp, ['x','y','z','azimuth','elevation','vr','snr','rcs'], speed)
                    data = f.read().splitlines()
                    for i in range(len(data)):
                        # Evaluating current line
                        curr = eval(data[i])
                        self.offset_dict[self.length] = u.tell()
                        u.write(str([curr[0], curr[1], curr[2]])+'\n')
                        speed = curr[2]
                        timestamp = curr[0]
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
            timestamp = torch.tensor([curr[0]], dtype=torch.double)
            data = np.array(curr[1])
            speed_ref = torch.tensor([curr[2]], dtype=torch.float)          
            # Augmenting data
            if self.augment_params['augment']:
                # Computing random angles to be used in the sequence
                delta_x = random.uniform(-self.augment_params['delta_x'], self.augment_params['delta_x']) * np.pi/180.
                delta_y = random.uniform(-self.augment_params['delta_y'], self.augment_params['delta_y']) * np.pi/180.
                delta_z = random.uniform(-self.augment_params['delta_z'], self.augment_params['delta_z']) * np.pi/180.
                data = self._rotate(data, delta_x, delta_y, delta_z)
            # Dropping out data
            if self.other_params['split'] == 'train':
                # Early dropout
                if self.other_params['epochs'] < self.other_params['threshold']:
                    indices = np.random.binomial(1, self.other_params['dropout'], data.shape[0])
                    data[np.bool_(indices), 3:] = 0.
            # Preparing tensors
            I = self._map(data)
            I = torch.tensor(I, dtype=torch.float)
        return I, speed_ref, timestamp
    
    def _rotate(self, data, delta_x, delta_y, delta_z):
        ''' data has columns in the following order: x, y, vr, rcs '''
        # Rotating points
        Rz = np.array([[np.cos(delta_z), -np.sin(delta_z), 0.],
                       [np.sin(delta_z), np.cos(delta_z), 0.],
                       [0., 0., 1.]])
        Ry = np.array([[np.cos(delta_y), 0., np.sin(delta_y)],
                       [0., 1., 0.],
                       [-np.sin(delta_y), 0., np.cos(delta_y)]])
        Rx = np.array([[1., 0., 0.],
                       [0., np.cos(delta_x), -np.sin(delta_x)],
                       [0., np.sin(delta_x), np.cos(delta_x)]])
        R = Rz @ Ry @ Rx
        data[:,:3] = np.dot(data[:,:3], R)
        return data
        
    def _map(self, data):
        ''' data has columns in the following order: ['x','y','z','azimuth','elevation','vr','snr','rcs'] '''
        # Adjusting coordinates from radar to image
        px = self.map_params['px']
        py = self.map_params['py']
        pz = self.map_params['pz']
        rx = self.map_params['rx']
        ry = self.map_params['ry']
        rz = self.map_params['rz']
        I = np.zeros((pz, px, py), dtype=np.float32)
        T = np.zeros((data.shape[0], 3), dtype=np.uint16)
        pz = pz // self.map_params['groups']
        # Computing mapped positions
        T[:,0] = np.uint16(np.maximum(np.minimum((px/rx) * data[:,0], px-1), 0))    
        T[:,1] = np.uint16(np.maximum(np.minimum((py/ry) * data[:,1] + (py/2), py-1), 0))
        T[:,2] = np.uint16(np.maximum(np.minimum((pz/rz) * data[:,2] + (pz/2), pz-1), 0))
        # Mapping channels
        for i in range(self.map_params['groups']):
            I[T[:,2] + pz * i, T[:,0], T[:,1]] = np.squeeze(data[:, 5 + i])
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

        # Getting sequences according to the split
        selection = []
        split_path = os.path.join(self.path, other_params['split'])
        files = os.listdir(split_path)
        for f in files:
            if other_params['sensor_id'] in f and other_params['sequence'] in f:
                selection.append(f)
        if not selection:
            print(f"Sequence '{other_params['sequence']}' for sensor '{other_params['sensor_id']}' not found in split '{other_params['split']}'.")

        # Creating unified file to optimize RAM
        processed_path = os.path.join(self.path, 'Processed', other_params['split'])
        os.makedirs(processed_path, exist_ok=True)
        _, _, files = next(os.walk(processed_path))
        files_nums = [int(f.split('.')[0].split('_')[-1]) for f in files]
        id = max(files_nums) + 1 if files_nums else 1
        self.offset_dict = {}
        self.length = 0
        self.unified_file = os.path.join(processed_path, f'unified_{id}.txt')

        # Writing the data
        with open(self.unified_file, 'w') as u:

            # Reading data for each selected sequence
            pbar = tqdm(total=len(selection), desc=f"Creating sequence {other_params['sequence']} set")
            for sequence in selection:
                sequence = os.path.join(split_path, sequence)
                with open(sequence) as f:
                    # Data is a list(timestamp, ['x','y','z','azimuth','elevation','vr','snr','rcs'], speed)
                    data = f.read().splitlines()
                    for i in range(len(data)):
                        # Evaluating current line
                        curr = eval(data[i])
                        self.offset_dict[self.length] = u.tell()
                        u.write(str([curr[0], curr[1], curr[2]])+'\n')
                        speed = curr[2]
                        timestamp = curr[0]
                        self.speeds.append(speed)
                        self.timestamps.append(timestamp)
                        self.length += 1

                pbar.update(1)
            pbar.close()
        print(f"Initialization of the sequence {other_params['sequence']} set complete. A total of {len(self.offset_dict)} synchronized frames available.")
            
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
            timestamp = torch.tensor([curr[0]], dtype=torch.double)
            data = np.array(curr[1])
            speed_ref = torch.tensor([curr[2]], dtype=torch.float)          
            # Augmenting data
            if self.augment_params['augment']:
                # Computing random angles to be used in the sequence
                delta_x = random.uniform(-self.augment_params['delta_x'], self.augment_params['delta_x']) * np.pi/180.
                delta_y = random.uniform(-self.augment_params['delta_y'], self.augment_params['delta_y']) * np.pi/180.
                delta_z = random.uniform(-self.augment_params['delta_z'], self.augment_params['delta_z']) * np.pi/180.
                data = self._rotate(data, delta_x, delta_y, delta_z)
            # Dropping out data
            if self.other_params['split'] == 'train':
                # Early dropout
                if self.other_params['epochs'] < self.other_params['threshold']:
                    indices = np.random.binomial(1, self.other_params['dropout'], data.shape[0])
                    data[np.bool_(indices), 3:] = 0.
            # Preparing tensors
            I = self._map(data)
            I = torch.tensor(I, dtype=torch.float)
        return I, speed_ref, timestamp
    
    def _rotate(self, data, delta_x, delta_y, delta_z):
        ''' data has columns in the following order: x, y, vr, rcs '''
        # Rotating points
        Rz = np.array([[np.cos(delta_z), -np.sin(delta_z), 0.],
                       [np.sin(delta_z), np.cos(delta_z), 0.],
                       [0., 0., 1.]])
        Ry = np.array([[np.cos(delta_y), 0., np.sin(delta_y)],
                       [0., 1., 0.],
                       [-np.sin(delta_y), 0., np.cos(delta_y)]])
        Rx = np.array([[1., 0., 0.],
                       [0., np.cos(delta_x), -np.sin(delta_x)],
                       [0., np.sin(delta_x), np.cos(delta_x)]])
        R = Rz @ Ry @ Rx
        data[:,:3] = np.dot(data[:,:3], R)
        return data
        
    def _map(self, data):
        ''' data has columns in the following order: ['x','y','z','azimuth','elevation','vr','snr','rcs'] '''
        # Adjusting coordinates from radar to image
        px = self.map_params['px']
        py = self.map_params['py']
        pz = self.map_params['pz']
        rx = self.map_params['rx']
        ry = self.map_params['ry']
        rz = self.map_params['rz']
        I = np.zeros((pz, px, py), dtype=np.float32)
        T = np.zeros((data.shape[0], 3), dtype=np.uint16)
        pz = pz // self.map_params['groups']
        # Computing mapped positions
        T[:,0] = np.uint16(np.maximum(np.minimum((px/rx) * data[:,0], px-1), 0))    
        T[:,1] = np.uint16(np.maximum(np.minimum((py/ry) * data[:,1] + (py/2), py-1), 0))
        T[:,2] = np.uint16(np.maximum(np.minimum((pz/rz) * data[:,2] + (pz/2), pz-1), 0))
        # Mapping channels
        for i in range(self.map_params['groups']):
            I[T[:,2] + pz * i, T[:,0], T[:,1]] = np.squeeze(data[:, 5 + i])
        return I

    def clear(self):
        if os.path.exists(self.unified_file):
            os.remove(self.unified_file)
            print(f"Unified file for {self.other_params['split']} set deleted successfully.")
        else:
            print("The file does not exist.")  


class SimpleRadarDataset(Dataset):
    
    def __init__(self, path, other_params):
        super(SimpleRadarDataset, self).__init__()
        self.path = path
        self.other_params = other_params
        self.speeds = []
        self.timestamps = []

        # Getting sequences according to the split
        selection = []
        split_path = os.path.join(self.path, other_params['split'])
        files = os.listdir(split_path)
        for f in files:
            if other_params['sensor_id'] in f and other_params['sequence'] in f:
                selection.append(f)
        if not selection:
            print(f"Sequence '{other_params['sequence']}' for sensor '{other_params['sensor_id']}' not found in split '{other_params['split']}'.")

        # Creating unified file to optimize RAM
        processed_path = os.path.join(self.path, 'Processed', other_params['split'])
        os.makedirs(processed_path, exist_ok=True)
        _, _, files = next(os.walk(processed_path))
        files_nums = [int(f.split('.')[0].split('_')[-1]) for f in files]
        id = max(files_nums) + 1 if files_nums else 1
        self.offset_dict = {}
        self.length = 0
        self.unified_file = os.path.join(processed_path, f'unified_{id}.txt')

        # Writing the data
        with open(self.unified_file, 'w') as u:

            # Reading data for each selected sequence
            pbar = tqdm(total=len(selection), desc=f"Creating sequence {other_params['sequence']} set")
            for sequence in selection:
                sequence = os.path.join(split_path, sequence)
                with open(sequence) as f:
                    # Data is a list(timestamp, ['x','y','z','azimuth','elevation','vr','snr','rcs'], speed)
                    data = f.read().splitlines()
                    for i in range(len(data)):
                        # Evaluating current line
                        curr = eval(data[i])
                        self.offset_dict[self.length] = u.tell()
                        u.write(str([curr[0], curr[1], curr[2]])+'\n')
                        speed = curr[2]
                        timestamp = curr[0]
                        self.speeds.append(speed)
                        self.timestamps.append(timestamp)
                        self.length += 1

                pbar.update(1)
            pbar.close()
        print(f"Initialization of the sequence {other_params['sequence']} set complete. A total of {len(self.offset_dict)} synchronized frames available.")
        
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
            timestamp = float(curr[0])
            data = np.array(curr[1])
            speed_ref = float(curr[2])          
        return timestamp, data, speed_ref 

    def clear(self):
        if os.path.exists(self.unified_file):
            os.remove(self.unified_file)
            print(f"Unified file for {self.other_params['sequence']} set deleted successfully.")
        else:
            print("The file does not exist.") 