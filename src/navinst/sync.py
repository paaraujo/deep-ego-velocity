""" Sync radar scans with reference.
"""

import os
import json
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


def get_set(dir):
    for k, v in sets.items():
        if dir in v:
            return k
    return None


# Getting sequences
root = os.path.join('/mnt', 'sda', 'paulo', 'delta')
data = os.path.join(root, 'data', 'navinst')
_, dirs, _ = next(os.walk(data))
dirs = sorted([dir for dir in dirs if dir != 'synced'])

# Preparing destination folder
os.makedirs(os.path.join(data, 'synced'), exist_ok=True)

# Reading dataset split
with open(os.path.join(data, 'sets.json')) as json_file:
    sets = json.load(json_file)
    for key in sets.keys():
        os.makedirs(os.path.join(data, 'synced', key), exist_ok=True)

# Syncing data
sensor_ids = ['front_right', 'front_left']
with tqdm(total=len(dirs), desc='Syncing data') as pbar:
    for d in dirs:
        d_set = get_set(d)
        if d_set != None:
            # Reading reference
            ref = pd.read_csv(os.path.join(data, d, 'inspva.csv'))
            ref['speed'] = np.sqrt(np.square(ref.loc[:, 'east_velocity'].to_numpy()) +
                                np.square(ref.loc[:, 'north_velocity'].to_numpy()) +
                                np.square(ref.loc[:, 'up_velocity'].to_numpy()))
            if 'header.stamp.secs' in ref.columns:
                ref.loc[:, 'Time'] = ref.loc[:, 'header.stamp.secs'] + ref.loc[:, 'header.stamp.nsecs'] * 1e-9
            ref_time = ref['Time'].to_numpy() 
            
            for sensor_id in sensor_ids:
                local = []
                s = pd.read_csv(os.path.join(data, d, sensor_id + '.csv'))
                s.rename(columns={'speed':'vr'}, inplace=True)

                # Syncing with reference time
                grouped = s.groupby(['Time'], sort=True)    
                for t in grouped.groups.keys():
                    group = grouped.get_group(t)
                    temp = np.abs(ref_time - t)
                    id = np.argmin(temp)
                    if temp[id] < 0.02:
                        local.append([t, group.loc[:,['x','y','z','vr','snr','rcs']].values.tolist(), ref['speed'].iat[id]])
                
                # Saving text file
                with open(os.path.join(data, 'synced', d_set, d + '-' + sensor_id + '.txt'), 'w') as fp:
                    local = sorted(local, key=lambda x: x[0])
                    for dt in local:
                        fp.write(str(dt)+'\n')

        pbar.update(1)