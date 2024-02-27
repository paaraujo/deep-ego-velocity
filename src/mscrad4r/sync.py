""" Synchronize point cloud with reference after extracting data from PCD files.
"""
import os
import json
import pickle
import navpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.signal import savgol_filter

# Compute velocities for each sequence and sync with the radar point cloud
root = os.path.join('/mnt', 'sda', 'paulo', 'delta')
dataset = os.path.join(root, 'data', 'mscrad4r')

# Looping through predefined sets
with open(os.path.join(dataset,'sets.json')) as json_file:
    sets = json.load(json_file)
    
    for key, tags in sets.items():
        total = 0
        gnss = 0
        e = []
        with tqdm(total=len(tags), leave=False, colour='yellow') as pbar:
            for entry in tags:
                # Read RTK_GPS.txt and computing forward speed
                with open(os.path.join(dataset, entry, '4_NAVIGATION', 'RTK_GPS.txt')) as rtk_gps:
                    rtk_gps_parsed = rtk_gps.readlines()
                    rtk_gps_parsed = [l.strip().split() for l in rtk_gps_parsed]
                    rtk_gps_parsed = [[l[0], float(l[1]) + float(l[2])*1e-9, float(l[3]), float(l[4]), float(l[5])] for l in rtk_gps_parsed ]
                    rtk_gps_parsed = pd.DataFrame(rtk_gps_parsed, columns=['code','timestamp','latitude','longitude','height'])
                    dt = np.diff(rtk_gps_parsed.timestamp.to_numpy())
                    ned = navpy.lla2ned(rtk_gps_parsed.latitude.to_numpy(), rtk_gps_parsed.longitude.to_numpy(), rtk_gps_parsed.height.to_numpy(),
                                        rtk_gps_parsed.latitude.iat[0], rtk_gps_parsed.longitude.iat[0], rtk_gps_parsed.height.iat[0])
                    ned_diff = np.diff(ned, axis=0)
                    speed = np.linalg.norm(ned_diff, axis=1)/dt
                    smoothed_speed = savgol_filter(speed, 51, 4)
                    rtk_gps_parsed['speed'] = [np.nan] + smoothed_speed.tolist()
                    rtk_gps_parsed.dropna(inplace=True)
                    gnss += rtk_gps_parsed.shape[0]
                    # plt.plot(speed, 'grey')
                    # plt.plot(rtk_gps_parsed.speed.to_numpy(), 'b')
                    # plt.show()

                # Read timestamp_radar.txt to sync with the reference
                with open(os.path.join(dataset, entry, '3_RADAR', 'timestamp_radar.txt')) as radar_timestamp:
                    radar_timestamp_parsed = radar_timestamp.readlines()
                    radar_timestamp_parsed = [l.strip().split() for l in radar_timestamp_parsed]
                    radar_timestamp_parsed = [[l[0], float(l[1]) + float(l[2])*1e-9] for l in radar_timestamp_parsed ]
                    radar_timestamp_parsed = pd.DataFrame(radar_timestamp_parsed, columns=['code','timestamp'])

                # Syncing Radar and Reference if speed is not NAN and point cloud has more than 3 entries
                data = []
                ref_time = radar_timestamp_parsed.timestamp.to_numpy()
                for i, t in enumerate(rtk_gps_parsed.timestamp.to_numpy()):
                    temp = np.abs(ref_time - t)
                    id = np.argmin(temp)
                    if temp[id] < 0.04:
                        speed = rtk_gps_parsed.speed.iat[i]
                        # Read converted point cloud
                        code = radar_timestamp_parsed.code.iat[id]
                        radar_pcd = pd.read_csv(os.path.join(dataset, entry, '3_RADAR', 'PCD', code + '.txt')) 
                        radar_pcd.dropna(inplace=True)
                        # Clipping data
                        radar_pcd_clipped = radar_pcd.loc[(radar_pcd.z <= 50) & (-42 <= radar_pcd.x) & (radar_pcd.x <= 42) & (-0.7 <= radar_pcd.y) & (radar_pcd.y <= 10),
                                                        ['x','y','z','alpha','beta','doppler','power']]
                        # Converting to traditional right-hand coordinate system used with radars
                        x = radar_pcd_clipped.z.to_numpy().reshape(-1,1)
                        y = -1 * radar_pcd_clipped.x.to_numpy().reshape(-1,1)
                        z = -1 * radar_pcd_clipped.y.to_numpy().reshape(-1,1)
                        array = np.hstack((x,y,z))
                        radar_pcd_clipped.loc[:, ['x','y','z']] = array
                        e.append(radar_pcd_clipped.shape[0])
                        # Adding data to the buffer
                        data.append([t, radar_pcd_clipped.loc[:,['x','y','z','alpha','beta','doppler','power']].values.tolist(), speed])
                        total += 1

                # Saving text file
                os.makedirs(os.path.join(dataset, 'synced', key), exist_ok=True)
                with open(os.path.join(dataset, 'synced', key, entry + '.pkl'), 'wb') as fp:
                    data = sorted(data, key=lambda x: x[0])
                    pickle.dump(data, fp)

                pbar.update(1)
            print(f'Set "{key}" has {total} synched frames. A total of {gnss} readings were found. Biggest scan has {max(e)} points.')
