""" Compare proposed model to benchmarks.
"""

import os
root = os.path.join('/mnt', 'sda', 'paulo', 'delta')

import sys
sys.path.insert(0, os.path.join(root, 'src'))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from utils.methods import KellnerMethod
from dataset import SimpleRadarDataset
from math import radians
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 18
})


def plot(references, kellner_estimations, proposed_estimations):
    # Plotting results
    fig, ax = plt.subplots()
    ax.plot(kellner_estimations, '.', color='#70AE6E', alpha=1.0, fillstyle='none', label='Kellner\'s Method')
    ax.plot(proposed_estimations, '.', color='#DC493A', alpha=1.0, fillstyle='none', label='Proposed Method')
    ax.plot(references, '-', color="#04080F", linewidth=1, label='Reference')
    ax.set_xlim([0, len(references)])
    ax.set_xlabel('Samples')
    ax.set_ylabel('m/s')
    plt.grid(linestyle=':')
    # plt.title(other_params['sequence'])
    # plt.legend(fancybox=False, markerscale=2, fontsize=14, shadow=False)
    plt.tight_layout()
    plt.show()


def boxplot(unified, names, colors):
    fig, ax = plt.subplots(figsize=(6.4,2.0))
    bplot = ax.boxplot(unified, labels=names, widths=0.5, notch=False, vert=False, showfliers=True, patch_artist=True, flierprops={'marker':'.','linestyle':'none','markeredgecolor':'k'})
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)  # 'lightgray'
    for patch in bplot['medians']:
        patch.set_color('k')
    ax.set_xlabel('Error~[m/s]', fontsize=14)
    ax.grid(linestyle=':', which='major', axis='x')
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.xscale('log')
    plt.tight_layout()
    plt.show()


def group_boxplot(benchmark, proposed, names, colors):
    fig, ax = plt.subplots(figsize=(6.4,4.0))
    bplot = ax.boxplot(benchmark, positions=np.array(np.arange(len(benchmark)))*2.0-0.35, widths=0.5, notch=False, vert=False, showfliers=True, patch_artist=True, flierprops={'marker':'.','linestyle':'none','markeredgecolor':'k'})
    for patch in bplot['boxes']:
        patch.set_facecolor(colors[0])
    for patch in bplot['medians']:
        patch.set_color('k')
    bplot = ax.boxplot(proposed, positions=np.array(np.arange(len(proposed)))*2.0+0.35, widths=0.5, notch=False, vert=False, showfliers=True, patch_artist=True, flierprops={'marker':'.','linestyle':'none','markeredgecolor':'k'})
    for patch in bplot['boxes']:
        patch.set_facecolor(colors[1])
    for patch in bplot['medians']:
        patch.set_color('k')
    ax.set_xlabel('Error~[m/s]', fontsize=14)
    ax.grid(linestyle=':', which='major', axis='x')
    ax.set_yticks(np.arange(0, len(names) * 2, 2), names)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.xscale('log')
    plt.tight_layout()
    plt.show()


def violinplot(unified, names, colors):
    fig, ax = plt.subplots(figsize=(6.4,2.4))
    parts = ax.violinplot(unified, showmeans=True, showmedians=False, showextrema=False, vert=False)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    xy = [[l.vertices[0,0], l.vertices[:,1].mean()] for l in parts['cmeans'].get_paths()]
    xy = np.array(xy)
    ax.scatter(xy[:,0], xy[:,1],s=121, c="white", edgecolors='black', marker="o", zorder=3)
    parts['cmeans'].set_visible(False)
    ax.set_xlabel('Error~[m/s]', fontsize=14)
    ax.set_yticks([y + 1 for y in range(unified.shape[1])], labels=names)
    ax.grid(linestyle=':', which='major', axis='x')
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.xscale('log')
    plt.tight_layout()
    plt.show()


# Sequences selected for test
test_set_sequences = [
    'sequence_5', 
    'sequence_19',
    'sequence_31',
    'sequence_48',
    'sequence_53',
    'sequence_58',
    'sequence_68',
    'sequence_73',
    'sequence_79',
    'sequence_93',
    'sequence_99',
    'sequence_147',
    'sequence_148',
    'sequence_153'
]

# Preparing the Kellner estimator based on the selected sensor
kellner_estimators = [
    KellnerMethod(b=-0.8737, l=3.6493, beta=radians(-85)),  # Sensor 1
    KellnerMethod(b=-0.6958, l=3.8615, beta=radians(-25)),  # Sensor 2
    KellnerMethod(b=0.6900, l=3.8649, beta=radians(25)),    # Sensor 3
    KellnerMethod(b=0.8730, l=3.6498, beta=radians(85))     # Sensor 4
]

# Preparing the proposed model based on the selected sensor
tags = [
    'A_001',
    'B_001',
    'C_001',
    'D_001'
]

sensor_ids = [4, 3, 2, 1]

all_kellner_errors = []
all_proposed_errors = []

for sensor_id in sensor_ids:
    kellner_errors = []
    kellner_time = []
    proposed_errors = []

    print(f'\n==== Sensor ID: {sensor_id} ====\n')
    for test_set_sequence in test_set_sequences:
        # Preparing dataset, dataloader and estimator
        other_params = {'sequence': test_set_sequence, 'sensor_id': sensor_id}
        test_set = SimpleRadarDataset(os.path.join(root, 'data', 'radarscenes', 'data'), other_params)
        test_loader = DataLoader(test_set, batch_size=1, drop_last=False, shuffle=False)
        kellner_estimator = kellner_estimators[other_params['sensor_id'] - 1]
        tag = tags[other_params['sensor_id'] - 1]

        # Auxiliary variables
        timestamps = []
        references = []
        kellner_estimations = []
        proposed_estimations = []
        kellner_inference_times = []

        with tqdm(total=len(test_loader), desc='Running Kellner\'s Method') as pbar:
            for t, detections, ref in test_loader:
                # Converting tensors to numpy arrays
                t = t.numpy().squeeze()
                detections = detections.numpy().squeeze()
                ref = ref.numpy().squeeze()
                # Estimating v and omega
                theta = detections[:,0]
                vr    = detections[:,1]
                tic = time.time()
                try:
                    out = kellner_estimator.estimate(vr=vr, theta=theta)
                except Exception as err:
                    print(err)
                    out = [np.nan, np.nan]
                toc = time.time()
                # Storing data
                kellner_estimations.append(out)
                kellner_inference_times.append(toc - tic)
                timestamps.append(t)
                references.append(ref)
                pbar.update(1)

        test_set.clear()
        kellner_time.append(np.mean(kellner_inference_times))
        kellner_estimations = np.array(kellner_estimations)

        # Loading proposed model data
        path = os.path.join(root, 'checkpoints', 'radarscenes', tag, 'sequences', other_params['sequence'], 'processed.csv')
        proposed_df = pd.read_csv(path)

        # Plotting results
        # plot(references, kellner_estimations[:, 0], proposed_df.loc[:, 'est'])
        # print(f"Kellner's method mean inference time: {np.mean(kellner_inference_times):.6f} secs.\n")

        # Storing errors
        kellner_errors.extend(np.square(references - kellner_estimations[:, 0]).tolist())
        proposed_errors.extend(np.square(references - proposed_df.loc[:, 'est']).tolist())

    # Computing RMSE
    kellner_error = np.sqrt(kellner_errors)
    kellner_rmse = np.mean(kellner_error)
    proposed_error = np.sqrt(proposed_errors)
    proposed_rmse = np.mean(proposed_error)
    
    print(f'\nStatistics:')
    print(f"RMSE (Kellner): {kellner_rmse:.3f} m/s.")
    print(f"RMSE (Proposed): {proposed_rmse:.3f} m/s.")

    # Computing Mean Inference Time
    kellner_MIT = np.mean(kellner_time)
    print(f"Kellner's method mean inference time: {kellner_MIT:.6f} secs.")

    # Storing data for global plots
    all_kellner_errors.append(kellner_error)
    all_proposed_errors.append(proposed_error)

    # Local plots
    # unified = np.concatenate((np.sqrt(kellner_errors).reshape(-1,1), np.sqrt(proposed_errors).reshape(-1,1)), axis=1)
    # colors = ['#70AE6E', '#DC493A']
    # names = ["Kellner's Method", "Proposed Method"]
    # boxplot(unified, names, colors)
    # violinplot(unified, names, colors)

# Creating group boxplot
colors = ['#70AE6E', '#DC493A']
names = [f"Sensor {i}" for i in sensor_ids]
group_boxplot(all_kellner_errors, all_proposed_errors, names, colors)