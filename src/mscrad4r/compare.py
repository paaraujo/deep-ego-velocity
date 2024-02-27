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

from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import DataLoader
from utils.methods import KellnerMethod, AlternativeMethod
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


def plot(references, method_1, method_2, colors, labels):
    # Plotting results
    fig, ax = plt.subplots()
    ax.plot(method_1[:, 1], '.', color=colors[0], alpha=1.0, fillstyle='none', label=labels[0])
    ax.plot(method_2[:, 1], '.', color=colors[1], alpha=1.0, fillstyle='none', label=labels[1])
    ax.plot(references[:, 1], '-', color="#04080F", linewidth=1, label='Reference')
    ax.set_xlim([0, len(references)])
    ax.set_ylim([references[:,1].min()-2.5, references[:,1].max()+2.5])
    ax.set_xlabel('Samples')
    ax.set_ylabel('m/s')
    xtickslabels = ax.get_xticklabels()
    ax.set_xticklabels([str(n*100) for n in np.arange(0, len(xtickslabels), 1)])
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


def group_boxplot(kellner, alternative, proposed_2d, proposed_3d, names, colors):
    fig, ax = plt.subplots(figsize=(6.4,4.0))
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='white')
    flierprops = dict(marker='.', linestyle='none', markeredgecolor='k')
    bplot = ax.boxplot(kellner, positions=np.array(np.arange(len(kellner)))*2.5-0.85, widths=0.5, notch=False, vert=False, showfliers=True, patch_artist=True, showmeans=True, meanprops=meanpointprops, flierprops=flierprops)
    for patch in bplot['boxes']:
        patch.set_facecolor(colors[0])
    for patch in bplot['medians']:
        patch.set_color('k')
        patch.set_alpha(1)
    bplot = ax.boxplot(alternative, positions=np.array(np.arange(len(alternative)))*2.5-0.285, widths=0.5, notch=False, vert=False, showfliers=True, patch_artist=True, showmeans=True, meanprops=meanpointprops, flierprops=flierprops)
    for patch in bplot['boxes']:
        patch.set_facecolor(colors[1])
    for patch in bplot['medians']:
        patch.set_color('k')
        patch.set_alpha(1)
    bplot = ax.boxplot(proposed_2d, positions=np.array(np.arange(len(proposed_2d)))*2.5+0.285, widths=0.5, notch=False, vert=False, showfliers=True, patch_artist=True, showmeans=True, meanprops=meanpointprops, flierprops=flierprops)
    for patch in bplot['boxes']:
        patch.set_facecolor(colors[2])
    for patch in bplot['medians']:
        patch.set_color('k')
        patch.set_alpha(1)
    bplot = ax.boxplot(proposed_3d, positions=np.array(np.arange(len(proposed_3d)))*2.5+0.85, widths=0.5, notch=False, vert=False, showfliers=True, patch_artist=True, showmeans=True, meanprops=meanpointprops, flierprops=flierprops)
    for patch in bplot['boxes']:
        patch.set_facecolor(colors[3])
    for patch in bplot['medians']:
        patch.set_color('k')
        patch.set_alpha(1)
    ax.set_xlabel('Error~[m/s]', fontsize=14)
    ax.grid(linestyle=':', which='major', axis='x')
    ax.set_yticks(np.arange(0, len(names) * 2.5, 2.5), names)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.xscale('log')
    plt.tight_layout()
    plt.show()


# Transformation from reference to radar
R = Rot.from_euler('zyx', [-90, 180, 0], degrees=True)
beta = 0.  # virtually aligned
t = np.array([2.380, 0., -1.13])

# Sequences selected for test
test_set_sequences = [
    'URBAN_H0',
]

# Preparing the Kellner estimators
kellner_estimators = [
    KellnerMethod(b=t[1], l=t[0], beta=beta, trials=300),
]

# Preparing alternative estimators
alternative_estimators = [
    AlternativeMethod(R=R.as_matrix().T, trials=300),
]

# Preparing the proposed model based on the selected sensor
tags = [
    ['2D_25_Feb_2024_05_16_11', '3D_26_Feb_2024_23_17_27'],
]

# Auxiliary variables
all_kellner_errors = []
all_alternative_errors = []
all_proposed_2d_errors = []
all_proposed_3d_errors = []
all_kellner_times = []
all_alternative_times = []

for i, test_set_sequence in enumerate(test_set_sequences):
    # Preparing dataset, dataloader and estimator
    other_params = {'sequence': test_set_sequence, 'split': 'test'}
    test_set = SimpleRadarDataset(os.path.join(root, 'data', 'mscrad4r', 'synced'), other_params)
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False, shuffle=False)
    kellner_estimator = kellner_estimators[i]
    alternative_estimator = alternative_estimators[i]
    tag = tags[i]

    # Auxiliary variables
    references = []
    kellner_estimations = []
    alternative_estimations = []
    kellner_inference_times = []
    alternative_inference_times = []

    with tqdm(total=len(test_loader), desc='Running Kellner\'s Method') as pbar:
        for t, detections, ref in test_loader:
            # Converting tensors to numpy arrays
            t = t.numpy().squeeze()
            detections = detections.numpy().squeeze()
            ref = ref.numpy().squeeze()
            if detections.ndim == 2 and detections.shape[0] >= 2:
                # Cropping data to mimic a 2D radar
                mask_lower = detections[:, 2] >= -0.150
                mask_upper = detections[:, 2] <= 0.150
                mask = np.bitwise_and(mask_lower, mask_upper)
                filtered_detections = detections[mask, :]
                # Estimating v
                theta = np.deg2rad(filtered_detections[:,3])
                vr    = filtered_detections[:,5]
                tic = time.time()
                try:
                    out = kellner_estimator.estimate(vr=vr, theta=theta)
                except Exception as err:
                    # print(err)
                    out = [np.nan, np.nan]
                toc = time.time()
                # Storing data
                kellner_estimations.append([t, *out])
                kellner_inference_times.append(toc - tic)
            references.append([t, ref])
            pbar.update(1)

    with tqdm(total=len(test_loader), desc='Running Alternative Method') as pbar:
        for t, detections, ref in test_loader:
            # Converting tensors to numpy arrays
            t = t.numpy().squeeze()
            detections = detections.numpy().squeeze()
            ref = ref.numpy().squeeze()
            if detections.ndim == 2 and detections.shape[0] >= 3:
                # Estimating v
                theta = np.deg2rad(detections[:,3])
                phi   = np.deg2rad(detections[:,4])
                vr    = detections[:,5]
                tic = time.time()
                try:
                    out = alternative_estimator.estimate(vr=vr, theta=theta, phi=phi)
                except Exception as err:
                    # print(err)
                    out = np.nan
                toc = time.time()
                # Storing data
                alternative_estimations.append([t, out])
                alternative_inference_times.append(toc - tic)
            pbar.update(1)

    test_set.clear()

    # Storing data
    all_kellner_times.append(np.mean(kellner_inference_times))
    kellner_estimations = np.array(kellner_estimations)
    all_alternative_times.append(np.mean(alternative_inference_times))
    alternative_estimations = np.array(alternative_estimations)
    references = np.array(references)

    # Loading proposed model data
    proposed_2d, proposed_3d = tag
    path = os.path.join(root, 'checkpoints', 'mscrad4r', proposed_2d, 'sequences', other_params['sequence'], 'processed.csv')
    proposed_2d_df = pd.read_csv(path)
    path = os.path.join(root, 'checkpoints', 'mscrad4r', proposed_3d, 'sequences', other_params['sequence'], 'processed.csv')
    proposed_3d_df = pd.read_csv(path)

    # Plotting results
    colors = ['b', 'r']
    labels = ['Kellner\'s Method', 'Alternative Method']
    plot(references, kellner_estimations[:, 0:2], alternative_estimations, colors, labels)

    colors = ['gold', 'magenta']
    labels = ['Proposed Method (2D)', 'Proposed Method (3D)']
    plot(references, proposed_2d_df.loc[:, ['Time', 'est']].to_numpy(), proposed_3d_df.loc[:, ['Time', 'est']].to_numpy(), colors, labels) 

    # Storing errors
    common_timestamps = np.intersect1d(references[:, 0], kellner_estimations[:, 0])
    filtered_references = references[np.isin(references[:, 0], common_timestamps), :]
    filtered_kellner_estimations = kellner_estimations[np.isin(kellner_estimations[:, 0], common_timestamps), :]
    ke = np.square(filtered_references[:, 1] - filtered_kellner_estimations[:, 1])
    all_kellner_errors.extend(ke[~np.isnan(ke)].tolist())

    common_timestamps = np.intersect1d(references[:, 0], alternative_estimations[:, 0])
    filtered_references = references[np.isin(references[:, 0], common_timestamps), :]
    filtered_alternative_estimations = alternative_estimations[np.isin(alternative_estimations[:, 0], common_timestamps), :]
    ae = np.square(filtered_references[:, 1] - filtered_alternative_estimations[:, 1])
    all_alternative_errors.extend(ae[~np.isnan(ae)].tolist())

    common_timestamps = np.intersect1d(references[:, 0], proposed_2d_df.Time.to_numpy())
    filtered_references = references[np.isin(references[:, 0], common_timestamps), :]
    filtered_proposed_2d_estimations = proposed_2d_df.loc[np.isin(proposed_2d_df.Time.to_numpy(), common_timestamps), ['Time','est']].to_numpy()
    p2e = np.square(filtered_references[:, 1] - filtered_proposed_2d_estimations[:, 1])
    all_proposed_2d_errors.extend(p2e[~np.isnan(p2e)].tolist())
    
    common_timestamps = np.intersect1d(references[:, 0], proposed_3d_df.Time.to_numpy())
    filtered_references = references[np.isin(references[:, 0], common_timestamps), :]
    filtered_proposed_3d_estimations = proposed_3d_df.loc[np.isin(proposed_3d_df.Time.to_numpy(), common_timestamps), ['Time','est']].to_numpy()
    p3e = np.square(filtered_references[:, 1] - filtered_proposed_3d_estimations[:, 1])
    all_proposed_3d_errors.extend(p3e[~np.isnan(p3e)].tolist())
    
# Computing RMSE
kellner_error = np.sqrt(all_kellner_errors)
kellner_rmse = np.mean(kellner_error)
alternative_error = np.sqrt(all_alternative_errors)
alternative_rmse = np.mean(alternative_error)
proposed_2d_error = np.sqrt(all_proposed_2d_errors)
proposed_2d_rmse = np.mean(proposed_2d_error)
proposed_3d_error = np.sqrt(all_proposed_3d_errors)
proposed_3d_rmse = np.mean(proposed_3d_error)

print(f'\nStatistics:')
print(f"RMSE (Kellner): {kellner_rmse:.3f} m/s.")
print(f"RMSE (Alternative): {alternative_rmse:.3f} m/s.")
print(f"RMSE (Proposed 2D): {proposed_2d_rmse:.3f} m/s.")
print(f"RMSE (Proposed 3D): {proposed_3d_rmse:.3f} m/s.")

# Computing Mean Inference Time
kellner_MIT = np.mean(all_kellner_times)
print(f"Kellner's method mean inference time: {kellner_MIT:.6f} secs.")
alternative_MIT = np.mean(all_alternative_times)
print(f"Alternative method mean inference time: {alternative_MIT:.6f} secs.")

# Creating group boxplot
colors = ['b', 'r', 'gold', 'magenta']
names = ["Front Bumper"]
group_boxplot([all_kellner_errors], [all_alternative_errors], [all_proposed_2d_errors], [all_proposed_3d_errors], names, colors)