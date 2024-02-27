""" Main script for training and testing the proposed models.
"""

import os
root = os.path.join('/mnt', 'sda', 'paulo', 'delta')

import sys
sys.path.insert(0, os.path.join(root, 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import time
import torch
import random
import numpy as np
import pandas as pd

from matplotlib.backends.backend_agg import FigureCanvasAgg
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from utils.network import RadarNet
from dataset import RadarDataset, SequenceRadarDataset


# GLOBAL SETTINGS
plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

device = 'cuda:0'
root = os.path.join('/mnt', 'sda', 'paulo', 'delta')

seed = 17
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    # np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


def hist(data, bins=6, res=2):
    n = [0]*bins
    for x in data:
        bin_id = int(x//res)
        n[bin_id] += 1
    centres = [i*res + res/2 for i in range(bins)]
    n = [i/sum(n)*100 for i in n]
    fig, ax = plt.subplots(figsize=[5,2.5])
    ax.grid(linestyle=':')
    bars = ax.bar(centres, n, res, color='b', edgecolor='black')
    ax.set_xlabel("Speed [$m/s$]")
    ax.set_ylabel("\%")
    ax.set_xticks(range(0, bins*res, res))
    ax.set_xlim(right=bins*res-res/2)
    ax.set_axisbelow(True)
    plt.show()


def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    train_loss = 0.
    for I, speed, _ in loader:
        I = I.to(device)
        speed = speed.to(device)
        y = model(I)
        loss = loss_fn(speed, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(loader.dataset)


def valid_epoch(model, loader, loss_fn):
    model.eval()
    valid_loss = 0.
    with torch.no_grad():
        for I, speed, _ in loader:
            I = I.to(device)
            speed = speed.to(device)
            y = model(I)
            loss = loss_fn(speed, y)
            valid_loss += loss.item()
    return valid_loss / len(loader.dataset)


def test_epoch(model, loader, loss_fn):
    model.eval()
    test_loss = 0.
    with torch.no_grad():
        for I, speed, _ in loader:
            I = I.to(device)
            speed = speed.to(device)
            y = model(I)
            loss = loss_fn(speed, y)
            test_loss += loss.item()
    return test_loss / len(loader.dataset)


def preview(model, loader, path, save=False):
    model.eval()
    ref = []
    est = []
    timestamps = []
    inference = []
    with torch.no_grad():
        for I, speed, t in loader:
            tic = time.time()
            I = I.to(device)
            speed = speed.to(device)
            t = t.to(device)
            y = model(I)
            toc = time.time()
            inference.append(toc-tic)
            ref.append(speed.cpu().numpy()[0][0])
            est.append(y.cpu().numpy()[0][0])
            timestamps.append(t.cpu().numpy()[0][0])
    fig, ax = plt.subplots(figsize=[12.8, 4.8])
    canvas = FigureCanvasAgg(fig)
    ax.plot(est, 'r', linewidth=0.5, label='$\hat{\mathbf{x}}$')
    ax.plot(ref, '--b', linewidth=0.75, label='$\mathbf{x}$')
    ax.set_xlabel('Samples')
    ax.set_ylabel('m/s')
    ax.legend(loc='upper left', fancybox=False)
    canvas.draw()
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))  
    if save:
        plt.savefig(os.path.join(path, 'test_set.png'))
        timestamps = np.array(timestamps, dtype=np.float64).reshape((-1,1))
        ref = np.array(ref, dtype=np.float32).reshape((-1,1))
        est = np.array(est, dtype=np.float32).reshape((-1,1))
        df = pd.DataFrame(np.hstack((timestamps, ref, est)), columns=['Time', 'ref', 'est'])
        df.to_csv(os.path.join(path, 'processed.csv'), index=False)
        print(f'Average Inference Time: {np.array(inference).mean():.6f} secs.')
    plt.close(fig)
    return image_array


def train(tag):

    # Creating the model
    convnet_params = {
        'kernel_size': [(5, 7)],  
        'dilation': [4],     
        'stride': [1],
        'padding': [0],
        'groups': 1,
        'in_channels': 2,  
        'channels': [64, 32, 16, 8, 4, 2],  # 0.5 m/pixel
        'input_size': (1, 2, 100, 168)
    }
    linearnet_params = {
        'channels': [256, 128, 64, 32, 16, 8, 4, 2, 1],
    }
    epochs = 500
    model = RadarNet(convnet_params, linearnet_params)
    model.to(device)
    stats = summary(model, input_size=[convnet_params['input_size']], device=device)

    # Commands used for transferring learning
    # tag_weights = '2D_23_Feb_2024_18_52_52'
    # epochs = 100
    # checkpoints_path = os.path.join(root, 'checkpoints', 'mscrad4r', tag_weights, 'weights.pt')
    # weights = torch.load(checkpoints_path)
    # model.load_state_dict(weights['model_state_dict'])
    # model.conv.requires_grad_(False)
    
    # Creating datasets
    map_params = {'px':convnet_params['input_size'][2], 'py':convnet_params['input_size'][3], 'pz':convnet_params['input_size'][1], 'rx':50, 'ry':42 * 2, 'rz':10, 'groups':convnet_params['groups']}
    augment_params = {'augment': True, 'delta_x': 15.0, 'delta_y': 15.0, 'delta_z': 15.0}
    other_params = {'split': 'train', 'dropout': 0.5, 'epochs':0, 'threshold':50, 'device': device}
    train_set = RadarDataset(os.path.join(root, 'data', 'mscrad4r', 'synced'), map_params, augment_params, other_params)
    # hist(train_set.speeds, bins=14, res=2)

    map_params = {'px':convnet_params['input_size'][2], 'py':convnet_params['input_size'][3], 'pz':convnet_params['input_size'][1], 'rx':50, 'ry':42 * 2, 'rz':10, 'groups':convnet_params['groups']}    
    augment_params = {'augment': False, 'delta_x': 15.0, 'delta_y': 15.0, 'delta_z': 15.0}
    other_params = {'split': 'validation', 'device': device}
    valid_set = RadarDataset(os.path.join(root, 'data', 'mscrad4r', 'synced'), map_params, augment_params, other_params)
    # hist(valid_set.speeds, bins=14, res=2)

    map_params = {'px':convnet_params['input_size'][2], 'py':convnet_params['input_size'][3], 'pz':convnet_params['input_size'][1], 'rx':50, 'ry':42 * 2, 'rz':10, 'groups':convnet_params['groups']}
    augment_params = {'augment': False, 'delta_x': 15.0, 'delta_y': 15.0, 'delta_z': 15.0}
    other_params = {'split': 'test', 'device': device}
    test_set = RadarDataset(os.path.join(root, 'data', 'mscrad4r', 'synced'), map_params, augment_params, other_params)
    # hist(test_set.speeds, bins=14, res=2)

    # Creating dataloaders
    batch_size = 512
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=16, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=16, worker_init_fn=seed_worker, generator=g)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16, worker_init_fn=seed_worker, generator=g)
    preview_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)
    # Creating directory to store the checkpoints
    checkpoints_path = os.path.join(root, 'checkpoints', 'mscrad4r', tag)
    os.makedirs(checkpoints_path, exist_ok=True)

    # Creating tensorboard log system
    writer = SummaryWriter(os.path.join(root, 'runs', 'mscrad4r', tag))
    model_summary = repr(stats).replace(' ', '&nbsp;').replace( '\n', '<br/>')
    writer.add_text("Model", model_summary)

    # Training model    
    loss_fn = nn.HuberLoss(delta=0.1)
    # loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

    best_valid_loss = float('inf')
    for epoch in range(epochs+1):
        # Training, validating and testing model
        tic = time.time()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        train_loader.dataset.other_params['epochs'] += 1
        valid_loss = valid_epoch(model, valid_loader, loss_fn)
        test_loss  = test_epoch(model, test_loader, loss_fn)
        scheduler.step(valid_loss)
        toc = time.time()        

        # Logging data
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("Loss/test",  test_loss,  epoch)

        if epoch == 0:
            print('-' * 75)
            print('{:^7s}{:^15s}{:^17s}{:^14s}{:^10s}{:^12s}'.format('Epoch','Training Loss','Validation Loss','Testing Loss','LR','Time [min]'))
            print('-' * 75)
        print('{:^7d}{:^15.6f}{:^17.6f}{:^14.6f}{:10.4f}{:^12.2f}'.format(epoch, train_loss, valid_loss, test_loss, optimizer.state_dict()['param_groups'][0]['lr'], (toc-tic)/60))

        # Saving best weights
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch':epoch,
                'valid_loss':valid_loss,
                'lr':optimizer.state_dict()['param_groups'][0]['lr'],
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()
            }, os.path.join(root, 'checkpoints', 'mscrad4r', tag, 'weights.pt'))


        # Previewing test data
        if epoch % 20 == 0:
            path = os.path.join(root, 'checkpoints', 'mscrad4r', tag, 'Images')
            image_array = preview(model, preview_loader, path)
            writer.add_image('Testing Set', image_array[:,:,:3], epoch, dataformats='HWC')

        # Early stop
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Validation loss below threshold, stopping training.")
            break

    # Saving last info
    writer.add_hparams({
    'num_linear_layers': len(linearnet_params['channels']),
    'total_params': stats.total_params,
    'lr':optimizer.state_dict()['param_groups'][0]['lr'],
    'weight_decay':optimizer.state_dict()['param_groups'][0]['weight_decay'],
    'batch_size': batch_size,
    'epochs': epochs,
    },
    {'best_val_loss': best_valid_loss}
        )
    writer.close()

    # Cleaning files
    train_set.clear()
    valid_set.clear()
    test_set.clear()


def test(tag):
    
    # Creating the model
    convnet_params = {
        'kernel_size': [(5, 7)],  
        'dilation': [4],     
        'stride': [1],
        'padding': [0],
        'groups': 1,
        'in_channels': 2,  
        'channels': [64, 32, 16, 8, 4, 2],  # 0.5 m/pixel
        'input_size': (1, 2, 100, 168)
    }
    linearnet_params = {
        'channels': [256, 128, 64, 32, 16, 8, 4, 2, 1],
    }
    model = RadarNet(convnet_params, linearnet_params)
    model.to(device)

    # Loading pretrained weights
    checkpoints_path = os.path.join(root, 'checkpoints', 'mscrad4r', tag, 'weights.pt')
    weights = torch.load(checkpoints_path)
    model.load_state_dict(weights['model_state_dict'])

    # Saving results
    map_params = {'px':convnet_params['input_size'][2], 'py':convnet_params['input_size'][3], 'pz':convnet_params['input_size'][1], 'rx':50, 'ry':42 * 2, 'rz':10, 'groups':convnet_params['groups']}
    augment_params = {'augment': False, 'delta_x': 15.0, 'delta_y': 15.0, 'delta_z': 15.0}
    other_params = {'split': 'test', 'sequence': 'URBAN_H0', 'device': device}
    test_set = SequenceRadarDataset(os.path.join(root, 'data', 'mscrad4r', 'synced'), map_params, augment_params, other_params)
    preview_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    path = os.path.join(root, 'checkpoints', 'mscrad4r', tag, 'sequences', other_params['sequence']) 
    os.makedirs(path, exist_ok=True)
    _ = preview(model, preview_loader, path, save=True)
    test_set.clear()


if __name__ == '__main__':
    # tag = '2D_' + datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    # train(tag)
    tag = '2D_25_Feb_2024_05_16_11'
    test(tag)