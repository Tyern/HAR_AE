import os
import sys
import argparse

import numpy as np
import pandas as pd

import torch
import torchvision

import matplotlib.pyplot as plt
import lightning as L

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.FFT import spectrogram, stft

parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_type", type=str, default="lin_gyr",
                    help=f"lin_gyr or acc_gyr")

parser.add_argument("-dp", "--dataset_path", type=str, default=os.path.join("dataset", "Torso"),
                    help=f"Dataset dir should contain lin_gyr text data")

parser.add_argument("-dp2", "--dataset2_path", type=str, default=os.path.join("dataset", "validate", "Torso"),
                    help=f"Additional dataset dir for concatenating with original dataset, should contain lin_gyr text data")

parser.add_argument("--is_stack", type=int, default=0, 
                    help=f"DEPRECATED")

parser.add_argument("--save_dir", type=str, default=os.path.join("dataset", "processed_concat_data"), 
                    help=f"Output dir for saving the FFT train, val, test data after generate")

parser.add_argument("--train_size", type=float, default=0.8, 
                    help=f"train_size * max_data_use will be the true number for training")

parser.add_argument("--val_size", type=float, default=0.1, 
                    help=f"val_size * max_data_use will be the true number for validating")

parser.add_argument("--test_size", type=float, default=0.1, 
                    help=f"test_size * max_data_use will be the true number for testing")

parser.add_argument("--use_class_num", type=float, default=5, 
                    help=f"The number of class use for train|test")

parser.add_argument("--max_data_use", type=int, default=10000, 
                    help=f"Data use each class for original training, total  value of train + val + test")

parser.add_argument("--max_additional_data_use", type=int, default=5000, 
                    help=f"Data use each class for active learning training")


if os.path.basename(sys.argv[0]) == 'ipykernel_launcher.py':
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

DATA_KEY = {
    "lin_gyr": ["LAcc_x", "LAcc_y", "LAcc_z", "Gyr_x", "Gyr_y", "Gyr_z"],
    "acc_gyr": ["Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Gyr_y", "Gyr_z"],
}
data_key = DATA_KEY[args.data_type]

data_path_dict = dict(zip(data_key, 
    [os.path.join(args.dataset_path, x + ".txt") for x in data_key]))

validate_path_dict = dict(zip(data_key, 
    [os.path.join(args.dataset2_path, x + ".txt") for x in data_key]))

data_dict = {}

for data_name in data_key:
    train_data = np.loadtxt(data_path_dict[data_name])

    if args.dataset2_path != "":
        val_data = np.loadtxt(validate_path_dict[data_name])
    
        one_data = np.concatenate([train_data, val_data], axis=0)
    else:
        one_data = train_data
    data_dict[data_name] = stft(one_data, verbose=True)[1]

print("data_dict.keys()", data_dict.keys())

stack = args.is_stack

save_folder = args.save_dir
os.makedirs(save_folder)
fft_file = os.path.join(save_folder, "torso_fft.npy")
label_file = os.path.join(save_folder, "torso_label.npy")

if stack==1:
    STACK_FREQ_NUMBER = 8

    one_data = data_dict[data_name]
    data_array_list = []

    for freq_step in range(STACK_FREQ_NUMBER):
        start_idx = freq_step
        end_idx = - STACK_FREQ_NUMBER + freq_step + 1
        if end_idx == 0:
            end_idx = None

        data_array = one_data[:, start_idx: end_idx]
        print("data_array.shape", data_array.shape)
        data_array_list.append(data_array)

    one_data_stacked = np.stack(data_array_list)
    print("one_data_stacked.shape", one_data_stacked.shape)

    data_stacked_list = []

    for data_name in data_key:
        one_data = data_dict[data_name]
        data_array_list = []

        for freq_step in range(STACK_FREQ_NUMBER):
            start_idx = freq_step
            end_idx = - STACK_FREQ_NUMBER + freq_step + 1
            if end_idx == 0:
                end_idx = None

            data_array = one_data[:, start_idx: end_idx]
            data_array_list.append(data_array)

        one_data_stacked = np.stack(data_array_list)
        data_stacked_list.append(one_data_stacked)

    data_stacked = np.stack(data_stacked_list)
    print("data_stacked.shape", data_stacked.shape)

    data_stacked = data_stacked.transpose(3, 0, 1, 2)
    print("data_stacked.shape", data_stacked.shape)

    fft_data = data_stacked
    
## NO STACK
if not stack:
    data_stacked_list = []

    for data_name in data_key:
        print(data_dict[data_name].shape)
        data_stacked_list.append(data_dict[data_name])
        
    data_stacked = np.stack(data_stacked_list)
    print("data_stacked.shape", data_stacked.shape)

    data_stacked = data_stacked.transpose(2, 0, 1)
    print("data_stacked.shape", data_stacked.shape)
    
    fft_data = data_stacked    

label_train_path = os.path.join(args.dataset_path, "Label.txt")
label_train = np.loadtxt(label_train_path)

if args.dataset2_path != "":
    label_val_path = os.path.join(args.dataset2_path, "Label.txt")
    label_val = np.loadtxt(label_val_path)

    label = np.concatenate([label_train, label_val], axis=0)
else:
    label = label_train

if stack==1:
    data_label = label[: - STACK_FREQ_NUMBER + 1, :]
    print(data_label.shape)
    label_data = data_label

if not stack:
    label_data = label

random_seed = 42
L.seed_everything(random_seed)

TRAIN_SIZE, VAL_SIZE, TEST_SIZE = args.train_size, args.val_size, args.test_size

label_list = ["still", "walking", "run", "bike", "car", "bus", "train", "subway"]

activity_range = list(range(1, 8+1))

label_idx = np.array([not any(x - x[0]) for x in label_data])
print("label_idx", label_idx)

# remove 2 classes data
data_filtered = fft_data[label_idx]
label_filtered = label_data[label_idx]

label = label_filtered[:, 0]

print("label", label)
print("np.unique(label, return_counts=True", np.unique(label, return_counts=True))

# filter first k activity, ignore the others
activity_range = list(range(1, args.use_class_num+1)) 
label_filter_idx = list(map(lambda x: x in activity_range, label))

label_filtered = label[label_filter_idx]
label_filtered_data = data_filtered[label_filter_idx]

np.save(fft_file, label_filtered_data)
np.save(label_file, label_filtered)

prefix = "torso_"
postfix = "_fft"

train_val_test_save_file = os.path.join(args.save_dir, prefix + "train_val_test" + postfix + ".npz")
train_val_test_choice_idx_save_file = os.path.join(args.save_dir, prefix + "train_val_test_choice_idx" + postfix + ".npz")

fft_data = np.load(fft_file)
label = np.load(label_file)

print("np.unique(label, return_counts=True)", np.unique(label, return_counts=True))

fft_data_down_samp_list = []
label_samp_list = []
choice_samp_list = []

for i in range(1, 1+args.use_class_num):
    one_class_idx = np.where(label == i)[0]
    choice_idx_list = np.random.choice(
        one_class_idx, 
        min(args.max_data_use, len(one_class_idx)), 
        replace=False)
    
    label_samp_list.append(label[choice_idx_list])

    fft_data_down_samp_list.append(fft_data[choice_idx_list])
    choice_samp_list.append(choice_idx_list)

choice_samp = np.concatenate(choice_samp_list, axis=0)
fft_data_down_samp = np.concatenate(fft_data_down_samp_list, axis=0)
label_samp = np.concatenate(label_samp_list, axis=0)

train_val_data, test_data, train_val_label, test_label, train_val_choice, test_choice = \
    train_test_split(fft_data_down_samp, label_samp, choice_samp, test_size=TEST_SIZE, stratify=label_samp, shuffle=True)

train_data, val_data, train_label, val_label, train_choice, val_choice = \
    train_test_split(train_val_data, train_val_label, train_val_choice, test_size=VAL_SIZE / (TRAIN_SIZE + VAL_SIZE), stratify=train_val_label, shuffle=True)

print("train_data.shape, train_label.shape", train_data.shape, train_label.shape)
print("val_data.shape, val_label.shape", val_data.shape, val_label.shape)
print("test_data.shape, test_label.shape", test_data.shape, test_label.shape)

additional_choice_idx_list = np.delete(np.arange(len(label)), choice_samp, axis=0)
np.unique(label[additional_choice_idx_list], return_counts=True)

additional_fft_data_list = []
additional_label_list = []
additional_choice_list = []

for i in range(1, 1+args.use_class_num):
    one_class_idx = np.where(label[additional_choice_idx_list] == i)[0]

    choice_idx_list = np.random.choice(
        one_class_idx, 
        min(args.max_additional_data_use, len(one_class_idx)), 
        replace=False)
    
    additional_label_list.append((label[additional_choice_idx_list])[choice_idx_list])
    additional_fft_data_list.append(fft_data[additional_choice_idx_list][choice_idx_list])
    additional_choice_list.append(additional_choice_idx_list[choice_idx_list])

additional_choice = np.concatenate(additional_choice_list, axis=0)
additional_fft_data = np.concatenate(additional_fft_data_list, axis=0)
additional_label = np.concatenate(additional_label_list, axis=0)

np.savez(train_val_test_save_file, 
         train_data=train_data, 
         train_label=train_label,
         val_data=val_data,
         val_label=val_label,
         test_data=test_data,
         test_label=test_label,
         additional_data=additional_fft_data,
         additional_label=additional_label)

np.savez(train_val_test_choice_idx_save_file,
         train_choice=train_choice,
         val_choice=val_choice,
         test_choice=test_choice,
         additional_choice=additional_choice)