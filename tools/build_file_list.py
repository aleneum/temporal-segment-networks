import sys
sys.path.append('.')
import pickle
import argparse
import os.path
from pyActionRecog import parse_directory, build_split_list

parser = argparse.ArgumentParser(description="Create training and testing splits")
parser.add_argument("split_file")
parser.add_argument('--shuffle', action='store_true', default=False)
args = parser.parse_args()

data_dir = '/generated/data'
frame_dir = '/generated/frames'

with open(args.split_file) as f:
    split_tp = pickle.load(f)

# operation
f_info = parse_directory(frame_dir, 'img_', 'flow_x', 'flow_y')

print 'writing list files for training/testing'
lists = build_split_list(split_tp, f_info, 0, args.shuffle)
open(os.path.join(data_dir, 'rgb_train_split.txt'), 'w').writelines(lists[0][0])
open(os.path.join(data_dir, 'rgb_val_split.txt'), 'w').writelines(lists[0][1])
open(os.path.join(data_dir, 'flow_train_split.txt'), 'w').writelines(lists[1][0])
open(os.path.join(data_dir, 'flow_val_split.txt'), 'w').writelines(lists[1][1])
