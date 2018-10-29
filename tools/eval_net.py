import sys

sys.path.append('/tsn_caffe')
sys.path.append('/tsn_caffe/lib/caffe-action/python')

import os
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix
import pickle
import argparse

from pyActionRecog import parse_directory
from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.action_caffe import CaffeNet

parser = argparse.ArgumentParser(description="Evaluate trained model")
parser.add_argument("modality")
parser.add_argument("model")
parser.add_argument("--num_gpu", default=1, type=int)
parser.add_argument("--num_worker", default=2, type=int)
args = parser.parse_args()

modality = args.modality
gpu_list = range(args.num_gpu)
num_worker = min(args.num_worker, args.num_gpu)

# default values
data_dir = '/generated/data'
frame_dir = '/generated/frames'
num_frame_per_video = 25
net_weights = '/generated/models/' + args.model
net_proto = '/tsn_caffe/models/tsn_{0}_deploy.prototxt'.format(modality)
rgb_prefix = 'img_'
flow_x_prefix = 'flow_x_'
flow_y_prefix = 'flow_y_'

f_info = parse_directory(frame_dir, 'img_', 'flow_x', 'flow_y')
eval_video_list = []
with open('/generated/data/{0}_val_split.txt'.format(modality)) as f:
    for l in f.readlines():
        # /folder/to/frame_folder num_images class -> (frame_folder, class)
        tmp = l.split('/')[-1].split(' ')
        eval_video_list.append((tmp[0], tmp[2]))

result_name = 'test_result_{0}'.format(modality)
score_name = 'fc-action'

def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if num_worker > 1 else 1
    if gpu_list is None:
        net = CaffeNet(net_proto, net_weights, my_id-1)
    else:
        net = CaffeNet(net_proto, net_weights, gpu_list[my_id - 1])


def eval_video(video):
    global net
    label = video[1]
    vid = video[0]

    video_frame_path = f_info[0][vid]
    if modality == 'rgb':
        cnt_indexer = 1
        stack_depth = 1
    elif modality == 'flow':
        cnt_indexer = 2
        stack_depth = 5
    else:
        raise ValueError(modality)
    frame_cnt = f_info[cnt_indexer][vid]

    step = (frame_cnt - stack_depth) / (num_frame_per_video-1)
    if step > 0:
        frame_ticks = range(1, min((2 + step * (num_frame_per_video-1)), frame_cnt+1), step)
    else:
        frame_ticks = [1] * num_frame_per_video

    assert(len(frame_ticks) == num_frame_per_video)

    frame_scores = []
    for tick in frame_ticks:
        if modality == 'rgb':
            name = '{}{:05d}.jpg'.format(rgb_prefix, tick)
            frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
            scores = net.predict_single_frame([frame,], score_name, frame_size=(340, 256))
            frame_scores.append(scores)
        if modality == 'flow':
            frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
            flow_stack = []
            for idx in frame_idx:
                x_name = '{}{:05d}.jpg'.format(flow_x_prefix, idx)
                y_name = '{}{:05d}.jpg'.format(flow_y_prefix, idx)
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
            scores = net.predict_single_flow_stack(flow_stack, score_name, frame_size=(340, 256))
            frame_scores.append(scores)

    print 'video {} done'.format(vid)
    sys.stdin.flush()
    return np.array(frame_scores), label


if num_worker > 1:
    pool = multiprocessing.Pool(num_worker, initializer=build_net)
    video_scores = pool.map(eval_video, eval_video_list)
else:
    build_net()
    video_scores = map(eval_video, eval_video_list)

video_pred = [np.argmax(default_aggregation_func(x[0])) for x in video_scores]
video_labels = [int(x[1]) for x in video_scores]

cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit/cls_cnt

print cls_acc

print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)

with open('{0}/{1}.npz'.format(data_dir, result_name), 'w') as f:
    np.savez(f, scores=video_scores, labels=video_labels)



