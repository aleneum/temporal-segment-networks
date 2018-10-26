import os
import sys
sys.path.append('/tsn_caffe')

from pyActionRec.action_classifier import ActionClassifier
from pyActionRec.anet_db import ANetDB
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("video_name", type=str)
parser.add_argument("rgb_model")
parser.add_argument("--use_flow", action="store_true", default=False)
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

VIDEO_NAME = args.video_name
USE_FLOW = args.use_flow
GPU=args.gpu

models=[]

models = [('/tsn_caffe/models/tsn_rgb_deploy.prototxt',
           '/generated/models/' + args.model,
           1.0, 0, True, 224)]


if USE_FLOW:
    models.append(('/tsn_caffe/models/tsn_flow_deploy.prototxt',
                   '/generated/models/' + args.rgb_model.replace('rgb', 'flow'),
                   0.2, 1, False, 224))

cls = ActionClassifier(models, dev_id=GPU)
rst = cls.classify(VIDEO_NAME)

scores = rst[0]
idx = np.argsort(scores)[::-1]

print('----------------Classification Results----------------------')
for i in xrange(len(idx)):
    k = idx[i]
    print(k, scores[k])
