import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="Configure models for training")
parser.add_argument("modality")
parser.add_argument("--num_gpu", default=1)
parser.add_argument("--batch_size", default=8)
args = parser.parse_args()

iter_size = int(128 / args.num_gpu / args.batch_size)
gpu_ids = ','.join([str(i) for i in range(args.num_gpu)])

with open('models/templates/tsn_{0}.prototxt.in'.format(args.modality)) as f:
    content = f.read().replace("@BATCH_SIZE@", args.batch_size)
with open('models/tsn_{0}.prototxt'.format(args.modality), 'w') as f:
    f.write(content)

with open('models/templates/tsn_{0}.prototxt.in'.format(args.modality)) as f:
    content = f.read().replace("@ITER_SIZE@", iter_size).replace("@GPU_IDS", gpu_ids)
with open('models/tsn_{0}.prototxt'.format(args.modality), 'w') as f:
    f.write(content)

cmd = [  'mpirun', '-np' ,str(args.num_gpu), 'lib/caffe-action/build/install/bin/caffe', 'train',
                '--solver=models/tsn_{1}_solver.prototxt'.format(args.modality),
                '-gpu', gpu_ids, '--weights=models/bn_inception_{0}_init.caffemodel'.format(args.modality)]

logfile = open('/generated/tsn_{0}.log'.format(args.modality), 'w')
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in proc.stdout:
    sys.stdout.write(line)
    logfile.write(line)
proc.wait()
