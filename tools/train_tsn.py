import argparse
import subprocess
import sys
import datetime

parser = argparse.ArgumentParser(description="Configure models for training")
parser.add_argument("modality")
parser.add_argument("--num_gpu", default=1, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--iter_size", type=int)
parser.add_argument("--max_iter", type=int)
parser.add_argument("--snapshot", help="Resume from solverstate")
args = parser.parse_args()

iter_size = 128 / args.num_gpu / args.batch_size if not args.iter_size else args.iter_size
gpu_ids = ','.join([str(i) for i in range(args.num_gpu)])

if not args.max_iter:
    max_iter = 4000 if args.modality == "rgb" else 18000
else:
    max_iter = args.max_iter

with open('/tsn_caffe/models/templates/tsn_{0}_train.prototxt.in'.format(args.modality)) as f:
    content = f.read().replace("@BATCH_SIZE@", str(args.batch_size))
with open('/generated/models/tsn_{0}_train.prototxt'.format(args.modality), 'w') as f:
    f.write(content)

with open('/tsn_caffe/models/templates/tsn_{0}_solver.prototxt.in'.format(args.modality)) as f:
    content = f.read().replace("@ITER_SIZE@", str(iter_size)).replace("@GPU_IDS@", gpu_ids)\
                      .replace("@MAX_ITER@", str(max_iter))
with open('/generated/models/tsn_{0}_solver.prototxt'.format(args.modality), 'w') as f:
    f.write(content)

cmd = ['mpirun', '-np' ,str(args.num_gpu), '/tsn_caffe/lib/caffe-action/build/install/bin/caffe', 'train',
       '--solver=/generated/models/tsn_{0}_solver.prototxt'.format(args.modality),
       '-gpu', gpu_ids]

if args.snapshot:
    cmd.append('--snapshot=/generated/models/{0}'.format(args.snapshot))
else:
    cmd.append('--weights=/tsn_caffe/models/bn_inception_{0}_init.caffemodel'.format(args.modality))

time_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
logfile = open('/generated/tsn_{0}_{1}_training.log'.format(time_str, args.modality), 'w')
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in proc.stdout:
    sys.stdout.write(line)
    logfile.write(line)
proc.wait()
