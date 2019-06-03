import os
import argparse
import logging
from types import MethodType


parser = argparse.ArgumentParser(description='Mxnet AlphaPose')



"----------------------------- General options -----------------------------"

parser.add_argument('--load_from_pyt', default=False, dest='load_from_pyt',
                    help='Load pretrained model from PyTorch model', action='store_true')


"----------------------------- Model options -----------------------------"
parser.add_argument('--loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')
parser.add_argument('--try_loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')
# parser.add_argument('--nClasses', default=17, type=int,
#                     help='Number of output channel')
parser.add_argument('--nClasses', default=33, type=int,
                    help='Number of output channel')
parser.add_argument('--pre_resnet', default=True, dest='pre_resnet',
                    help='Use pretrained resnet', action='store_true')
parser.add_argument('--dtype', default='float32', type=str,
                    help='Model dtype')
parser.add_argument('--use_pretrained_base', default=True, dest='use_pretrained_base',
                    help='Use pretrained base', action='store_true')
parser.add_argument('--det_model', default='frcnn', type=str,
                    help='Det model name')
parser.add_argument('--syncbn', default=False, dest='syncbn',
                    help='Use Sync BN', action='store_true')

"----------------------------- Data options -----------------------------"
parser.add_argument('--inputResH', default=256, type=int,
                    help='Input image height')
parser.add_argument('--inputResW', default=192, type=int,
                    help='Input image width')
parser.add_argument('--scale', default=0.3, type=float,
                    help='Degree of scale augmentation')
parser.add_argument('--rotate', default=40, type=float,
                    help='Degree of rotation augmentation')
parser.add_argument('--hmGauss', default=1, type=int,
                    help='Heatmap gaussian size')

"----------------------------- Log options -----------------------------"
parser.add_argument('--logging-file', type=str, default='training.log',
                    help='name of training log file')
parser.add_argument('--board', default=True, dest='board',
                    help='Logging with tensorboard', action='store_true')
parser.add_argument('--visdom', default=False, dest='visdom',
                    help='Visualize with visdom', action='store_true')
parser.add_argument('--map', default=True, dest='map',
                    help='Evaluate mAP per epoch', action='store_true')

"----------------------------- Detection options -----------------------------"
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--mode', dest='mode',
                    help='detection mode, fast/normal/accurate', default="normal")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--inp_dim', dest='inp_dim', type=str, default='608',
                    help='inpdim')
parser.add_argument('--conf', dest='confidence', type=float, default=0.1,
                    help='bounding box confidence threshold')
parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.6,
                    help='bounding box nms threshold')
parser.add_argument('--save_img', default=True, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--detbatch', type=int, default=1,
                    help='detection batch size')
# parser.add_argument('--detbatch', type=int, default=2,
#                     help='detection batch size')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size')

opt = parser.parse_args()

opt.outputResH = opt.inputResH // 4
opt.outputResW = opt.inputResW // 4


