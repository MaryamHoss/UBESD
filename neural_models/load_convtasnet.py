import os, argparse
import torch
from torch.autograd import Variable

from TrialsOfNeuralVocalRecon.neural_models.convtasnet import ConvTasNet

parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
parser.add_argument('--N', default=256, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--L', default=20, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=256, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=512, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=4, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=2, type=int,
                    help='Number of speakers')
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')
args = parser.parse_args()

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'Conv_TasNet'))

ds = os.listdir(DATAPATH)
print(ds)
pth_path = os.path.join(DATAPATH, 'final.pth.tar')
onnx_path = os.path.join(DATAPATH, 'convtasnet.onnx')

model = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                   args.C, norm_type=args.norm_type, causal=args.causal,
                   mask_nonlinear=args.mask_nonlinear)
# print(model.keys())

if not os.path.isfile(onnx_path):
    checkpoint = torch.load(pth_path)
    # print(checkpoint)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])

    checkpoint['training'] = False
    # print(checkpoint.training)
    # Export the trained model to ONNX
    dummy_input = Variable(torch.randn(256, 20))
    torch.onnx.export(model, dummy_input, onnx_path)
    print('here!')


