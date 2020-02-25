"""
Author: Yingru Liu
Institute: Stony Brook University
This file contains the experiment of TAAN.
"""


import sys
sys.path.append('..')

import argparse, os
from Omniglot.TAAN import TAAN
from Omniglot.dataTools import fit

parser = argparse.ArgumentParser(description='Evaluate TAAN by various splitting.')
parser.add_argument('--regularize', required=False, metavar='REGULARIZE', default=None, type=str,
                    help='coefficient of TAAN regularization.')
parser.add_argument('--gpu', required=False, metavar='GPU', default='0', type=str,
                    help='ID of GPU.')
args = vars(parser.parse_args())

REG=args['regularize']
repeat = 5
os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']

if __name__ == '__main__':
    Results = []
    txtLog = 'Results-TAAN-Omniglot-' + str(REG) + '.txt'
    for l in range(4):
        Lambda = (l + 1) * 0.1
        f = open(txtLog, "a")
        f.write("---------------Lambda = {:f}.---------------\n".format(Lambda))
        f.close()
        for j in range(repeat):
            model = TAAN(regularize=REG, basis=32, c=Lambda, lr=1e-5)
            saveto = os.path.join(REG, 'TAAN-Lambda-' + str(Lambda) + str(j))
            test_acc = fit(model, batchSize=5, testInterval=100, num_iter=15000, earlyStop=100, saveto=saveto)
            f = open(txtLog, "a")
            f.write("Experiment {:d}, ACC: {:.4f}.\n".format(j + 1, test_acc))
            f.close()

