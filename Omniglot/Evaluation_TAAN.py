"""
Author: Yingru Liu
Institute: Stony Brook University
This file contains the experiment of TAAN.
"""


import sys
sys.path.append('..')

import argparse, os
from TAAN import TAAN
from dataTools import DataLayer, fit

parser = argparse.ArgumentParser(description='Evaluate TAAN by various splitting.')
parser.add_argument('--VMTL', required=True, default='NoVMTL', metavar='VMTL', type=str,
                    help='the type of VMTL regularizations.')
parser.add_argument('--gpu', required=True, metavar='GPU', default='0', type=str,
                    help='ID of GPU.')
args = vars(parser.parse_args())

VMTL=args['VMTL']
repeat = 5
os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']

if __name__ == '__main__':
    Results = []
    txtLog = 'Results-TAAN-Omniglot-' + VMTL + '.txt'
    for l in range(4):
        Lambda = (l + 1) * 0.1
        f = open(txtLog, "a")
        f.write("---------------Lambda = {:f}.---------------\n".format(Lambda))
        f.close()
        for j in range(repeat):
            model = TAAN(Lambda=Lambda, VMTL=VMTL, dimBasis=32, lr=1e-5)
            saveto = os.path.join(VMTL, 'TAAN-Lambda-' + str(Lambda) + str(j))
            test_acc = fit(model, batchSize=5, testInterval=100, num_iter=15000, earlyStop=100, saveto=saveto)
            f = open(txtLog, "a")
            f.write("Experiment {:d}, ACC: {:.4f}.\n".format(j + 1, test_acc))
            f.close()

