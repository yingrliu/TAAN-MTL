"""
Author: Yingru Liu
Institute: Stony Brook University
"""

import sys
sys.path.append('..')

import argparse, os
import torch, ignite
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Youtube8M.models import *
from tqdm import tqdm
from random import shuffle
from sklearn.metrics import average_precision_score
from Youtube8M.torch_loader import MultiTask_Dataloader

parser = argparse.ArgumentParser(description='Evaluate TAAN by various splitting.')
# Training setting.
parser.add_argument('--batch_size', required=False, default=256, metavar='BATCH_SIZE', type=int, help='batch size.')
parser.add_argument('--max_epoch', required=False, metavar='MAX_EPOCH', default=10, type=int, help='max epoch.')
parser.add_argument('--hidden_feature', required=False, metavar='HIDDEN', default=1024, type=int, help='the dimension of hidden output.')
parser.add_argument('--lr', required=False, metavar='LR', default=1e-4, type=float, help='learning rate.')
parser.add_argument('--model', required=False, metavar='MODEL', default="STL", type=str, help='type of model.')
parser.add_argument('--saveto', required=False, metavar='SAVETO', default="", type=str, help='path to save model.')
parser.add_argument('--checkname', required=False, metavar='CHECKNAME', default="", type=str, help='path to load model.')
parser.add_argument('--early_stop', required=False, metavar='EARLY_STOP', default=3, type=int, help='epoch tolerance for early stopping .')
# Model Setting of TAAN.
parser.add_argument('--taan_constant', required=False, metavar='CONSTANT_T', default=1.0, type=float,
                    help='coefficient of TAAN regularization.')
parser.add_argument('--regularize', required=False, metavar='REGULARIZE', default=None, type=str,
                    help='coefficient of TAAN regularization.')
parser.add_argument('--basis', required=False, metavar='BASIS', default=24, type=int,
                    help='number of basis functions.')
# Model Setting of MRN.
parser.add_argument('--regularization_task', required=False, default=False, action='store_true',
                    help='tag for MRN regularization.')
parser.add_argument('--regularization_feature', required=False, default=False, action='store_true',
                    help='tag for MRN regularization.')
parser.add_argument('--regularization_input', required=False, default=False, action='store_true',
                    help='tag for MRN regularization.')
parser.add_argument('--update_interval', required=False, metavar='UPDATE_INTERVAL', default=50, type=int,
                    help='frequency to unpdate the covariance matrices.')
parser.add_argument('--mrn_constant', required=False, metavar='CONSTANT', default=1e-3, type=float,
                    help='coefficient of MRN regularization.')
# Model Setting of DMTRL.
parser.add_argument('--method', required=False, metavar='METHOD', default='Tucker', type=str,
                    help='tensor decomposition method for DMTRL.')
# Model Setting of Soft-Order.
#
parser.add_argument('--gpu', required=False, metavar='GPU', default='0', type=str, help='ID of GPU.')

def compute_mAP(y_pred, y, K):
    y_score = torch.sigmoid(y_pred).detach().cpu().numpy()
    y_np = y.cpu().numpy()
    #
    num_samples = y_score.shape[0]
    topK = np.argsort(y_score, axis=-1)[:, -K:]
    y_score_topK = np.zeros((num_samples, K), dtype=np.float32)
    y_topK = np.zeros((num_samples, K), dtype=np.int32)
    for i in range(num_samples):
        y_score_topK[i] = y_score[i, topK[i]]
        y_topK[i] = np.asarray(y_np[i, topK[i]], dtype=np.int32)
    # remove all zero cases.
    index = y_topK.sum(axis=-1) > 0
    y_topK = y_topK[index]
    y_score_topK = y_score_topK[index]
    remain_num_samples = y_topK.shape[0]
    if remain_num_samples:
        mAP = average_precision_score(y_true=y_topK, y_score=y_score_topK, average='samples') * remain_num_samples / num_samples
    else:
        mAP = 0
    return mAP, num_samples

class Trainer():
    def __init__(self, args):
        # model setting.
        self.args = args
        # load the dataset.
        _, task_classes = MultiTask_Dataloader('train', args.batch_size, shuffle=True)
        # define various model based on the type of model.
        if args.model == 'STL':
            self.model = SingleTaskModel(hidden_feature=args.hidden_feature, task_classes=task_classes).cuda()
        elif args.model == 'Hard3':
            self.model = HardSharedModel(hidden_feature=args.hidden_feature, task_classes=task_classes).cuda()
        elif args.model == 'MRN':
            self.model = MultiRelationNet(hidden_feature=args.hidden_feature, task_classes=task_classes,
                                          c=args.mrn_constant, regularization_task=args.regularization_task,
                                          regularization_feature=args.regularization_feature,
                                          regularization_input=args.regularization_input,
                                          update_interval=args.update_interval
                                          ).cuda()
        elif args.model == 'SoftOrder':
            self.model = SoftOrderNet(hidden_feature=args.hidden_feature, task_classes=task_classes).cuda()
        elif args.model == 'DMTRL':
            self.model = DMTRL(hidden_feature=args.hidden_feature, task_classes=task_classes,
                               method=args.method).cuda()
        elif args.model == 'CrossStitch':
            self.model = CrossStitch(hidden_feature=args.hidden_feature, task_classes=task_classes).cuda()
        elif args.model == 'MMoE':
            self.model = MMoE(hidden_feature=args.hidden_feature, task_classes=task_classes).cuda()
        elif args.model == 'TAAN':
            self.model = TAAN(hidden_feature=args.hidden_feature, task_classes=task_classes,
                              basis=args.basis, c=args.taan_constant, regularization=args.regularize).cuda()
        #
        self.optim = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0.005)
        # compute num of parameter.
        num_params = 0
        for param in self.model.parameters():
            size = list(param.size())
            num_params += np.prod(size)
        print('Model has %d parameters.' % num_params)
        return

    def train(self):
        worse_epochs, best_score = 0, -1
        trainSet = MultiTask_Dataloader('train', args.batch_size, shuffle=True)[0]
        validSet = MultiTask_Dataloader('valid', args.batch_size, shuffle=True)[0]
        for epoch in range(self.args.max_epoch):
            # Training.
            task_idx = list(range(16))
            len_renew = min([len(loader) - 1 for loader in trainSet])
            len_iters = max([len(loader) - 1 for loader in trainSet])
            data_iters = [iter(data_loader) for data_loader in trainSet]
            bar = tqdm(range(len_iters))
            for iter_num in bar:
                if iter_num % len_renew == 0:
                    data_iters = [iter(data_loader) for data_loader in trainSet]
                # prepare data.
                data_list = []
                for iter_ in data_iters:
                    data_list.append(iter_.next())
                #
                loss = 0
                for taskID, [feature, label] in enumerate(data_list):
                    feature, label = feature.cuda(), label.cuda()
                    loss += self.model.return_loss(feature, label, taskID)
                loss /= 16
                # update parameter.
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                bar.set_description('Epoch %03d: loss=%6.4f' % (epoch, loss.detach().cpu().numpy()))
            # Validating.
            task_idx = list(range(16))
            task_mAPs = []
            for taskID in task_idx:
                mAP, num_samples = 0, 0
                data_loader = validSet[taskID]
                bar = tqdm(data_loader)
                for feature, label in bar:
                    feature, label = feature.cuda(), label.cuda()
                    y_pred = self.model.forward(feature, taskID)
                    mAP_batch, num_samples_batch = compute_mAP(y_pred, label, K=10)
                    mAP += mAP_batch * num_samples_batch
                    num_samples += num_samples_batch
                    bar.set_description('Epoch %03d, Task %02d: mAP@10=%6.4f' % (epoch, taskID, mAP_batch))
                mAP_task = mAP / num_samples
                task_mAPs.append(mAP_task)
            # average through task.
            print(task_mAPs)
            average_mAP = np.mean(task_mAPs)
            print(average_mAP)
            ########### early stopping.
            if average_mAP > best_score:
                best_score = average_mAP
                worse_epochs = 0
                # save model.
                if self.args.saveto:
                    self.save()
            else:
                worse_epochs += 1
            if worse_epochs >= self.args.early_stop:
                break
        print("END OF TRAINING.")
        return

    def test(self):
        testSet = MultiTask_Dataloader('test', args.batch_size, shuffle=True)[0]
        self.args.checkname = self.args.saveto
        self.load()
        #
        task_idx = list(range(16))
        task_mAPs = []
        for taskID in task_idx:
            mAP, num_samples = 0, 0
            data_loader = testSet[taskID]
            bar = tqdm(data_loader)
            for feature, label in bar:
                feature, label = feature.cuda(), label.cuda()
                y_pred = self.model.forward(feature, taskID)
                mAP_batch, num_samples_batch = compute_mAP(y_pred, label, K=10)
                mAP += mAP_batch * num_samples_batch
                num_samples += num_samples_batch
                bar.set_description('Test Step, Task %02d: mAP@10=%6.4f' % (taskID, mAP_batch))
            mAP_task = mAP / num_samples
            task_mAPs.append(mAP_task)
        # average through task.
        print(task_mAPs)
        average_mAP = np.mean(task_mAPs)
        print(average_mAP)
        if self.args.saveto:
            if not os.path.exists(self.args.saveto):
                os.makedirs(self.args.saveto)
            path1 = os.path.join(self.args.saveto, self.args.model+'-task_mAP%K.txt')
            np.savetxt(path1, np.asarray(task_mAPs))
            path2 = os.path.join(self.args.saveto, self.args.model + '-average_mAP%K.txt')
            np.savetxt(path2, average_mAP)
        return

    def save(self):
        if self.args.saveto:
            if not os.path.exists(self.args.saveto):
                os.makedirs(self.args.saveto)
            PATH = os.path.join(self.args.saveto, self.args.model+'.pth')
            torch.save(self.model.state_dict(), PATH)
        return

    def load(self):
        if self.args.checkname:
            print("Load trained model.")
            PATH = os.path.join(self.args.checkname, self.args.model + '.pth')
            self.model.load_state_dict(torch.load(PATH))
        return

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    trainer = Trainer(args)
    trainer.train()
    print("START TESTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    trainer.test()