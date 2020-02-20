"""
Author: Yingru Liu
Institute: Stony Brook University
This file contains the implementation of TAAN by pytorch.
"""


import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from TAL_Layers import TorchDenseTAL_APL, TorchConvTAL_APL
from TAL_Layers import Loss2Euclidean, LossCosine, LossTraceNorm
from dataTools import DataLayer, fit

class_annotations = "../data/Omniglot/class_annotation.txt"

class TAAN(nn.Module):
    def __init__(self, Lambda, VMTL='NoVMTL', num_tasks=50, dimBasis=16, lr=0.01):
        super(TAAN, self).__init__()
        self.conv1 = TorchConvTAL_APL(in_channels=1, out_channels=6, kernel_size=3, dimBasis=dimBasis,
                                      numTask=num_tasks, allTask=False).cuda()
        self.conv2 = TorchConvTAL_APL(in_channels=6, out_channels=12, kernel_size=3, dimBasis=dimBasis,
                                      numTask=num_tasks, allTask=False).cuda()
        self.conv3 = TorchConvTAL_APL(in_channels=12, out_channels=24, kernel_size=3, dimBasis=dimBasis,
                                      numTask=num_tasks, allTask=False).cuda()
        self.conv4 = TorchConvTAL_APL(in_channels=24, out_channels=64, kernel_size=3, dimBasis=dimBasis,
                                      numTask=num_tasks, allTask=False).cuda()
        self.pool1 = nn.MaxPool2d(2, 2).cuda()
        self.pool2 = nn.MaxPool2d(2, 2).cuda()
        self.pool3 = nn.MaxPool2d(2, 2).cuda()
        self.pool4 = nn.MaxPool2d(2, 2).cuda()
        self.fc1 = TorchDenseTAL_APL(dimIn=64 * 4 * 4, dimOut=512, dimBasis=dimBasis, numTask=50, allTask=False,
                                     DropOut=False).cuda()
        self.alphas, self.b = [self.conv1.alpha, self.conv2.alpha, self.conv3.alpha, self.conv4.alpha, self.fc1.alpha], \
                              [self.conv1.b, self.conv2.b, self.conv3.b, self.conv4.b, self.fc1.b]
        #
        if not os.path.exists(class_annotations):
            raise ValueError("Please run the convert_path in dataTools.py first to set up the experiment.")
        #
        class_per_language = open(class_annotations, 'r').readlines()
        self.class_informations = []
        self.OutputLayers = []
        for line in class_per_language:
            language, num_classes = line.split(' ')
            num_classes = int(num_classes[0:-1])
            layer = nn.Linear(512, num_classes).cuda()
            layer.weight.data.normal_(0, 0.01)
            layer.bias.data.fill_(0.0)
            setattr(self, "TOutput_%s" % language, layer)
            self.class_informations.append((language, num_classes))
            self.OutputLayers.append(layer)
        # save the parameters.
        parameter_dict = [{"params": self.conv1.parameters(), "lr": lr}]
        parameter_dict += [{"params": self.conv2.parameters(), "lr": lr}]
        parameter_dict += [{"params": self.conv3.parameters(), "lr": lr}]
        parameter_dict += [{"params": self.conv4.parameters(), "lr": lr}]
        parameter_dict += [{"params": self.fc1.parameters(), "lr": lr}]
        parameter_dict += [{"params": self.OutputLayers[i].parameters(), "lr": 50 * lr} for i in range(num_tasks)]
        self.optimizer = optim.Adam(parameter_dict, lr=lr, weight_decay=0.005)
        self.criterion = nn.CrossEntropyLoss()
        """VMTL Losses"""
        self.Lambda = Lambda
        self.VMTL = VMTL
        """-----------"""
        self.params = parameter_dict
        self.num_tasks = num_tasks
        self.iter_num = 0
        return

    def forward(self, input, taskID):
        conv1 = self.pool1(self.conv1([input, taskID])[0])
        conv2 = self.pool2(self.conv2([conv1, taskID])[0])
        conv3 = self.pool3(self.conv3([conv2, taskID])[0])
        conv4 = self.pool4(self.conv4([conv3, taskID])[0])
        x = conv4.view(-1, 64 * 4 * 4)
        fc1 = self.fc1([x, taskID])[0]
        predictY = self.OutputLayers[taskID](fc1)
        Y = torch.argmax(predictY, dim=-1)
        return Y, predictY, conv1, conv2, conv3, conv4, fc1

    def eval(self):
        self.conv1.eval()
        self.conv2.eval()
        self.conv3.eval()
        self.conv4.eval()
        self.fc1.eval()
        return

    def train(self, mode=True):
        self.conv1.train(mode=mode)
        self.conv2.train(mode=mode)
        self.conv3.train(mode=mode)
        self.conv4.train(mode=mode)
        self.fc1.train(mode=mode)
        return

    def train_step(self, batchX, batchY):
        self.train(True)
        # ONE STEP OF UPDATE.
        self.optimizer.zero_grad()
        predictY = [self.forward(batchX[i], taskID=i)[1] for i in range(self.num_tasks)]
        losses = [self.criterion(predictY[i], batchY[i]) for i in range(self.num_tasks)]
        classifier_loss = sum(losses) / self.num_tasks
        total_loss = classifier_loss
        """VMTL Losses"""
        if self.VMTL != 'NoVMTL' and self.Lambda > 0:
            if self.VMTL == 'LowRank':
                regularizations = [LossTraceNorm(self.alphas[i]) for i in range(len(self.alphas))]
            elif self.VMTL == 'Euclidean':
                regularizations = [Loss2Euclidean(self.alphas[i], self.b[i])[0] for i in range(len(self.alphas))]
            elif self.VMTL == 'Cosine':
                regularizations = [LossCosine(self.alphas[i], self.b[i])[0] for i in range(len(self.alphas))]
            else:
                raise ValueError("VMTL should be LowRank/Euclidean/Cosine!")
            total_loss = total_loss + self.Lambda * sum(regularizations)
        """-----------"""
        total_loss.backward()
        self.optimizer.step()
        self.iter_num += 1
        if self.iter_num % 10 == 0:
            print("Iter {:05d}, Average Cross Entropy Loss: {:.4f}".format(self.iter_num, classifier_loss.item()))
        return

    def valid_step(self, batchX, batchY):
        self.eval()
        predictY = [self.forward(batchX[i], taskID=i)[1] for i in range(self.num_tasks)]
        losses = [self.criterion(predictY[i], batchY[i]) for i in range(self.num_tasks)]
        classifier_loss = sum(losses)
        return classifier_loss.item()

    def saveModel(self, path):
        torch.save(self.state_dict(), path)
        return

    def loadModel(self, path):
        pretrain_dict = torch.load(path)
        model_dict = self.state_dict()
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)
        self.eval()
        return

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == "__main__":
    model = TAAN(Lambda=0.1, VMTL='Cosine', dimBasis=32, lr=1e-5)
    # model.loadModel("Model/model_params.pt")
    fit(model, batchSize=5, testInterval=100, num_iter=15000, earlyStop=100, saveto=None)