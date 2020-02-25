"""
Author: Yingru Liu
Institute: Stony Brook University
This file contains the implementation of TAAN by pytorch.
"""


import torch, os
import torch.optim as optim
from Layers.TAL_pytorch import *

MAIN_DIR = "data"
class_annotations = os.path.join(MAIN_DIR, 'class_annotation.txt')

class TAAN(nn.Module):
    def __init__(self, tasks=50, basis=16, lr=0.01, c=0.1, regularize=None):
        super(TAAN, self).__init__()
        self.conv1 = TAL_Conv2d(in_channels=1, out_channels=6, kernel_size=3, basis=basis,
                                      tasks=tasks, regularize=regularize).cuda()
        self.conv2 = TAL_Conv2d(in_channels=6, out_channels=12, kernel_size=3, basis=basis,
                                      tasks=tasks, regularize=regularize).cuda()
        self.conv3 = TAL_Conv2d(in_channels=12, out_channels=24, kernel_size=3, basis=basis,
                                      tasks=tasks, regularize=regularize).cuda()
        self.conv4 = TAL_Conv2d(in_channels=24, out_channels=64, kernel_size=3, basis=basis,
                                      tasks=tasks, regularize=regularize).cuda()
        self.pool1 = nn.MaxPool2d(2, 2).cuda()
        self.pool2 = nn.MaxPool2d(2, 2).cuda()
        self.pool3 = nn.MaxPool2d(2, 2).cuda()
        self.pool4 = nn.MaxPool2d(2, 2).cuda()
        self.fc1 = TAL_Linear(in_features=64 * 4 * 4, out_features=512, basis=basis, tasks=tasks,
                              regularize=regularize).cuda()
        self.alphas, self.b = [self.conv1.alpha, self.conv2.alpha, self.conv3.alpha, self.conv4.alpha, self.fc1.alpha], \
                              [self.conv1.basis, self.conv2.basis, self.conv3.basis, self.conv4.basis, self.fc1.basis]
        #
        if not os.path.exists(class_annotations):
            raise ValueError("Please run the convert_path() in dataTools.py first to set up the experiment.")
        # set output layers.
        class_per_language = open(class_annotations, 'r').readlines()
        self.class_informations = []
        self.OutputLayers = []
        for line in class_per_language:
            language, num_classes = line.split(' ')
            num_classes = int(num_classes[0:-1])
            layer = nn.Linear(512, num_classes).cuda()
            layer.weight.data.normal_(0, 0.01)
            layer.bias.data.fill_(0.0)
            setattr(self, "Output_%s" % language, layer)
            self.class_informations.append((language, num_classes))
            self.OutputLayers.append(layer)
        # save the parameters.
        parameter_dict = [{"params": self.conv1.parameters(), "lr": lr}]
        parameter_dict += [{"params": self.conv2.parameters(), "lr": lr}]
        parameter_dict += [{"params": self.conv3.parameters(), "lr": lr}]
        parameter_dict += [{"params": self.conv4.parameters(), "lr": lr}]
        parameter_dict += [{"params": self.fc1.parameters(), "lr": lr}]
        parameter_dict += [{"params": self.OutputLayers[i].parameters(), "lr": 50 * lr} for i in range(tasks)]
        self.optimizer = optim.Adam(parameter_dict, lr=lr, weight_decay=0.005)
        self.criterion = nn.CrossEntropyLoss()
        """Regularization coefficent."""
        self.c = c
        """-----------"""
        self.params = parameter_dict
        self.num_tasks = tasks
        self.iter_num = 0
        return

    def forward(self, input, taskID):
        conv1 = self.pool1(self.conv1(input, taskID))
        conv2 = self.pool2(self.conv2(conv1, taskID))
        conv3 = self.pool3(self.conv3(conv2, taskID))
        conv4 = self.pool4(self.conv4(conv3, taskID))
        x = conv4.view(-1, 64 * 4 * 4)
        fc1 = self.fc1(x, taskID)
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
        """Regularization"""
        total_loss += self.conv1.regularization(self.c)
        total_loss += self.conv2.regularization(self.c)
        total_loss += self.conv3.regularization(self.c)
        total_loss += self.conv4.regularization(self.c)
        total_loss += self.fc1.regularization(self.c)
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