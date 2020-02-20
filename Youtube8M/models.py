"""
Author: Yingru Liu
Institute: Stony Brook University
"""
import torch.nn as nn
from Layers.MRN_pytorch import *
from Layers.DMTRL_pytorch import *
from Layers.TAL_pytorch import *

class SingleTaskModel(nn.Module):
    def __init__(self, hidden_feature, task_classes):
        super(SingleTaskModel, self).__init__()
        #
        self.nets = []
        self.num_classes = task_classes
        for d, num_classes in enumerate(task_classes):
            net = nn.Sequential(
                nn.Linear(1152, hidden_feature),
                # nn.BatchNorm1d(hidden_feature),
                nn.ReLU(),
                nn.Linear(hidden_feature, hidden_feature),
                # nn.BatchNorm1d(hidden_feature),
                nn.ReLU(),
                nn.Linear(hidden_feature, hidden_feature),
                # nn.BatchNorm1d(hidden_feature),
                nn.ReLU(),
                nn.Linear(hidden_feature, num_classes),
            )
            self.nets.append(net)
            setattr(self, 'net_%d' % d, net)
        self.loss = nn.BCEWithLogitsLoss()
        return

    def forward(self, feature, taskID):
        return self.nets[taskID](feature)

    def return_loss(self, x, y, taskID):
        y_pred = self.forward(x, taskID)
        return self.loss(y_pred, y) * self.num_classes[taskID]


class HardSharedModel(nn.Module):
    def __init__(self, hidden_feature, task_classes):
        super(HardSharedModel, self).__init__()

        self.backend = nn.Sequential(
                nn.Linear(1152, hidden_feature),
                # nn.BatchNorm1d(hidden_feature),
                nn.ReLU(),
                nn.Linear(hidden_feature, hidden_feature),
                # nn.BatchNorm1d(hidden_feature),
                nn.ReLU(),
                nn.Linear(hidden_feature, hidden_feature),
                # nn.BatchNorm1d(hidden_feature),
                nn.ReLU(),
            )
        self.num_classes = task_classes
        self.nets = []
        for d, num_classes in enumerate(task_classes):
            net = nn.Sequential(
                nn.Linear(hidden_feature, num_classes),
            )
            self.nets.append(net)
            setattr(self, 'net_%d' % d, net)
        self.loss = nn.BCEWithLogitsLoss()
        return

    def forward(self, feature, taskID):
        hidden = self.backend(feature)
        return self.nets[taskID](hidden)

    def return_loss(self, x, y, taskID):
        y_pred = self.forward(x, taskID)
        return self.loss(y_pred, y) * self.num_classes[taskID]


class MultiRelationNet(nn.Module):
    def __init__(self, hidden_feature, task_classes, c=1e-1, regularization_task=True,
                 regularization_feature=True, regularization_input=True, update_interval=50):
        super(MultiRelationNet, self).__init__()
        self.layer1 = MRN_Linear(1152, hidden_feature, len(task_classes), dropout=False, bn=False,
                                 regularization_task=regularization_task, regularization_feature=regularization_feature,
                                 regularization_input=regularization_input, update_interval=update_interval)
        self.layer2 = MRN_Linear(hidden_feature, hidden_feature, len(task_classes), dropout=False, bn=False,
                                 regularization_task=regularization_task, regularization_feature=regularization_feature,
                                 regularization_input=regularization_input, update_interval=update_interval)
        self.layer3 = MRN_Linear(hidden_feature, hidden_feature, len(task_classes), dropout=False, bn=False,
                                 regularization_task=regularization_task, regularization_feature=regularization_feature,
                                 regularization_input=regularization_input, update_interval=update_interval)
        ###
        self.num_classes = task_classes
        self.nets = []
        for d, num_classes in enumerate(task_classes):
            net = nn.Sequential(
                nn.Linear(hidden_feature, num_classes),
            )
            self.nets.append(net)
            setattr(self, 'net_%d' % d, net)
        self.loss = nn.BCEWithLogitsLoss()
        self.c = c
        return

    def forward(self, feature, taskID):
        layer1 = self.layer1(feature, taskID)
        layer2 = self.layer2(layer1, taskID)
        layer3 = self.layer3(layer2, taskID)
        return self.nets[taskID](layer3)

    def return_loss(self, x, y, taskID):
        y_pred = self.forward(x, taskID)
        loss = self.loss(y_pred, y) * self.num_classes[taskID]
        loss += self.layer1.regularization(self.c)
        loss += self.layer2.regularization(self.c)
        loss += self.layer3.regularization(self.c)
        return loss


class SoftOrderNet(nn.Module):
    def __init__(self, hidden_feature, task_classes):
        super(SoftOrderNet, self).__init__()
        #
        self.backend = nn.Sequential(
            nn.Linear(1152, hidden_feature),
            # nn.BatchNorm1d(hidden_feature),
            nn.ReLU(),
        )
        # soft order layer.
        self.softlayers = []
        for d in range(2):
            net = nn.Sequential(
                nn.Linear(hidden_feature, hidden_feature),
                # nn.BatchNorm1d(hidden_feature),
                nn.ReLU(),
            )
            self.softlayers.append(net)
            setattr(self, 'soft_layer_%d' % d, net)
        # soft order matrix.
        self.S = nn.Parameter(0.01 * torch.randn(size=(16, 2, 2)))
        self.nets = []
        self.num_classes = task_classes
        for d, num_classes in enumerate(task_classes):
            net = nn.Sequential(
                nn.Linear(hidden_feature, num_classes),
            )
            self.nets.append(net)
            setattr(self, 'net_%d' % d, net)
        self.loss = nn.BCEWithLogitsLoss()
        return

    def forward(self, feature, taskID):
        selection = torch.softmax(self.S, dim=-1)
        layer1 = self.backend(feature)
        layer21 = self.softlayers[0](layer1)
        layer22 = self.softlayers[1](layer1)
        layer2 = selection[taskID, 0, 0] * layer21 + selection[taskID, 0, 1] * layer22
        layer31 = self.softlayers[0](layer2)
        layer32 = self.softlayers[1](layer2)
        layer3 = selection[taskID, 1, 0] * layer31 + selection[taskID, 1, 1] * layer32
        return self.nets[taskID](layer3)

    def return_loss(self, x, y, taskID):
        y_pred = self.forward(x, taskID)
        return self.loss(y_pred, y) * self.num_classes[taskID]

class DMTRL(nn.Module):
    def __init__(self, hidden_feature, task_classes, method='Tucker'):
        super(DMTRL, self).__init__()
        self.layer1 = DMTRL_Linear(1152, hidden_feature, len(task_classes), method)
        self.layer2 = DMTRL_Linear(hidden_feature, hidden_feature, len(task_classes), method)
        self.layer3 = DMTRL_Linear(hidden_feature, hidden_feature, len(task_classes), method)
        self.nets = []
        self.num_classes = task_classes
        for d, num_classes in enumerate(task_classes):
            net = nn.Sequential(
                nn.Linear(hidden_feature, num_classes),
            )
            self.nets.append(net)
            setattr(self, 'net_%d' % d, net)
        self.loss = nn.BCEWithLogitsLoss()
        return

    def forward(self, feature, taskID):
        layer1 = self.layer1(feature, taskID)
        layer2 = self.layer2(layer1, taskID)
        layer3 = self.layer3(layer2, taskID)
        return self.nets[taskID](layer3)

    def return_loss(self, x, y, taskID):
        y_pred = self.forward(x, taskID)
        return self.loss(y_pred, y) * self.num_classes[taskID]


class TAAN(nn.Module):
    def __init__(self, hidden_feature, task_classes, basis=16, c=0.1, regularization=None, RBF=0):
        super(TAAN, self).__init__()
        self.layer1 = TAL_Linear(1152, hidden_feature, basis=basis, tasks=len(task_classes),
                                 regularize=regularization, RBF=RBF)
        self.layer2 = TAL_Linear(hidden_feature, hidden_feature, basis=basis, tasks=len(task_classes),
                                 regularize=regularization, RBF=RBF)
        self.layer3 = TAL_Linear(hidden_feature, hidden_feature, basis=basis, tasks=len(task_classes),
                                 regularize=regularization, RBF=RBF)
        self.nets = []
        self.num_classes = task_classes
        for d, num_classes in enumerate(task_classes):
            net = nn.Sequential(
                nn.Linear(hidden_feature, num_classes),
            )
            self.nets.append(net)
            setattr(self, 'net_%d' % d, net)
        self.loss = nn.BCEWithLogitsLoss()
        self.c = c
        self.regularization = regularization
        self.RBF = RBF
        return

    def forward(self, feature, taskID):
        layer1 = self.layer1(feature, taskID)
        layer2 = self.layer2(layer1, taskID)
        layer3 = self.layer3(layer2, taskID)
        return self.nets[taskID](layer3)

    def return_loss(self, x, y, taskID):
        y_pred = self.forward(x, taskID)
        loss = self.loss(y_pred, y) * self.num_classes[taskID]
        loss += self.layer1.regularization(self.c)
        loss += self.layer2.regularization(self.c)
        loss += self.layer3.regularization(self.c)
        return loss

class CrossStitch(nn.Module):
    def __init__(self, hidden_feature, task_classes):
        super(CrossStitch, self).__init__()
        self.num_tasks = len(task_classes)
        self.linear1 = [nn.Linear(1152, hidden_feature) for _ in range(self.num_tasks)]
        self.linear2 = [nn.Linear(hidden_feature, hidden_feature) for _ in range(self.num_tasks)]
        self.linear3 = [nn.Linear(hidden_feature, hidden_feature) for _ in range(self.num_tasks)]
        for l, layer in enumerate(self.linear1):
            setattr(self, 'layer1_%d' % l, layer)
        for l, layer in enumerate(self.linear2):
            setattr(self, 'layer2_%d' % l, layer)
        for l, layer in enumerate(self.linear3):
            setattr(self, 'layer3_%d' % l, layer)
        #
        self.alpha1 = nn.Parameter(0.01 * torch.randn(size=(self.num_tasks, self.num_tasks)))
        self.alpha2 = nn.Parameter(0.01 * torch.randn(size=(self.num_tasks, self.num_tasks)))
        self.alpha3 = nn.Parameter(0.01 * torch.randn(size=(self.num_tasks, self.num_tasks)))
        #
        self.nets = []
        self.num_classes = task_classes
        for d, num_classes in enumerate(task_classes):
            net = nn.Sequential(
                nn.Linear(hidden_feature, num_classes),
            )
            self.nets.append(net)
            setattr(self, 'net_%d' % d, net)
        #
        self.loss = nn.BCEWithLogitsLoss()
        return

    def forward(self, feature, taskID):
        # normalize alpha.
        expALphaNorm = torch.norm(self.alpha1, dim=-1, keepdim=True).detach()
        self.alpha1.data /= expALphaNorm.data
        expALphaNorm = torch.norm(self.alpha2, dim=-1, keepdim=True).detach()
        self.alpha2.data /= expALphaNorm.data
        expALphaNorm = torch.norm(self.alpha3, dim=-1, keepdim=True).detach()
        self.alpha3.data /= expALphaNorm.data
        # forwarding.
        feature1 = torch.tensordot(self.alpha1, torch.cat([F.relu(linear(feature)).unsqueeze(0) for linear in self.linear1], 0),
                                   dims=([-1], [0]))
        feature2 = torch.tensordot(self.alpha2,
                                   torch.cat([F.relu(linear(feature1[i])).unsqueeze(0) for i, linear in enumerate(self.linear2)], 0),
                                   dims=([-1], [0]))
        feature3 = torch.tensordot(self.alpha3,
                                   torch.cat([F.relu(linear(feature2[i])).unsqueeze(0) for i, linear in enumerate(self.linear3)], 0),
                                   dims=([-1], [0]))
        return self.nets[taskID](feature3[taskID])

    def return_loss(self, x, y, taskID):
        y_pred = self.forward(x, taskID)
        return self.loss(y_pred, y) * self.num_classes[taskID]

class MMoE(nn.Module):
    def __init__(self, hidden_feature, task_classes, Expert=6):
        super(MMoE, self).__init__()
        self.num_tasks = len(task_classes)
        self.Experts = []
        for i in range(Expert):
            nets = nn.Sequential(
                nn.Linear(1152, hidden_feature),
                nn.ReLU(),
                nn.Linear(hidden_feature, hidden_feature),
                nn.ReLU(),
                nn.Linear(hidden_feature, hidden_feature),
                nn.ReLU(),
            ).cuda()
            self.Experts.append(nets)
            setattr(self, 'Expert_%d' % i, nets)
        #
        self.gates = nn.Parameter(0.01 * torch.randn(size=(self.num_tasks, Expert, 1152)))
        #
        self.nets = []
        self.num_classes = task_classes
        for d, num_classes in enumerate(task_classes):
            net = nn.Sequential(
                nn.Linear(hidden_feature, num_classes),
            )
            self.nets.append(net)
            setattr(self, 'net_%d' % d, net)
        #
        self.loss = nn.BCEWithLogitsLoss()
        return

    def forward(self, feature, taskID):
        # shape = [experts, batches, dims]
        hidden_features = torch.cat([subnetwork(feature).unsqueeze(0) for subnetwork in self.Experts], 0)
        # shape = [experts, batches, 1]
        gate = F.softmax(F.linear(feature, self.gates[taskID])).transpose(1, 0).unsqueeze(-1)
        # shape = [batches, dims]
        hidden_features = torch.sum(gate * hidden_features, dim=0)
        return self.nets[taskID](hidden_features)

    def return_loss(self, x, y, taskID):
        y_pred = self.forward(x, taskID)
        return self.loss(y_pred, y) * self.num_classes[taskID]