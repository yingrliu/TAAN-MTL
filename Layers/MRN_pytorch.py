"""
Author: Yingru Liu
Institute: Stony Brook University
This files contain the Tools of MRN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable
import numpy as np


################################Copy from the official code of MRN######################################################
epsilon = 0.00001
def TensorUnfold(input_tensor, k):
    shape_tensor = list(input_tensor.size())
    num_dim = len(shape_tensor)
    permute_order = [k] + np.delete(range(num_dim), k).tolist()
    middle_result = input_tensor.permute(*permute_order)
    shape_middle = list(middle_result.size())
    result = middle_result.view([shape_middle[0], np.prod(shape_middle[1:])])
    return result


def TensorProduct(tensor1, tensor2, axes=(0, 0)):
    shape1 = list(tensor1.size())
    shape2 = list(tensor2.size())
    shape_out = np.delete(shape1, axes[0]).tolist() + np.delete(shape2, axes[1]).tolist()
    result = torch.matmul(torch.t(TensorUnfold(tensor1, axes[0])), TensorUnfold(tensor2, axes[1]))
    return result.resize_(shape_out)


def UpdateCov(weight_matrix, tensor1, tensor2):
    size0 = weight_matrix.size(0)
    final_result = torch.mm(weight_matrix.view(size0, -1), torch.t(
        torch.matmul(tensor1, torch.matmul(weight_matrix, torch.t(tensor2))).view(size0, -1)))
    return final_result + epsilon * torch.eye(final_result.size(0)).cuda()


def MultiTaskLoss(weight_matrix, tensor1, tensor2, tensor3):
    size_dim0 = weight_matrix.size(0)
    size_dim1 = weight_matrix.size(1)
    size_dim2 = weight_matrix.size(2)
    middle_result1 = torch.matmul(weight_matrix, torch.t(tensor3))
    middle_result2 = torch.matmul(tensor2, middle_result1)
    final_result = torch.matmul(tensor1, middle_result2.permute(1, 0, 2)).permute(1, 0, 2).contiguous()
    return torch.mm(weight_matrix.view(1, -1), final_result.view(-1, 1)).view(1)

#######################################################################################################################

class MRN_Linear(nn.Module):
    def __init__(self, in_features, out_features, tasks, dropout=True, bn=True, regularization_task=True,
                 regularization_feature=True, regularization_input=True, update_interval=50):
        super(MRN_Linear, self).__init__()
        self.regularization_task = regularization_task
        self.regularization_feature = regularization_feature
        self.regularization_input = regularization_input
        self.out_features = out_features
        self.num_tasks = tasks
        self.linear = [nn.Linear(in_features, out_features) for _ in range(self.num_tasks)]
        for l, layer in enumerate(self.linear):
            setattr(self, 'layer_%d' % l, layer)
        self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm1d(out_features)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        # counter to updata regularization parameters.
        self.counter = 0
        self.update_interval = update_interval
        #
        # initialize covariance matrix
        self.task_cov = torch.eye(tasks)
        self.class_cov = torch.eye(out_features)
        self.feature_cov = torch.eye(in_features)

        self.task_cov_var = Variable(self.task_cov).cuda()
        self.class_cov_var = Variable(self.class_cov).cuda()
        self.feature_cov_var = Variable(self.feature_cov).cuda()
        return

    def forward(self, input, taskID):
        Size = input.size()
        #
        if isinstance(taskID, int):
            if len(Size) > 2:
                raise ValueError("Dimensions are inconsistent. Input size should be [batch, out_channel] when task is int.")
            features = self.linear[taskID](input)
        #
        if isinstance(taskID, torch.Tensor):
            if len(Size) == 2:
                features = [self.linear[id](input).unsqueeze(1) for id in taskID]
            else:
                # input.size = [batch, task, feature]
                assert Size[1] == taskID.size()[0]              # assure that the task dim is as same as taskID.
                features = [self.linear[id](input[:, id, :]).unsqueeze(1) for id in taskID]
            features = torch.cat(features, dim=1)
        features = self.relu(features)
        # Batch Normalization if used.
        if hasattr(self, 'bn'):
            Size = features.size()
            if len(Size) == 2:  # affine.size = [batch, out_channel]
                features = self.bn(features)
            else:  # affine.size = [batch, task, out_channel]
                features = self.bn(features.view(-1, self.out_features)).view(*Size)
        if hasattr(self, 'dropout'):
            features = self.dropout(features)
        return features

    def regularization(self, c):
        self.counter += 1
        if c > 0.:
            all_weights = [self.linear[i].weight.unsqueeze(0) for i in range(self.num_tasks)]
            weights = torch.cat(all_weights, dim=0).contiguous()
            loss = MultiTaskLoss(weights, self.task_cov_var, self.class_cov_var, self.feature_cov_var)[0]
        else:
            loss = 0.
        return c * loss

    def update(self):
        all_weights = [self.linear[i].weight.unsqueeze(0) for i in range(self.num_tasks)]
        weights = torch.cat(all_weights, dim=0).contiguous()
        ############
        if self.regularization_task:
            """Update the task covariance."""
            # update cov parameters
            temp_task_cov_var = UpdateCov(weights.data, self.class_cov_var.data, self.feature_cov_var.data)
            # task covariance
            u, s, v = torch.svd(temp_task_cov_var)
            s = s.cpu().apply_(self.select_func).cuda()
            self.task_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(self.task_cov_var)
            if this_trace > 3000.0:
                self.task_cov_var = Variable(self.task_cov_var / this_trace * 3000.0).cuda()
            else:
                self.task_cov_var = Variable(self.task_cov_var).cuda()
        ###########
        if self.regularization_feature:
            """Update the out_feature_covariance."""
            temp_class_cov_var = UpdateCov(weights.data.permute(1, 0, 2).contiguous(), self.task_cov_var.data,
                                                     self.feature_cov_var.data)
            u, s, v = torch.svd(temp_class_cov_var)
            s = s.cpu().apply_(self.select_func).cuda()
            self.class_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(self.class_cov_var)
            if this_trace > 3000.0:
                self.class_cov_var = Variable(self.class_cov_var / this_trace * 3000.0).cuda()
            else:
                self.class_cov_var = Variable(self.class_cov_var).cuda()
        ###########
        if self.regularization_input:
            """Update the in_feature_covariance."""
            temp_feature_cov_var = UpdateCov(weights.data.permute(2, 0, 1).contiguous(),
                                                       self.task_cov_var.data, self.class_cov_var.data)
            u, s, v = torch.svd(temp_feature_cov_var)
            s = s.cpu().apply_(self.select_func).cuda()
            temp_feature_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(temp_feature_cov_var)
            if this_trace > 3000.0:
                self.feature_cov_var += 0.0003 * Variable(temp_feature_cov_var / this_trace * 3000.0).cuda()
            else:
                self.feature_cov_var += 0.0003 * Variable(temp_feature_cov_var).cuda()
        return

    def select_func(self, x):
        return 1./x if x > 0.1 else x