"""
Author: Yingru Liu
Institute: Stony Brook University
This files contain the Tools of KAL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

#------------------------------------------------Trace Norm----------------------------------------------------------#
def LossTraceNorm(matrix):
    """
    return the trace_norm regularization of a matrix.
    :param matrix:
    :return:
    """
    InnerProduct = torch.mm(matrix, matrix.t())
    return torch.sqrt(torch.diag(InnerProduct)).sum()

#--------------------------------------------Inner_Product----------------------------------------------------------#
def LossCosine(alpha, basis):
    """
    define the cosine distance loss.
    :param alpha: TxM coefficient matrix.
    :param b: M-d vector.
    :return:

    """
    T, M = alpha.size()[0], basis.size()[1]
    U = (0.4 * basis + 0.5 - Normal(0., 1.).cdf(basis)) * F.relu(basis)
    bt = basis.t()
    B, Bt = basis.repeat(M, 1), bt.repeat(1, M)
    Bmin, Bmax = torch.min(B, Bt), torch.max(B, Bt)
    W = (1 + B * Bt) * Normal(0., 1.).cdf(Bmin) + 0.4 * Bmax * torch.exp(-0.5 * (Bmin ** 2))
    U, W = torch.squeeze(U).detach(), W.detach()  # M, TxM
    alphaplus = torch.unsqueeze(alpha, 0) + torch.unsqueeze(alpha, 1)  # TxTxM
    loss = torch.matmul(alphaplus, U) + torch.mm(torch.mm(alpha, W), alpha.t()) + 0.5
    trace = torch.sqrt(torch.diag(loss)).unsqueeze(0).detach()
    loss = (loss / trace) / trace.t().detach()
    return -torch.mean(loss), loss

def LossDis(alpha, basis):
    """
    define the MTL loss.
    :param alpha: TxM coefficient matrix.
    :param b: M-d vector.
    :return:
    """
    T, M = alpha.size()[0], basis.size()[1]
    bt = basis.t()
    B, Bt = basis.repeat(M, 1), bt.repeat(1, M)
    Bmin, Bmax = torch.min(B, Bt), torch.max(B, Bt)
    W = (1 + B * Bt) * Normal(0., 1.).cdf(Bmin) + 0.4 * Bmax * torch.exp(-0.5 * (Bmin ** 2))
    alpha_minor = torch.unsqueeze(alpha, 0) - torch.unsqueeze(alpha, 1)  # TxTxM
    loss = torch.sum(torch.tensordot(alpha_minor, W, dims=([-1], [0])) * alpha_minor, dim=-1)
    return torch.mean(loss), loss


#--------------------------------------------------------------------------------------------------------------------#
class TAL_Linear(nn.Linear):
    def __init__(self, in_features, out_features, basis, tasks, bias=True, bn=False,
                 dropout=False, normalize=False, regularize=None):
        """

        :param in_features:
        :param out_features:
        :param basis:
        :param Tasks:
        :param bias:
        :param bn: [boolean] whether use bn.
        :param normalize: [boolean] whether normalize the coordinate.
        :param regularize: [string] the type of regularization (tracenorm/distance/cosine) or None.
        """
        super(TAL_Linear, self).__init__(in_features, out_features, bias)
        # extra module parameters of TAL.
        self.alpha = nn.Parameter(0.01 * torch.randn(size=(tasks, basis)))
        self.basis = nn.Parameter((torch.rand(size=(1, basis)) - 0.5).float())
        # specify the regularization method by string parameter.
        self.regularize = regularize
        #
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        self.dropout = dropout
        self.tasks = tasks
        if bn:
            self.bn = nn.BatchNorm1d(out_features)
        return

    def regularization(self, c):
        if self.regularize:
            loss = None
            if self.regularize.lower() == 'tracenorm':
                loss = LossTraceNorm(self.alpha)
            if self.regularize.lower() == 'cosine':
                loss = LossCosine(self.alpha, self.basis)[0]
            if self.regularize.lower() == 'distance':
                loss = LossDis(self.alpha, self.basis)[0]
            return c * loss
        else:
            return 0.

    def forward(self, x, task=None):
        if self.normalize:
            expALphaNorm = torch.norm(self.alpha, dim=-1, keepdim=True).detach()
            self.alpha.data /= expALphaNorm.data
        # x.size = [batch, task, out_channel] or [batch, task, in_channel]
        affine = F.linear(x, self.weight, self.bias)
        Size = affine.size()
        # Batch Normalization if used.
        if hasattr(self, 'bn'):
            if len(Size) == 2:                              # affine.size = [batch, out_channel]
                affine = self.bn(affine)
            else:                                           # affine.size = [batch, task, out_channel]
                affine = self.bn(affine.view(-1, self.out_features)).view(*Size)
        # Compute the Output.
        out = None
        X = torch.unsqueeze(affine, -1)                     # X.size = [batch, out_channel, 1] or [batch, task, in_channel, 1]
        basis = F.relu(-X + self.basis)      # basis.size = [batch, out_channel, basis] or [batch, task, out_channel, basis]
        # output the feature of a single task.
        if isinstance(task, int):
            if len(Size) > 2:
                raise ValueError("Dimensions are inconsistent. Input size should be [batch, out_channel] when task is int.")
            alpha = self.alpha.narrow(0, task, 1)           # alpha.size = [1, basis]
            out = F.relu(affine) + torch.sum(basis * alpha, -1)     # out.size = [batch, out_channel]
        # output the feature of a list of tasks.
        if isinstance(task, torch.Tensor):
            if len(Size) == 2:
                basis = basis.unsqueeze(-3)                 # basis.size = [batch, 1, output_channel, basis]
                affine = affine.unsqueeze(1)
            alpha = self.alpha.index_select(0, task)        # alpha.size = [task, basis]
            alpha = alpha.view(alpha.size()[0], 1, -1)      # alpha.size = [task, 1, basis]
            out = F.relu(affine) + torch.sum(basis * alpha, -1)     # out.size = [batch, task, out_channel]
        return F.dropout(out, p=0.5) if self.dropout else out

    def output_num(self):
        return self.out_features, ()


#--------------------------------------------------------------------------------------------------------------------#
class TAL_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, basis, tasks, stride=1, padding=0, dilation=1,
                  groups=1, bias=True, padding_mode='zeros', bn=True, dropout=False,
                  normalize=False, regularize=None):
        super(TAL_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        # extra module parameters of TAL.
        self.alpha = nn.Parameter(0.01 * torch.randn(size=(tasks, basis)))
        self.basis = nn.Parameter((torch.rand(size=(1, basis)) - 0.5).float())
        #
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.normalize = normalize
        self.dropout = dropout
        self.tasks = tasks
        self.regularize = regularize
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        return

    def regularization(self, c):
        if self.regularize:
            loss = None
            if self.regularize.lower() == 'tracenorm':
                loss = LossTraceNorm(self.alpha)
            if self.regularize.lower() == 'cosine':
                loss = LossCosine(self.alpha, self.basis)[0]
            if self.regularize.lower() == 'distance':
                loss = LossDis(self.alpha, self.basis)[0]
            return c * loss
        else:
            return 0.

    def forward(self, x, task=None):
        if self.normalize:
            expALphaNorm = torch.norm(self.alpha, dim=-1, keepdim=True).detach()
            self.alpha.data /= expALphaNorm.data
        # Conv2d operation.
        X = x if len(x.size()) == 4 else x.view(-1, x.size()[-3], x.size()[-2], x.size()[-1])
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            affine = F.conv2d(F.pad(X, expanded_padding, mode='circular'),
                              self.weight, self.bias, self.stride, 0, self.dilation, self.groups)
        else:
            affine = F.conv2d(X, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if len(x.size()) > 4:
            batch = x.size()[0]
            channel, height, width = affine.size()[-3:]
            affine = affine.view(batch, -1, channel, height, width)
        # affine.size = [batch, channel, height, width] or [batch, task, channel, height, width]
        Size = affine.size()
        # Batch Normalization if used.
        if hasattr(self, 'bn'):
            if len(Size) == 4:
                affine = self.bn(affine)
            else:
                affine = self.bn(affine.view(-1, self.out_channels, Size[-2], Size[-1])).view(*Size)
        # Compute the Output.
        out = None
        X = torch.unsqueeze(affine, -1)  # X.size = [batch, channel, height, width, 1] or [batch, task, channel, height, width, 1]
        basis = F.relu(-X + self.basis)  # basis.size = [batch, channel, height, width, 1] or [batch, task, channel, height, width, 1]
        # output the feature of a single task.
        if isinstance(task, int):
            if len(Size) > 4:
                raise ValueError(
                    "Dimensions are inconsistent. Input size should be [batch, out_channel, height, weight] when task is int.")
            alpha = self.alpha.narrow(0, task, 1)                   # alpha.size = [1, basis]
            out = F.relu(affine) + torch.sum(basis * alpha, -1)     # out.size = [batch, out_channel]
        # output the feature of a list of tasks.
        if isinstance(task, torch.Tensor):
            if len(Size) == 4:
                basis = basis.unsqueeze(1)  # basis.size = [batch, 1/task, output_channel, hight, width, basis]
                affine = affine.unsqueeze(1)
            #
            alpha = self.alpha.index_select(0, task)    # alpha.size = [task, basis]
            alpha = alpha.view(alpha.size()[0], 1, 1, 1, -1)  # alpha.size = [task, 1, 1, 1, basis]
            out = F.relu(affine) + torch.sum(basis * alpha, -1)  # out.size = [batch, task, out_channel]
        return F.dropout(out, p=0.5) if self.dropout else out

    def output_num(self):
        return self.out_channels, ()