"""
Author: Yingru Liu
Institute: Stony Brook University
This files contain the Tools of DMTRL.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg.interpolative import svd

"""This part is refered from the official code in https://github.com/wOOL/DMTRL/blob/master/tensor_toolbox_yyang.py"""
#########################################Numpy Operation####################################################
# numpy SVD.
def my_svd(A, eps_or_k=0.01):
    if A.dtype != np.float64:
        A = A.astype(np.float64)
    U, S, V = svd(A, eps_or_k, rand=False)
    return U, S, V.T


def t_unfold(A, k):
    A = np.transpose(A, np.hstack([k, np.delete(np.arange(A.ndim), k)]))
    A = np.reshape(A, [A.shape[0], np.prod(A.shape[1:])])

    return A


def t_dot(A, B, axes=(-1, 0)):
    return np.tensordot(A, B, axes)


def tt_dcmp(A, eps_or_k=0.01):
    d = A.ndim
    n = A.shape

    max_rank = [min(np.prod(n[:i + 1]), np.prod(n[i + 1:])) for i in range(d - 1)]

    if np.any(np.array(eps_or_k) > np.array(max_rank)):
        raise ValueError('the rank is up to %s' % str(max_rank))

    if not isinstance(eps_or_k, list):
        eps_or_k = [eps_or_k] * (d - 1)

    r = [1] * (d + 1)

    TT = []
    C = A.copy()

    for k in range(d - 1):
        C = C.reshape((r[k] * n[k], C.size // (r[k] * n[k])))
        (U, S, V) = my_svd(C, eps_or_k[k])
        r[k + 1] = U.shape[1]
        TT.append(U[:, :r[k + 1]].reshape((r[k], n[k], r[k + 1])))
        C = np.dot(np.diag(S[:r[k + 1]]), V[:r[k + 1], :])
    TT.append(C.reshape(r[k + 1], n[k + 1], 1))

    return TT


def tucker_dcmp(A, eps_or_k=0.01):
    d = A.ndim
    n = A.shape

    max_rank = list(n)

    if np.any(np.array(eps_or_k) > np.array(max_rank)):
        raise ValueError('the rank is up to %s' % str(max_rank))

    if not isinstance(eps_or_k, list):
        eps_or_k = [eps_or_k] * d

    U = [my_svd(t_unfold(A, k), eps_or_k[k])[0] for k in range(d)]
    S = A
    for i in range(d):
        S = t_dot(S, U[i], (0, 0))

    return U, S


def tt_cnst(A):
    S = A[0]
    for i in range(len(A) - 1):
        S = t_dot(S, A[i + 1])

    return np.squeeze(S, axis=(0, -1))


def tucker_cnst(U, S):
    for i in range(len(U)):
        S = t_dot(S, U[i], (0, 1))

    return S

##############################################################################################################
def TensorUnfold(A, k):
    tmp_arr = np.arange(len(A.size()))
    A = A.permute(*([tmp_arr[k]] + np.delete(tmp_arr, k).tolist()))
    shapeA = A.size()
    A = A.contiguous().view(*([shapeA[0], np.prod(shapeA[1:])]))
    return A


def TensorProduct(A, B, axes=(-1, 0)):
    shapeA = A.size()
    shapeB = B.size()
    shapeR = np.delete(shapeA, axes[0]).tolist() + np.delete(shapeB, axes[1]).tolist()
    result = torch.mm(torch.t(TensorUnfold(A, axes[0])), TensorUnfold(B, axes[1]))
    return result.view(*shapeR)


def TTTensorProducer(A):
    S = A[0]
    for i in range(len(A) - 1):
        S = TensorProduct(S, A[i + 1])

    return S.squeeze(0).squeeze(-1)


def TuckerTensorProducer(U, S):
    for i in range(len(U)):
        S = TensorProduct(S, U[i], (0, 1))
    return S

"""Core Component."""
def TensorProducer(X, method, eps_or_k=10, datatype=np.float32):
    param_dict = {}
    if method == 'Tucker':
        U, S = tucker_dcmp(X, eps_or_k)
        U = [nn.Parameter(torch.Tensor(i.astype(datatype))) for i in U]
        S = nn.Parameter(torch.Tensor(S.astype(datatype)))
        param_dict = {'U': U, 'S': S}
    elif method == 'TT':
        A = tt_dcmp(X, eps_or_k)
        A = [nn.Parameter(torch.Tensor(i.astype(datatype))) for i in A]                           # todo:
        param_dict = {'U': A}
    elif method == 'LAF':
        U, S, V = my_svd(np.transpose(t_unfold(X, -1)), eps_or_k)
        U = nn.Parameter(torch.Tensor(U.astype(datatype)))
        V = nn.Parameter(torch.Tensor(np.dot(np.diag(S), V).astype(datatype)))
        param_dict = {'U': U, 'V': V}
    return param_dict
"""END"""


###########################################################################################################
class DMTRL_Linear(nn.Module):
    def __init__(self, in_feature, out_feature, tasks, method='Tucker'):
        super(DMTRL_Linear, self).__init__()
        self.in_feature, self.out_feature, self.tasks = in_feature, out_feature, tasks
        #
        self.b = nn.Parameter(torch.ones(size=(out_feature, tasks)))
        # Tensor Decomposition
        X = 0.01 * np.random.randn(in_feature, out_feature, tasks)
        self.method = method
        K = 5 if self.method == 'LAF' else 0.5
        params = TensorProducer(X, method, eps_or_k=K)
        # split into various method.
        if method == 'Tucker':
            self.U, self.S = params['U'], params['S']
            for l, param in enumerate(self.U):
                setattr(self, 'U_%d' % l, param)
        elif method == 'TT':
            self.U = params['U']
            for l, param in enumerate(self.U):
                setattr(self, 'U_%d' % l, param)
        elif method =='LAF':
            self.U, self.V = params['U'], params['V']
        return

    def forward(self, input, taskID):
        # build weight.
        if self.method == 'Tucker':
            W = TuckerTensorProducer(self.U, self.S)                      # todo:
        elif self.method == 'TT':
            W = TTTensorProducer(self.U)
        elif self.method == 'LAF':
            W = torch.mm(self.U, self.V).view(self.in_feature, self.out_feature, self.tasks)
        else:
            raise NotImplementedError
        W_task, b_task = W[:, :, taskID], self.b[:, taskID]
        feature = torch.mm(input, W_task) + b_task
        return feature