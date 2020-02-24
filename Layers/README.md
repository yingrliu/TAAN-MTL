# Multi-Task Hidden Layers

## DMTRL Layers
```python
layer = DMTRL_Linear(in_feature, out_feature, tasks, method)
"""
:param tasks: number of tasks.
:param method: method for tensor decomposition [Tucker/LAF/TT].
"""
```

## MRN Layers
```python
layer = MRN_Linear(in_features, out_features, tasks, dropout, bn, regularization_task,
                 regularization_feature, regularization_input, update_interval)
"""
:param regularization_task: [bool] indicate whether add regularization on the task direction.
:param regularization_feature: [bool] indicate whether add regularization on the feature direction.
:param regularization_input: [bool] indicate whether add regularization on the input direction.
:param update_interval: [int] iteration interval that updates the covariance matrices.
"""
```

## TAL Layers
```python
layer = TAL_Linear(in_features, out_features, basis, tasks, bias, bn,
                 dropout, normalize, regularize)
"""
:param normalize: [boolean] whether normalize the coordinate.
:param regularize: [string] the type of regularization [tracenorm/distance/cosine] or None."""
```
```python
layer = TAL_Conv2d(self, in_channels, out_channels, kernel_size, basis, tasks, stride, padding, dilation,
                  groups, bias, padding_mode, bn, dropout, normalize, regularize)
"""
:param normalize: [boolean] whether normalize the coordinate.
:param regularize: [string] the type of regularization [tracenorm/distance/cosine] or None."""
```
