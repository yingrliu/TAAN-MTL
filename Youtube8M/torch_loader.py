"""
Author: Yingru Liu
Institute: Stony Brook University
"""
import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

data_path = 'data'

class Video_Level_Dataset(Dataset):
    def __init__(self, taskID, mode):
        """

        :param taskID: task.
        :param mode: train/valid/test.
        """
        assert mode in ['train', 'valid', 'test']
        assert taskID < 16
        hdf5_path = os.path.join(data_path, "%s_task_%d.hdf5" % (mode, taskID))
        f = h5py.File(hdf5_path, 'r')
        self.features = f.get('features')
        self.labels = f.get('labels')
        self.num_classes = f.attrs['num_classes']
        self.length = self.features.shape[0]
        return

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        feature = self.features[index]
        label = np.asarray(self.labels[index], dtype=np.float32)
        return feature, label

def MultiTask_Dataloader(mode, batch_size, shuffle=True):
    """
    return the Dataloader of all the task.
    notice that hdf5 do not support multi-process, so do not set the number of workers.
    :param mode:
    :param batch_size:
    :return:
    """
    task_list = [Video_Level_Dataset(ID, mode) for ID in range(16)]
    num_classes = [task.num_classes for task in task_list]
    Data_Loaders = [DataLoader(data, batch_size=batch_size, shuffle=shuffle) for data in task_list]
    return Data_Loaders, num_classes