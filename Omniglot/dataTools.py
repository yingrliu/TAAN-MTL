"""
Author: Yingru Liu
Institute: Stony Brook University
This file contains the tools to load data.
"""

import torch, os, random
from torchvision import transforms
from PIL import Image

MAIN_DIR = "data"

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = Image.open(path)
    return img.convert('1')

"""----------------------------------------------------------------------------------------------------------------
convert_paths: Save the list of data paths. The images of each language is split and the pathes are saved in 
language_train.txt and language_test.txt. The whole list of language_train/test.txt is saved into train_list.txt 
and test_list.txt. urthermore, class_annotation.txt denotes the amount of classes in each language.
:type: Function
:return: None
----------------------------------------------------------------------------------------------------------------"""
def convert_paths():
    languages = os.listdir(MAIN_DIR)
    train_list, test_list, class_annotation = open(os.path.join(MAIN_DIR, 'train_list.txt'), "w"), \
                                              open(os.path.join(MAIN_DIR, 'test_list.txt'), "w"), \
                                              open(os.path.join(MAIN_DIR, 'class_annotation.txt'), "w")
    for language in languages:
        if language[-3:] == 'txt':
            continue
        train_list.write(os.path.join(MAIN_DIR, language + '_train.txt')+'\n')
        test_list.write(os.path.join(MAIN_DIR, language + '_test.txt')+'\n')
        f_train = open(os.path.join(MAIN_DIR, language + '_train.txt'), "w")
        f_test = open(os.path.join(MAIN_DIR, language + '_test.txt'), "w")
        language_path = os.path.join(os.path.join(MAIN_DIR, language))
        class_amount = os.listdir(language_path)
        class_annotation.write(language + ' ' + str(len(class_amount)) + '\n')
        # split the characters to train/test.
        for c in class_amount:
            Label = str(int(c[-2:])-1)
            character_path = os.path.join(language_path, c)
            files = os.listdir(character_path)
            for file in files:
                file_path = os.path.join(character_path, file)
                split = random.random()
                if split < 0.9:
                    f_train.write(file_path + ' ' + Label+'\n')
                else:
                    f_test.write(file_path + ' ' + Label + '\n')
    return

"""----------------------------------------------------------------------------------------------------------------
ImageList: a data loader to load the batch of images for a specific language.
:type: Class
----------------------------------------------------------------------------------------------------------------"""
class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_list):
        """
        a pytorch loader to load the images of each language.
        :param image_list: train_list.txt/test_list.txt
        """
        with open(image_list, "r") as f0:
            contents = f0.readlines()
            self.image_list = []
            for image in contents:
                path, label = image.split(' ')
                label = int(label[0:-1])
                self.image_list.append((path, label))
        return

    def __getitem__(self, index):
        path, label = self.image_list[index]
        img = TRANSFORM(pil_loader(path))
        return img, label

    def __len__(self):
        return len(self.image_list)

"""----------------------------------------------------------------------------------------------------------------
DataLayer: a full data loader to load the batch of images of all languages to train models.
:type: Class
----------------------------------------------------------------------------------------------------------------"""
class DataLayer(object):
    def __init__(self, batchSize):
        # Access the list of files.
        f = open(os.path.join(MAIN_DIR, 'train_list.txt'), "r").readlines()
        train_file_list = [line[0:-1] for line in f]
        f = open(os.path.join(MAIN_DIR, 'test_list.txt'), "r").readlines()
        test_file_list = [line[0:-1] for line in f]
        #
        dsets = {"train": [ImageList(image_list) for image_list in train_file_list],
                 "test": [ImageList(image_list) for image_list in test_file_list]}
        # define pytorch loader.
        self.train_loader = []
        self.test_loader = []
        for train_dset in dsets["train"]:
            self.train_loader.append(
                torch.utils.data.DataLoader(train_dset, batch_size=batchSize, shuffle=True, num_workers=4))
        for test_dset in dsets["test"]:
            self.test_loader.append(
                torch.utils.data.DataLoader(test_dset, batch_size=batchSize, shuffle=True, num_workers=4))
        return

# todo: Getting OSError: [Errno 24] Too many open files. Not sure why.
def ComputeACC(model, testSet):
    model.eval()
    iter_test = [iter(loader) for loader in testSet]
    start_test = True
    for i in range(len(iter_test)):
        iter_ = iter_test[i]
        for j in range(len(testSet[i])):
            inputs, labels = iter_.next()
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model.forward(inputs, i)[0]
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    taskAcc = torch.sum(torch.squeeze(all_output).float() == all_label).float() / float(all_label.size()[0])
    return taskAcc

"""----------------------------------------------------------------------------------------------------------------
fit: fit the model.
:type: function
----------------------------------------------------------------------------------------------------------------"""
def fit(model, batchSize, testInterval=100, num_iter=5000, earlyStop=100, saveto=None):
    dataset = DataLayer(batchSize)
    trainSet = dataset.train_loader
    testSet = dataset.test_loader
    if saveto and not os.path.exists(saveto):
        os.makedirs(saveto)
    len_renew = min([len(loader) - 1 for loader in trainSet])
    best_acc, worseCase = 0, 0
    #
    for iter_num in range(0, num_iter):
    # Validation Phrase.
        if iter_num % testInterval == 0:
            # epoch_acc, epoch_acc_train = ComputeACC(model, testSet), ComputeACC(model, trainSet)
            # print('Iter {:05d} Average Train Acc: {:.4f}; Average Test Acc: {:.4f};'.format(
            #     iter_num, epoch_acc_train, epoch_acc))

            epoch_acc = ComputeACC(model, testSet)
            print('Iter {:05d} Average Test Acc: {:.4f};'.format(
                iter_num, epoch_acc))
            if epoch_acc > best_acc:
                best_acc, worseCase = epoch_acc, 0
                if saveto:
                    model.saveModel(os.path.join(saveto, 'model_params.pt'))
            else:
                worseCase += 1
            print('Best val Acc: {:4f}'.format(best_acc))
            if worseCase >= earlyStop:
                break
        if iter_num % len_renew == 0:
            iter_list = [iter(loader) for loader in trainSet]
        dataX, dataY = [], []
        # Training phrase.
        for iter_ in iter_list:
            data = iter_.next()
            dataX.append(data[0].cuda())
            dataY.append(data[1].cuda())
        model.train_step(dataX, dataY)
    return best_acc