"""
Author: Yingru Liu
Tool to extract data.
"""
import os
import csv
import glob
import h5py
import numpy as np
import tensorflow as tf
from random import shuffle
###################################
csv_path = "E:\\Dataset\\vocabulary.csv"
train_path = "E:\\Dataset\\Train\\*.tfrecord"
valid_path = "E:\\Dataset\\Validate\\*.tfrecord"

class TFRecord():
    def __init__(self, fileList):
        self.num_classes = 3862
        self.feature_sizes = [1024, 128]
        self.feature_names = ["mean_rgb", "mean_audio"]
        reader = tf.TFRecordReader()
        feature_map = {
            "id": tf.FixedLenFeature([], tf.string),
            "labels": tf.VarLenFeature(tf.int64)
        }
        for feature_index in range(len(self.feature_names)):
            feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
                [self.feature_sizes[feature_index]], tf.float32)
        filename_queue = tf.train.string_input_producer(fileList, num_epochs=1, shuffle=False)
        _, serialized_example = reader.read_up_to(filename_queue, 1024)
        features = tf.parse_example(serialized_example, feature_map)
        #
        self.labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
        # labels.set_shape([None, self.num_classes])
        self.concatenated_features = tf.concat([features[feature_name] for feature_name in self.feature_names], 1)
        return

    def transform(self, saveto, table, tag):
        """
        the path to save the features as hdf5.
        :param saveto:
        :return:
        """
        with h5py.File(saveto, 'w') as Dataset:
            Dataset.create_dataset('features', (1, sum(self.feature_sizes)), maxshape=(None, sum(self.feature_sizes)),
                                   chunks=True)
            Dataset.create_dataset('labels', (1, tag), maxshape=(None, tag), chunks=True)
            Dataset.attrs['num_classes'] = tag
            END = 0
            with tf.Session() as sess:
                tf.local_variables_initializer().run()
                init = tf.global_variables_initializer()
                sess.run(init)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)
                try:
                    while True:
                        features, labels = sess.run([self.concatenated_features, self.labels])
                        labels = np.asarray(labels, dtype=np.float32)
                        # filter out the data that's not in the category.
                        # change labels according to table.
                        labels = labels[:, list(table.keys())]
                        inCat = np.asarray(labels.sum(axis=-1), dtype=np.bool)
                        features = features[inCat]
                        labels = labels[inCat]
                        batchsize = features.shape[0]
                        if not batchsize:
                            continue
                        Dataset['features'].resize((END + batchsize, sum(self.feature_sizes)))
                        Dataset['features'][END:END + batchsize, :] = features
                        Dataset['labels'].resize((END + batchsize, len(table.keys())))
                        Dataset['labels'][END:END + batchsize, :] = np.asarray(labels, dtype=np.int16)
                        END += batchsize
                except tf.errors.OutOfRangeError:
                    coord.request_stop()
                finally:
                    coord.request_stop()
                    coord.join(threads)
                coord.request_stop()
                coord.join(threads)
        return

def split_categories(cat, csv_path):
    """
    access the sub-class of a categories and re-number the class ID.
    :param cat:
    :return:
    """
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        lines = [row for row in reader][1:]
        isCat = []
        for line in lines:
            if line[5] == cat:
                isCat.append(line)
        table = {}
        tag = 0
        for line in isCat:
            table[int(line[0])] = tag
            tag += 1
    return table, tag                       # tag is the length of the table.

def TFRecord2hdf5(mode):
    categories = ['Arts & Entertainment', 'Games', 'Autos & Vehicles', 'Sports',
                  'Food & Drink', 'Computers & Electronics', 'Business & Industrial', 'Pets & Animals',
                  'Hobbies & Leisure', 'Beauty & Fitness', 'Shopping', 'Internet & Telecom',
                  'Home & Garden', 'Science', 'Travel', 'Law & Government']
    if mode not in ['train', 'valid']:
        raise ValueError("Mode is not correct.")
    saveto_ = 'data'
    if not os.path.exists(saveto_):
        os.makedirs(saveto_)
    if mode == 'train':
        file_path = glob.glob(train_path)
        for d, cat in enumerate(categories):
            print('Access %s set and %d category.' % (mode, d))
            saveto = os.path.join(saveto_, "train_task_%s.hdf5" % d)
            table, tag = split_categories(cat, csv_path)
            reader = TFRecord(file_path)
            reader.transform(saveto, table, tag)

    else:
        file_path = glob.glob(valid_path)
        shuffle(file_path)
        """process valid set."""
        file_path_valid = file_path[0:len(file_path) // 2]
        for d, cat in enumerate(categories):
            print('Access %s set and %d category.' % ('valid', d))
            saveto = os.path.join(saveto_, "valid_task_%s.hdf5" % d)
            table, tag = split_categories(cat, csv_path)
            reader = TFRecord(file_path_valid)
            reader.transform(saveto, table, tag)
        """process test set."""
        file_path_test = file_path[len(file_path) // 2:]
        for d, cat in enumerate(categories):
            print('Access %s set and %d category.' % ('test', d))
            saveto = os.path.join(saveto_, "test_task_%s.hdf5" % d)
            table, tag = split_categories(cat, csv_path)
            reader = TFRecord(file_path_test)
            reader.transform(saveto, table, tag)
    return


# if __name__ == "__main__":
#     TFRecord2hdf5(mode='train')
#     TFRecord2hdf5(mode='valid')

