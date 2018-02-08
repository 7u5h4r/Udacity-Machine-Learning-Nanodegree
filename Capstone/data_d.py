
# Downloading the MSIT Data from yann's website
import gzip
import os
from six.moves.urllib.request import urlretrieve
import numpy as npy
data_src = 'http://yann.lecun.com/exdb/mnist/'


def check_dl(file_n, dir_path):
# checking if the data exists if dosent we will download it
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    path_f = os.path.join(dir_path, file_n)
    if not os.path.exists(path_f):
        path_f, _ = urlretrieve(data_src + file_n, path_f)
        statinfo = os.stat(path_f)
        print('Download Done', file_n, statinfo.st_size, 'b.')
    return path_f


def _read32(bytestream):
    dt = npy.dtype(npy.uint32).newbyteorder('>')
    return npy.frombuffer(bytestream.read(4), dtype=dt)[0]


def img_ex(file_n):
#Extracting the images into 4D Numpy array ([index, y, x, depth])
    print('Extractng.........', file_n)
    with gzip.open(file_n) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Magic number is invalid %d in MNIST image file: %s' %
                (magic, file_n))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = npy.frombuffer(buf, dtype=npy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def one_hot_conv(l_d, num_classes=10):
#One Hot Encoding
    num_labels = l_d.shape[0]
    index_offset = npy.arange(num_labels) * num_classes
    l_one_hot = npy.zeros((num_labels, num_classes))
    l_one_hot.flat[index_offset + l_d.ravel()] = 1
    return l_one_hot


def extract_labels(file_n, one_hot=False):
#Extracting the labels into 1D numpy array
    print('Extracting....', file_n)
    with gzip.open(file_n) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Magic number is invalid %d in MNIST image file: %s' %
                (magic, file_n))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = npy.frombuffer(buf, dtype=npy.uint8)
        if one_hot:
            return one_hot_conv(labels)
        return labels


class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Converting the shape from [num examples, rows, columns, depth] to [num examples, rows*columns] while (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Converting from [0, 255] -> [0.0, 1.0].
            images = images.astype(npy.float32)
            images = npy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
#returning the next batch size examples
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
#epoch is finishes
            self._epochs_completed += 1
#Data Shuffling
            perm = npy.arange(self._num_examples)
            npy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
#Next Epoch Starting
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 5000
    local_file = check_dl(TRAIN_IMAGES, train_dir)
    train_images = img_ex(local_file)
    local_file = check_dl(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(local_file, one_hot=one_hot)
    local_file = check_dl(TEST_IMAGES, train_dir)
    test_images = img_ex(local_file)
    local_file = check_dl(TEST_LABELS, train_dir)
    test_labels = extract_labels(local_file, one_hot=one_hot)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets
