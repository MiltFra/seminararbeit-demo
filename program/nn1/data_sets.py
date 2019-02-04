import os
from mxnet import nd, gluon
from data_utils import sym2indx as s2i
from data_utils import indx2onehot as i2o
from data_utils import indx2sym as i2s
from data_utils import dist2prob as d2p
import pickle
import numpy

def get_data1(
    batch_size,
    num_batches,
    PATH='/home/miltfra/projects/Seminararbeit/Data/pwn_6_7.txt'
):
    def get_data(file, size):
        if not os.path.isfile(file):
            print(f'Invalid File: {file}')
        fsize = os.path.getsize(file)
        if fsize < size * 2:
            size = fsize // 2
        with open(file) as f:
            train_data = f.read(size)
            valid_data = f.read(size)
        print('> Getting training data')
        train_dataset = get_dataset(train_data)
        print('> Getting validation data')
        valid_dataset = get_dataset(valid_data)
        return train_dataset, valid_dataset

    def get_dataset(data):
        a = analyze(data)
        X = nd.zeros((len(a.keys()), 192))
        y = nd.zeros((len(a.keys()), 96))
        for i, k in enumerate(a.keys()):
            x = nd.zeros((2, 96))
            x[0] = i2o(s2i(k[0]), 96)
            x[1] = i2o(s2i(k[1]), 96)
            X[i] = nd.reshape(x, 192)
            y[i] = a[k]
        return gluon.data.dataset.ArrayDataset(
            X, y
        )

    def analyze(data):
        analysis_dct = dict()
        for i in range(len(data) - 2):
            l = analysis_dct.get(
                (data[i], data[i + 1]), []
            )
            l.append(s2i(data[i + 2]))
            analysis_dct[(data[i],
                          data[i + 1])] = l
            print(
                f'{100*i/(len(data)-2):.2f}%',
                end='\r'
            )
        for k in analysis_dct.keys():
            k_array = nd.array(
                numpy.asarray(analysis_dct[k])
            )
            k_array = nd.one_hot(
                k_array, depth=96
            )
            k_array = nd.sum(k_array, axis=0)
            analysis_dct[k] = d2p(
                nd.reshape(k_array, 96), axis=0
            )
        return analysis_dct

    def acc(output, label):
        min_acc = 1
        for i in range(output.shape[0]):
            x = nd.sum(
                nd.abs(
                    nd.subtract(
                        output[i], label[i]
                    )
                ),
                axis=0
            )
            if 1 - x / 96 < min_acc:
                min_acc = 1 - x / 96
        return min_acc.asscalar()

    file = PATH.split('/')[-1]
    data_set_path = f'/home/miltfra/projects/Seminararbeit/Data/datasets/{file}-bs{batch_size}-nb{num_batches}.pkl'
    if os.path.isfile(data_set_path):
        with open(data_set_path, 'rb') as f:
            train_data, valid_data = pickle.load(
                f
            )
    else:
        train_data, valid_data = get_data(
            PATH, num_batches * batch_size
        )
        with open(data_set_path, 'wb') as f:
            pickle.dump(
                (train_data, valid_data), f, -1
            )
    train_data = gluon.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    valid_data = gluon.data.DataLoader(
        valid_data,
        batch_size=batch_size * num_batches,
        num_workers=8
    )
    for data, label in train_data:
        print(
            'input.shape: ', data.shape,
            'output.shape: ', label.shape
        )
        break
    return train_data, valid_data, acc, gluon.loss.SoftmaxCELoss(
        sparse_label=False
    )

def get_data2(
    batch_size,
    num_batches,
    PATH='data/gamigo_10_11'
):
    def get_data(file, size):
        if not os.path.isfile(file):
            print(f'data2> Invalid File: {file}')
        fsize = os.path.getsize(file)
        if fsize < size * 2:
            size = fsize // 2
        with open(file) as f:
            train_data = f.read(size)
            valid_data = f.read(size)
        print('data2> Getting training data')
        train_dataset = get_dataset(train_data)
        print('data2> Getting validation data')
        valid_dataset = get_dataset(valid_data)
        return train_dataset, valid_dataset

    def get_dataset(data):
        X = nd.zeros((len(data) - 2, 2))
        y = nd.zeros((len(data) - 2))
        for i in range(len(data) - 2):
            a, b, c = data[i:i + 3]
            X[i] = nd.array([s2i(a), s2i(b)])
            y[i] = s2i(c)
            print(
                f'{i*100/(len(data)-2):.2f}%',
                end='\r'
            )
        X = nd.reshape(
            nd.one_hot(X, depth=96),
            (len(data) - 2, 192)
        )
        y = nd.one_hot(y, depth=96)
        return gluon.data.dataset.ArrayDataset(
            X, y
        )

    def acc(output, label):
        return (
            output.argmax(axis=1) == label.argmax(
                axis=1
            )
        ).mean().asscalar()

    file = PATH.split('/')[-1]
    data_set_path = f'data/{file}-s{batch_size*num_batches}'
    print(
        'Checking for existance of ',
        data_set_path, '.train_data'
    )
    if os.path.isfile(
        data_set_path + '.train_data'
    ):
        print(
            'data2> Data set already exists, loading ...'
        )
        with open(
            data_set_path + '.train_data', 'rb'
        ) as f:
            train_data = pickle.load(f)
        with open(
            data_set_path + '.valid_data', 'rb'
        ) as f:
            valid_data = pickle.load(f)
    else:
        print(
            'data2> No existing data set found...'
        )
        train_data, valid_data = get_data(
            PATH, num_batches * batch_size
        )
        with open(
            data_set_path + '.train_data', 'wb'
        ) as f:
            pickle.dump(train_data, f, -1)
        with open(
            data_set_path + '.valid_data', 'wb'
        ) as f:
            pickle.dump(train_data, f, -1)
    train_data = gluon.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    valid_data = gluon.data.DataLoader(
        valid_data,
        batch_size=batch_size * num_batches,
        num_workers=8
    )
    for data, label in train_data:
        print(
            'input.shape: ', data.shape,
            'output.shape: ', label.shape
        )
        break
    return train_data, valid_data, acc, gluon.loss.KLDivLoss(
        from_logits=False
    )

def get_data3(
    batch_size,
    num_batches,
    PATH='data/gamigo_10_11'
):
    def get_data(path, num_batches, batch_size):
        with open(path) as f:
            train_data = f.read(
                batch_size * num_batches
            )
            valid_data = f.read(
                batch_size * num_batches
            )
        print('data3> Getting train_data...')
        train_dataset = get_dataset(
            train_data, num_batches, batch_size
        )
        print('data3> Geting valid_data...')
        valid_dataset = get_dataset(
            valid_data, num_batches, batch_size
        )
        return train_dataset, valid_dataset

    def get_dataset(
        data, num_batches, batch_size
    ):
        X = nd.zeros(
            (num_batches, batch_size + 1)
        )
        Y = nd.zeros(
            (num_batches, batch_size + 1)
        )
        l = len(data)
        for i in range(num_batches):
            for j in range(batch_size):
                X[i, j + 1] = s2i(
                    data[batch_size * i + j]
                )
                Y[i, j + 1] = s2i(
                    data[(batch_size * i + j + 1)
                         % l]
                )
                print(
                    f'{(batch_size*i+j)*100/(num_batches*batch_size):.2f}%',
                    end='\r'
                )
        X = nd.one_hot(X, depth=96)
        Y = nd.one_hot(Y, depth=96)
        return gluon.data.ArrayDataset(X, Y)

    def acc(output, label):
        return (
            output.argmax(axis=1) == label.argmax(
                axis=1
            )
        ).mean().asscalar()

    file = PATH.split('/')[-1]
    data_set_path = f'data/{file}-d{num_batches}x{batch_size}'
    if not os.path.isfile(PATH):
        print(f'data3> Invalid file: {PATH}')
        size = os.path.getsize(PATH)
        if size < num_batches * batch_size * 2:
            num_batches = size // (batch_size * 2)
    print(
        'Checking for existance of ',
        data_set_path, '.train_data'
    )
    if os.path.isfile(
        data_set_path + '.train_data'
    ):
        print(
            'data2> Data set already exists, loading ...'
        )
        with open(
            data_set_path + '.train_data', 'rb'
        ) as f:
            train_data = pickle.load(f)
        with open(
            data_set_path + '.valid_data', 'rb'
        ) as f:
            valid_data = pickle.load(f)
    else:
        train_data, valid_data = get_data(
            PATH, num_batches, batch_size
        )
        with open(
            data_set_path + '.train_data', 'wb'
        ) as f:
            pickle.dump(train_data, f, -1)
        with open(
            data_set_path + '.valid_data', 'wb'
        ) as f:
            pickle.dump(train_data, f, -1)
    train_data = gluon.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    valid_data = gluon.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        num_workers=8
    )
    return train_data, valid_data, acc, gluon.loss.KLDivLoss(
        from_logits=False
    )
