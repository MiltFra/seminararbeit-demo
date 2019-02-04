from mxnet import gluon, initializer


def nn1(n):
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(192, activation='relu'))
    for _ in range(n):
        net.add(gluon.nn.Dense(192, activation='relu'))
    net.add(gluon.nn.Dense(96, activation='relu'))
    net.initialize(init=initializer.Xavier())
    return net


def gru1(n):
    net = gluon.rnn.GRU(96, input_size=96, num_layers=n,
                        i2h_weight_initializer='xavier', h2h_weight_initializer='xavier')
    net.initialize()
    return net
