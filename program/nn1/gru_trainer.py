import sys
def train(
    batch_size=7,
    num_batches=10000,
    lr=1,
    min_chg=1,
    patience=2,
    path='/home/miltfra/projects/Seminararbeit/Data/ANN-a3/',
    _name='big',
    hidden=2
):
    import os
    import numpy as np
    from mxnet import nd, gluon, initializer, autograd
    from time import time
    import models
    import data_sets
    import pickle
    #=== PARAMETERS ===
    #TARGET_ACC = 0.99
    print('gru_trainer> Loading Parameters')
    if _name == '':
        name = f'gru{hidden}-s{num_batches}x{batch_size}-{_name}'
    else:
        name = f'gru-{_name}'

    #=== INIT DATA ===
    print('gru_trainer> Obtaining Data')
    train_data, valid_data, acc, loss_f = data_sets.get_data3(
        batch_size, num_batches
    )
    net = models.gru1(hidden)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': lr}
    )
    batch_size += 1
    #=== VARIABLES ===
    print('gru_trainer> Initializing Variables')
    best_loss = 0
    train_acc = 0
    valid_acc = 0
    avg_train_acc = 0
    avg_valid_acc = 0
    epoch = 0
    best_loss = float('inf')
    last_decrease = 0
    stats = []
    #===TRAINING LOOP ===
    print('gru_trainer> Starting Training')

    def detach(hidden):
        if isinstance(hidden, (tuple, list)):
            return [i.detach() for i in hidden]
        else:
            return hidden.detach()

    while epoch < 20:
        train_loss, train_acc, valid_acc = 0., 0., 0.
        tic = time()
        hidden = net.begin_state(
            batch_size, func=nd.zeros
        )
        for data_sets, label in train_data:
            # forward + backward
            hidden = net.begin_state(
                batch_size, func=nd.zeros
            )
            with autograd.record():
                output, hidden = net(
                    data_sets, hidden
                )
                loss = loss_f(output, label)
            loss.backward()
            # update parameters
            trainer.step(batch_size)
            # calculate training metrics
            train_loss += loss.mean().asscalar()
            train_acc += acc(output, label)

        # calculate validation accuracy
        for data_sets, label in valid_data:
            hidden = net.begin_state(
                batch_size, func=nd.zeros
            )
            valid_acc += acc(
                net(data_sets, hidden)[0], label
            )
        avg_train_acc = train_acc / len(
            train_data
        )
        avg_valid_acc = valid_acc / len(
            valid_data
        )

        # if best_avg_acc increases, reset patience, save models
        if train_loss < best_loss:
            net.save_parameters(
                f'{path}{name}.params'
            )
            best_loss = train_loss
            last_decrease = 0
        # if best_avg_acc doesn't increase in PATIENCE steps, end training
        else:
            last_decrease += 1
        epoch += 1
        best_loss = train_loss
        stats.append(
            (
                epoch, train_loss, avg_train_acc,
                avg_valid_acc, time() - tic
            )
        )
        print(
            f'E: {epoch:3d} | D: {last_decrease:1d} | L: {train_loss:.2f} | T: {avg_train_acc*100:5.2f}% | V: {avg_valid_acc*100:5.2f}% | t: {time()-tic:6.2f}s'
        )
    stats = np.asarray(stats)
    with open(f'{path}{name}.stats', 'wb') as f:
        pickle.dump(stats, f, -1)
    print(
        f'gru_trainer> Stats have been saved to {path}{name}.stats'
    )
    del hidden, loss, loss_f, train_data, valid_data, stats
    return f'{path}{name}.params', f'{path}{name}.stats'

import stats
import gru_loader
seq_name = 'demo'
stat_list = []
params = []
if len(sys.argv) > 2:
    print(
        f'gru_train_loop> training {seq_name}-{sys.argv[1]}-{sys.argv[2]}'
    )
    p, s = train(
        lr=float(sys.argv[2]),
        hidden=int(sys.argv[1]),
        path='',
        _name=f'{seq_name}-{sys.argv[1]}-{sys.argv[2]}'
    )
    params.append(p)
    stat_list.append(s)
    print(f'gru_train_loop> drawing {s}')
    stats.plot_f(s)