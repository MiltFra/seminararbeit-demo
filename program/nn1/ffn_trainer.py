import sys
def train(
    lr=1,
    batch_size=7,
    num_batches=10000,
    min_chg=5,
    patience=5,
    path='',
    name='a1',
    hidden=1,
    summary=False
):
    import numpy as np
    from mxnet import nd, gluon, initializer, autograd
    from time import time
    import models
    import data_sets
    import pickle
    #=== PARAMETERS ===
    #TARGET_ACC = 0.99
    print('ffn_trainer> Loading Parameters')
    _name = f'ffn{hidden}-s{num_batches}x{batch_size}-{name}'

    #=== INIT DATA ===
    print('ffn_trainer> Obtaining Data')
    train_data, valid_data, acc, loss_f = data_sets.get_data2(
        batch_size, num_batches
    )
    net = models.nn1(hidden)

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': lr}
    )
    #=== VARIABLES ===
    print('ffn_trainer> Initializing Variables')
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
    print('ffn_trainer> Starting Training')
    while epoch < 20:
        train_loss, train_acc, valid_acc = 0., 0., 0.
        tic = time()
        for data_sets, label in train_data:
            # forward + backward
            with autograd.record():
                output = net(data_sets)
                loss = loss_f(output, label)
            loss.backward()
            # update parameters
            trainer.step(batch_size)
            # calculate training metrics
            train_loss += loss.mean().asscalar()
            train_acc += acc(output, label)

        # calculate validation accuracy
        for data_sets, label in valid_data:
            valid_acc += acc(
                net(data_sets), label
            )
        avg_train_acc = train_acc / len(
            train_data
        )
        avg_valid_acc = valid_acc / len(
            valid_data
        )

        # if best_avg_acc increases, reset patience, save models
        if train_loss < best_loss - min_chg:
            net.save_parameters(
                f'{path}{_name}.params'
            )
            best_loss = train_loss
            last_decrease = 0
        # if best_avg_acc doesn't increase in PATIENCE steps, end training
        else:
            last_decrease += 1
        epoch += 1
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
    with open(f'{path}{_name}.stats', 'wb') as f:
        pickle.dump(stats, f, -1)
        print(
            f'ffn_trainer> Stats have been saved to {path}{_name}.stats'
        )
    return f'{path}{_name}.params', f'{path}{_name}.stats'

import stats
import ffn_loader
seq_name = 'demo'
if len(sys.argv) > 2:
    print(
        f'ffn_train_loop> training {seq_name}-{sys.argv[1]}-{sys.argv[2]}'
    )
    p, s = train(
        lr=float(sys.argv[2]),
        hidden=int(sys.argv[1]),
        path='',
        name=f'{seq_name}-{sys.argv[1]}-{sys.argv[2]}'
    )