import sys, os, time
sys.path.append('../src')
import tables as tb
import numpy as np
import learner

# Dataset conversion
kind = 'map_90_60'
dset_convs = ['norm_abs']
n_x_n_y = [(4, 4), (4, 8), (8, 4), (8, 8)]
shifts = [4]
# Optimizers: ['Adadelta', 'Nadam', 'SGD', 'RMSprop', 'Adagrad', 'Adamax', 'Adam', 'Ftrl']
optimizers = ['Nadam']
# Loss function: ['mse', 'mae', 'msle']
losses = ['mse']
# Neural networks: ['ST_MDF_star', 'ST_MDF_w', 'ST_MDF_t', 'ST_MDF', 'persistence', 'naive', 'fc', 'lstm', 'biLstm']
nns = ['ST_MDF_star']

# Create the learner
l = learner.DeepLearner()
epochs = 50
l.models.set_lr(0)
experiment = 'experiment01'


def train(l, model, epochs):
    if not os.path.exists(os.path.join(l.models_path, model.mod_name)):
        os.mkdir(os.path.join(l.models_path, model.mod_name))
    tic = time.time()
    model = l._DeepLearner__train(model, epochs)
    train_duration_s = time.time() - tic
    fname = os.path.join(l.models_path, model.mod_name, model.mod_name + '.h5')
    # If model already existed, add those previous epochs
    if os.path.exists(fname):
        with tb.open_file(fname, 'r') as h5_mod:
            node = h5_mod.get_node('/')
            epochs += node._v_attrs['epochs']
    model.save(fname)
    # For reproducibility:
    l._DeepLearner__add_meta(fname, model.mod_name, epochs, train_duration_s)
    # Do tests
    l._DeepLearner__plot_model(model)
    l._DeepLearner__plot_loss(model)
    #l._DeepLearner__add_attr(model.mod_name, 'train_duration', 0)
    l._DeepLearner__test(model)
    l._DeepLearner__test_counts(model)
    l._DeepLearner__test_date(model, date='14/11/2019')
    l._DeepLearner__test_date(model, date='15/11/2019')
    l._DeepLearner__test_date(model, date='16/11/2019')
    #l._DeepLearner__robustness_test(model, n_reps=20)

for dset_conv in dset_convs:
    l.set_dset_conv(dset_conv)
    l.models_path = os.path.join(os.path.dirname(l.models_path), experiment)
    if not os.path.exists(l.models_path):
        os.mkdir(l.models_path)
    for optimizer in optimizers:
        l.models.set_optimizer(optimizer)
        for loss in losses:
            l.models.set_loss(loss)
            for n_x, n_y in n_x_n_y:
                l.set_n_x(n_x)
                l.set_n_y(n_y)
                for shift in shifts:
                    l.set_shift(shift)
                    for nn in nns:
                        model = l.models.model(nn)
                        model.mod_name = '{}_nx{:02}_ny{:02}_sh{:02}_{}_{}_ep{:03}'.format(nn, n_x, n_y, shift, optimizer, loss, epochs)
                        train(l, model, epochs)
    # Produce table with results when finished
    l.results_table('results_{}.csv'.format(experiment), auto=False)
