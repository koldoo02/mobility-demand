import os, time, re
import numpy as np
import pandas as pd
import tables as tb
import datetime as dt
import deep_playground, modelUtils, plotUtils
from trainUtils import HistoryCallback, DataGenerator
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.interpolate import griddata
from pandas.plotting import register_matplotlib_converters


class DeepLearner():
    '''
    Class that encapsulates common tasks when training models.
    '''

    def __init__(self, work_path='..', time_gran='15m', city='chicago',
                 kind='map_90_60', dset_conv='norm_abs',
                 n_x=4, n_y=4, shift=1,
                 batch_size=2**8, time_aware=True,
                 zone_ids=slice(None),
                ):
        # For files and paths
        self.work_path = work_path
        self.time_gran = time_gran
        self.models_path = os.path.join(self.work_path, 'models', city, 'shifted')
        self.data_path = os.path.join(self.work_path, 'data', city, 'clean')
        self.other_path = os.path.join(self.work_path, 'data', city, 'other')
        self.dataset = '{}_{}_{}_{}.h5'
        self.dataset_d = {'time_series':[os.path.join(self.data_path, self.dataset.format(time_gran, kind, 'taxi', dset_conv)),
                                         os.path.join(self.data_path, self.dataset.format(time_gran, kind, 'bike', dset_conv))]}
        self.group = '/2013'
        if time_aware:
            self.dataset_d['weather'] = [os.path.join(self.data_path, '01h_weather_norm.h5')]
            self.dataset_d['holidays'] = [os.path.join(self.other_path, 'holidays.csv')]
        self.city, self.kind, self.dset_conv = city, kind, dset_conv
        # For training and prediction
        self.n_x = n_x
        self.n_y = n_y
        self.shift = shift
        self.batch_size = batch_size
        self.time_aware = time_aware
        # train = [2013, 2017]; val = [2018]; test = [2019, 2020]
        # Caveat! Leap years: 2016 and 2020
        #self.train_extent = (0, 1250)
        #self.val_extent = (1250, 1650)
        #self.test_extent = (1650, 2000)
        ints_in_day = 4 * 24 # depends on time_gran
        train_lim = (5 * 365 + 1) * ints_in_day
        self.train_extent = (0, train_lim)
        val_lim = train_lim + 1 * 365 * ints_in_day
        self.val_extent = (train_lim, val_lim)
        #test_lim = val_lim + (2 * 365 + 1) * ints_in_day
        # Or, begining of COVID pandemic, idx = 252576
        test_lim = self.__datetime_to_idx(dt.datetime(2020, 3, 16, 0, 0))
        self.test_extent = (val_lim, test_lim)
        self.df_stats = pd.read_csv(os.path.join(self.other_path, 'metrics-per-zone-taxi.csv'), index_col=0)
        # Mean of taxi and bike trips per zone, normalized
        with tb.open_file('../data/chicago/clean/15m_flat_taxi_count.h5', mode='r') as h5_taxi:
            t = h5_taxi.get_node('/2013')[:].mean(axis=0)
        self.tnorm = (t - t.min()) / (t.max() - t.min()) 
        with tb.open_file('../data/chicago/clean/15m_flat_bike_count.h5', mode='r') as h5_bike:
            b = h5_bike.get_node('/2013')[:].mean(axis=0)
        self.bnorm = (b - b.min()) / (b.max() - b.min())
        # Target variable
        self.target_d = {'flat_count': 'Trip counts',
                         'map_40_40_count': 'Trip counts map 40x40 [-]',
                         'map_90_60_count': 'Trip counts map 90x60 [-]',
                         'flat_norm': 'Trip counts normalized per zone',
                         'map_40_40_norm': 'Normalized trip counts map 40x40 [-]',
                         'map_90_60_norm': 'Normalized trip counts map 90x60 [-]',
                         'flat_norm_abs': 'Trip counts normalized abs.',
                         'map_40_40_norm_abs': 'Normalized abs. trip counts map 40x40 [-]',
                         'map_90_60_norm_abs': 'Normalized abs. trip counts map 90x60 [-]',
                         'norm_abs_050': 'Trip counts normalized abs. at 50',
                         'flat_stand': 'Trip counts standardized per zone',
                         'map_40_40_stand': 'Standardized trip counts map 40x40 [-]',
                         'flat_stand_abs': 'Trip counts standardized abs.',
                         'map_40_40_stand_abs': 'Standardized abs. trip counts map 40x40 [-]'
                        }
        self.target = self.target_d['{}_{}'.format(kind, dset_conv)]
        with tb.open_file(self.dataset_d['time_series'][0], mode='r') as h5_file:
            self.zones = h5_file.get_node(self.group)._v_attrs['columns']
        self.zone_ids = zone_ids
        # Additional stuff
        self.menu = deep_playground.Menu()
        self.helper = deep_playground.Helper()
        self.models = modelUtils.Models(n_x=self.n_x, n_y=self.n_y,
                                        act_out='sigmoid' if 'norm' in dset_conv else 'linear')
        self.set_kind(self.kind)
        self.plotter = plotUtils.Plotter()
        self.mod_queue = []
        register_matplotlib_converters() # <- to avoid a warning

    def __idx_to_datetime(self, idx, year=2013, freq=None):
        # Frequency expressed in minutes
        if freq is None:
            freq = int(self.time_gran.replace('m', ''))
        base = dt.datetime(year, 1, 1, 0, 0)
        return base + dt.timedelta(minutes=freq * idx)

    def __datetime_to_idx(self, date, freq=None):
        # Frequency expressed in minutes
        if freq is None:
            freq = int(self.time_gran.replace('m', ''))
        # old: base = dt.datetime(date.year, 1, 1, 0, 0)
        base = dt.datetime(2013, 1, 1, 0, 0)
        return int((date - base).total_seconds() / (60 * freq))

    #################################################
    #                  TRAINING                     #
    #################################################

    def prepare_training(self):
        title = 'Training phase:'
        opts = {'Create new model.': self.__create_model,
                'Load existing model.': self.__load_model,
                'Check models that will be trained.': self.__models_to_train,
                'Ready for training.': self.__train_models,
                'Back to main menu.': self.menu.exit
               }
        stop = False
        while not stop:
            opt = self.menu.run(list(opts.keys()), title=title)
            if opt:
                stop = opts[opt]()

    def __create_model(self):
        title = 'Available models are:'
        mod_list = self.models.get_models_list()
        while True:
            opt = self.menu.run(mod_list, title=title)
            if opt:
                break
        model = self.models.model(opt)
        model.mod_name = self.helper.read('Please enter a new model name')
        self.__prep_model(model)

    def __load_model(self, prep_train=True):
        title = 'Available models are:'
        mod_list = sorted([m for m in os.listdir(self.models_path) 
                           if m[0] != '.' and os.path.isdir(os.path.join(self.models_path, m))])
        while True:
            opt = self.menu.run(mod_list, title=title)
            if opt:
                mod_path = os.path.join(self.models_path, opt, opt + '.h5')
                if os.path.exists(mod_path):
                    break
                else:
                    self.helper._print('Wrong file name! -> {}'.format(mod_path))
        do_compile = self.helper.read('Compile model? [y/n]')
        if self.helper.read('See model summary? [y/n]') == 'y':
            model.summary()
        model = self.models.load_model(mod_path, True if do_compile == 'y' else False)
        model.mod_name = opt
        if prep_train: # for training
            self.__prep_model(model)
        else:          # for testing
            return model

    def __prep_model(self, model):
        epochs = self.helper.read('Number of epochs', cast=int)
        self.mod_queue.append((model, epochs))
        self.helper._print('Model added to queue.')
        self.helper._continue()

    def __models_to_train(self):
        for idx, (model, epochs) in enumerate(self.mod_queue):
            self.helper._print('\t{}. {}: {} epochs'.format(idx+1, model.mod_name, epochs))
        self.helper._continue()

    def __train_models(self, do_test=True):
        idx = 0
        while len(self.mod_queue) > 0:
            idx +=1
            model, epochs = self.mod_queue.pop(0)
            self.helper._print('{}. Starting to train model {}...'.format(idx, model.mod_name))
            if not os.path.exists(os.path.join(self.models_path, model.mod_name)):
                os.mkdir(os.path.join(self.models_path, model.mod_name))
            tic = time.time()
            model = self.__train(model, epochs)
            train_duration_s = time.time() - tic
            fname = os.path.join(self.models_path, model.mod_name, model.mod_name + '.h5')
            # If model already existed, add those previous epochs
            if os.path.exists(fname):
                with tb.open_file(fname, 'r') as h5_mod:
                    node = h5_mod.get_node('/')
                    epochs += node._v_attrs['epochs']
            model.save(fname)
            # For reproducibility:
            self.__add_meta(fname, model.mod_name, epochs, train_duration_s)
            # After training, do tests
            if do_test:
                self.__plot_model(model)
                self.__plot_loss(model)
                self.__test(model)

    def __train(self, model, epochs):
        # Define parameters for the generator
        params = {'n_x': self.n_x,
                  'n_y': self.n_y,
                  'shift': self.shift,
                  'time_aware': self.time_aware,
                  'dataset_d': self.dataset_d,
                  'batch_size': self.batch_size,
                  'target_ids': self.zone_ids,
                  'X_shape': model.layers[0].input_shape[0] if 'map' in self.kind else model.layers[0].input_shape,
                  'y_shape': model.layers[-1].output_shape
                 }
        train_generator = DataGenerator(extent=self.train_extent, **params)
        val_generator = DataGenerator(extent=self.val_extent, **params)
        log_file = os.path.join(self.models_path, model.mod_name, model.mod_name + '_logs.txt') 
        model.fit(x=train_generator,
                  validation_data=val_generator,
                  epochs=epochs,
                  callbacks=[HistoryCallback(mod_name=model.mod_name,
                                             log_file=log_file,
                                             )],
                  use_multiprocessing=True, workers=0)
        return model

    def __add_meta(self, fname, mod_name, epochs, train_duration_s=0):
        # Add metadata to the model
        with tb.open_file(fname, 'a') as h5_mod:
            node = h5_mod.get_node('/')
            node._v_attrs['name'] = mod_name
            node._v_attrs['epochs'] = epochs
            node._v_attrs['train_duration_s'] = train_duration_s
            # Data
            node._v_attrs['time_gran'] = self.time_gran
            node._v_attrs['dataset_d'] = self.dataset_d
            node._v_attrs['city'] = self.city
            node._v_attrs['kind'] = self.kind
            node._v_attrs['dset_conv'] = self.dset_conv
            node._v_attrs['zone_ids'] = self.zone_ids
            # Model
            node._v_attrs['batch_size'] = self.batch_size
            node._v_attrs['n_x'] = self.n_x
            node._v_attrs['n_y'] = self.n_y
            node._v_attrs['shift'] = self.shift
            node._v_attrs['time_aware'] = self.time_aware
            node._v_attrs['train_extent'] = self.train_extent
            node._v_attrs['val_extent'] = self.val_extent
            node._v_attrs['test_extent'] = self.test_extent
            # Optimizer
            node._v_attrs['optimizer'] = self.models.get_optimizer()
            node._v_attrs['loss'] = self.models.get_loss()
            node._v_attrs['metrics'] = self.models.get_metrics()
            node._v_attrs['lr'] = self.models.get_lr()

    def __add_attr(self, mod_name, attr, value):
        # Add single attribute
        fname = os.path.join(self.models_path, mod_name, mod_name + '.h5')
        with tb.open_file(fname, 'a') as h5_mod:
            node = h5_mod.get_node('/')
            node._v_attrs[attr] = value

    def __get_attr(self, mod_name, attr):
        # Get a single attribute
        fname = os.path.join(self.models_path, mod_name, mod_name + '.h5')
        with tb.open_file(fname, 'r') as h5_mod:
            node = h5_mod.get_node('/')
            return node._v_attrs[attr]


    #################################################
    #                  TESTING                      #
    #################################################

    def prepare_test(self):
        model = self.__load_model(prep_train=False)
        title = 'Testing phase for {}:'.format(model.mod_name)
        opts = {'Plot model graph.': self.__plot_model,
                'Check model hyperparameters.': self.__model_hyperparams,
                'Plot model loss and metrics.': self.__plot_loss,
                'Test on (unseen) years.': self.__test,
                'Test on (unseen) years using trip counts.': self.__test_counts,
                'Test and plot a certain day.': self.__test_date,
                'Robustness test.': self.__robustness_test,
                'Back to main menu.': self.menu.exit,
               }
        stop = False
        while not stop:
            opt = self.menu.run(list(opts.keys()), title=title)
            if opt:
                stop = opts[opt](model)
                self.helper._continue()

    def __plot_model(self, model):
        aux_path = os.path.join(self.models_path, model.mod_name, model.mod_name + '.pdf')
        self.models.plot_model(model, aux_path)

    def __model_hyperparams(self, model):
        fname = os.path.join(self.models_path, model.mod_name, model.mod_name + '.h5')
        with tb.open_file(fname, 'a') as h5_mod:
            node = h5_mod.get_node('/')
            for idx, attr in enumerate(node._v_attrs._v_attrnames):
                if '_config' not in attr:
                    self.helper._print('\t{:02}. {}: {}'.format(idx+1, attr, node._v_attrs[attr]))

    def __plot_loss(self, model):
        # Load history file
        aux_path = os.path.join(self.models_path, model.mod_name, model.mod_name)
        hist = pd.read_csv(aux_path + '_hist.csv')
        days = hist['duration [s]'].sum() // (24 * 3600)
        train_duration = '{} days and {}'.format(days, time.strftime('%H:%M:%S', 
                                                                     time.gmtime(hist['duration [s]'].sum())))
        self.helper._print('Total training time for model {} was {}'.format(model.mod_name, train_duration))
        self.__add_attr(model.mod_name, 'train_duration', train_duration)
        loss_names = {'mse': 'Mean Squared Error',
                      'val_mse': 'Mean Squared Error',
                      'mae': 'Mean Absolute Error',
                      'val_mae': 'Mean Absolute Error',
                      'acc': 'Accuracy',
                      'val_acc': 'Accuracy',
                      'categorical_crossentropy': 'Categorical Crossentropy',
                      'msle': 'Mean Squared Logarithmic Error',
                      'val_msle': 'Mean Squared Logarithmic Error'
                     }
        series_d = dict()
        # Here we reverse the columns for a better order in the plot and legend
        print(hist.columns[2:])
        for hkey in reversed(hist.columns[3:]): # after epoch & duration
            if 'loss' in hkey: # Skip loss due to the keras bug
                continue
            metric_name = loss_names[hkey]
            if self.models.get_loss() in hkey:
                metric_name += ' - Loss'
            t_set = 'Validation set' if 'val_' in hkey else 'Training set'
            if metric_name in series_d.keys():
                series_d[metric_name][t_set] = hist.loc[:, hkey].values
            else:
                series_d[metric_name] = {t_set: hist.loc[:, hkey].values}
        self.plotter.plot_series(series_d, ('epoch', ''), int_ticker=True,
                                 out_path=aux_path + '_loss.pdf')#yscale='log'

    def __test(self, model):
        # Define parameters for the generator
        params = {'n_x': self.n_x,
                  'n_y': self.n_y,
                  'shift': self.shift,
                  'time_aware': self.time_aware,
                  'dataset_d': self.dataset_d,
                  'batch_size': self.batch_size,
                  'target_ids': self.zone_ids,
                  'X_shape': model.layers[0].input_shape[0] if 'map' in self.kind else model.layers[0].input_shape,
                  'y_shape': model.layers[-1].output_shape
                 }
        test_generator = DataGenerator(extent=self.test_extent, **params)
        ret = model.evaluate(test_generator, verbose=1,
                             use_multiprocessing=True, workers=0)
        for id_m, metric in enumerate(self.models.get_metrics()):
            self.__add_attr(model.mod_name, 'test_{}'.format(metric), round(ret[id_m], 5))

    def __conv(self, X):
        if 'count' in self.dataset_path:
            pass
        elif 'norm_abs_050' in self.dataset_path:
            X = X / 50
            X[X > 1] = 1
        elif 'norm_abs' in self.dataset_path:
            X = X / np.array([418, 84])
        elif 'norm' in self.dataset_path:
            aux = self.df_stats
            aux.loc[aux['max'] == 0, 'max'] = 1
            X = np.divide(X, aux['max'].values[self.zone_ids])
        elif 'stand_abs' in self.dataset_path:
            X = (X - 0.8202286089339693) / 6.217741360934683
        elif 'stand' in self.dataset_path:
            X = np.divide(np.substract(X, self.df_stats['mean'].values[self.zone_ids]),
                          self.df_stats['std'].values[self.zone_ids])
        return X

    def __to_count(self, y):
        if 'count' in self.dset_conv:
            pass
        elif 'norm_abs_050' in self.dset_conv:
            y = y * 50
        elif 'norm_abs' in self.dset_conv:
            y = y * np.array([418, 84])
        elif 'norm' in self.dset_conv:
            y = np.multiply(y, self.df_stats['max'].values[self.zone_ids])
        elif 'stand_abs' in self.dset_conv:
            y = y * 6.217741360934683 + 0.8202286089339693
        elif 'stand' in self.dset_conv:
            y = np.add(np.multiply(y, self.df_stats['std'].values[self.zone_ids]),
                       self.df_stats['mean'].values[self.zone_ids])
        return y

    def __test_counts(self, model):
        # Define parameters for the generator
        params = {'n_x': self.n_x,
                  'n_y': self.n_y,
                  'shift': self.shift,
                  'time_aware': self.time_aware,
                  'dataset_d': self.dataset_d,
                  'batch_size': self.batch_size,
                  'target_ids': self.zone_ids,
                  'X_shape': model.layers[0].input_shape[0] if 'map' in self.kind else model.layers[0].input_shape,
                  'y_shape': model.layers[-1].output_shape
                 }
        dset_taxi_count = {'time_series': ['../data/chicago/clean/15m_flat_taxi_count.h5']}
        dset_bike_count = {'time_series': ['../data/chicago/clean/15m_flat_bike_count.h5']}
        is_map = True if 'map' in self.kind else False
        n_zones = (self.xy_taxi.shape[0], self.xy_bike_g.shape[0])
        params_t, params_b = params.copy(), params.copy()
        params_t['dataset_d'], params_b['dataset_d'] = dset_taxi_count, dset_bike_count
        params_t['time_aware'], params_b['time_aware'] = False, False
        params_t['X_shape'], params_b['X_shape'] = (None, self.n_x, n_zones[0], 1), (None, self.n_x, n_zones[1], 1)
        params_t['y_shape'], params_b['y_shape'] = (None, self.n_y, n_zones[0], 1), (None, self.n_y, n_zones[1], 1)
        sae_mod_taxi, sse_mod_taxi = np.zeros((self.n_y, n_zones[0])), np.zeros((self.n_y, n_zones[0]))
        sae_mod_bike, sse_mod_bike = np.zeros((self.n_y, n_zones[1])), np.zeros((self.n_y, n_zones[1]))
        n_samp = 0
        pred_generator = DataGenerator(extent=self.test_extent, **params)
        taxi_generator = DataGenerator(extent=self.test_extent, **params_t)
        bike_generator = DataGenerator(extent=self.test_extent, **params_b)
        for n_batch, ((X_b, _), (_, y_t), (_, y_b)) in enumerate(zip(pred_generator, taxi_generator, bike_generator)):
            y_pred = model.predict_on_batch(X_b)
            y_t, y_b = y_t[..., 0], y_b[..., 0]
            if is_map: # When working with maps, (de)interpolate the result for each dataset (taxis and bikes)
                # First, place map axes first and flatten along the remaining dimensions
                y_pred_taxi = y_pred[:, :, :, :, 0].transpose((3, 2, 0, 1)).reshape((self.map_shape[1] * self.map_shape[0], -1))
                # Then interpolate, undo the flatten and the transpose
                y_pred_taxi = griddata(self.grid, y_pred_taxi, self.xy_taxi, method='linear',
                                       fill_value=0).reshape((n_zones[0], -1, self.n_y)).transpose((1, 2, 0)) * 418
                # Now for bikes
                y_pred_bike = y_pred[:, :, self.lat_bike_g, self.lng_bike_g, 1] * 84
            # Converted predictions to count in the if before
            sae_mod_taxi += np.abs(y_t - y_pred_taxi).sum(axis=0)
            sae_mod_bike += np.abs(y_b - y_pred_bike).sum(axis=0)
            sse_mod_taxi += ((y_t - y_pred_taxi)**2).sum(axis=0)
            sse_mod_bike += ((y_b - y_pred_bike)**2).sum(axis=0)
            n_samp += y_b.shape[0]
        # Compute and save the MAE and RMSE for the model
        mae_count_taxi = sae_mod_taxi / n_samp
        mae_count_bike = sae_mod_bike / n_samp
        rmse_count_taxi = np.sqrt(sse_mod_taxi / n_samp)
        rmse_count_bike = np.sqrt(sse_mod_bike / n_samp)
        self.__add_attr(model.mod_name, 'mae_count_taxi', mae_count_taxi)
        self.__add_attr(model.mod_name, 'mae_count_bike', mae_count_bike)
        self.__add_attr(model.mod_name, 'rmse_count_taxi', rmse_count_taxi)
        self.__add_attr(model.mod_name, 'rmse_count_bike', rmse_count_bike)
        mae_w_count_taxi = mae_count_taxi * self.tnorm
        mae_w_count_bike = mae_count_bike * self.bnorm
        rmse_w_count_taxi = rmse_count_taxi * self.tnorm
        rmse_w_count_bike = rmse_count_bike * self.bnorm
        self.__add_attr(model.mod_name, 'mae_w_count_taxi', mae_w_count_taxi)
        self.__add_attr(model.mod_name, 'mae_w_count_bike', mae_w_count_bike)
        self.__add_attr(model.mod_name, 'rmse_w_count_taxi', rmse_w_count_taxi)
        self.__add_attr(model.mod_name, 'rmse_w_count_bike', rmse_w_count_bike)
        self.plotter.plot_boxplot(mae_w_count_taxi.T, x_labels=range(1, self.n_y + 1),
                                  labels=('Forecast horizon [{}in]'.format(self.time_gran), 'MAE of taxi trip counts'),
                                  title='MAE variation among zones for each horizon', yscale='log',
                                  out_path=os.path.join(self.models_path, model.mod_name, model.mod_name + '_taxi_mae_w.pdf'))
        self.plotter.plot_boxplot(mae_w_count_bike.T, x_labels=range(1, self.n_y + 1),
                                  labels=('Forecast horizon [{}in]'.format(self.time_gran), 'MAE of bike trip counts'),
                                  title='MAE variation among areas for each horizon', yscale='log',
                                  out_path=os.path.join(self.models_path, model.mod_name, model.mod_name + '_bike_mae_w.pdf'))
        self.plotter.plot_boxplot(rmse_w_count_taxi.T, x_labels=range(1, self.n_y + 1),
                                  labels=('Forecast horizon [{}in]'.format(self.time_gran), 'RMSE of taxi trip counts'),
                                  title='RMSE variation among zones for each horizon', yscale='log',
                                  out_path=os.path.join(self.models_path, model.mod_name, model.mod_name + '_taxi_rmse_w.pdf'))
        self.plotter.plot_boxplot(rmse_w_count_bike.T, x_labels=range(1, self.n_y + 1),
                                  labels=('Forecast horizon [{}in]'.format(self.time_gran), 'RMSE of bike trip counts'),
                                  title='RMSE variation among areas for each horizon', yscale='log',
                                  out_path=os.path.join(self.models_path, model.mod_name, model.mod_name + '_bike_rmse_w.pdf'))

    def __predict(self, model, day, month, year):
        # Define parameters for the generator
        params = {'n_x': self.n_x,
                  'n_y': self.n_y,
                  'shift': self.shift,
                  'time_aware': self.time_aware,
                  'dataset_d': self.dataset_d,
                  'batch_size': self.batch_size,
                  'target_ids': self.zone_ids,
                  'X_shape': model.layers[0].input_shape[0] if 'map' in self.kind else model.layers[0].input_shape,
                  'y_shape': model.layers[-1].output_shape
                 }
        dset_taxi_count = {'time_series': ['../data/chicago/clean/15m_flat_taxi_count.h5']}
        dset_bike_count = {'time_series': ['../data/chicago/clean/15m_flat_bike_count.h5']}
        is_map = True if 'map' in self.kind else False
        n_zones = (self.xy_taxi.shape[0], self.xy_bike_g.shape[0])
        start_id = self.__datetime_to_idx(dt.datetime(year, month, day, 0, 0)) - self.n_x - self.shift + 1
        end_id = self.__datetime_to_idx(dt.datetime(year, month, day, 23, 45)) + 1
        pred_generator = DataGenerator(extent=(start_id, end_id), **params)
        y_pred = model.predict(pred_generator, verbose=1, use_multiprocessing=True, workers=0)
        if is_map: # When working with maps, (de)interpolate the result for each dataset (taxis and bikes)
            # First, place map axes first and flatten along the remaining dimensions
            y_pred_taxi = y_pred[:, :, :, :, 0].transpose((3, 2, 0, 1)).reshape((self.map_shape[1] * self.map_shape[0], -1))
            # Then interpolate, undo the flatten and the transpose
            y_pred_taxi = griddata(self.grid, y_pred_taxi, self.xy_taxi, method='linear',
                                   fill_value=0).reshape((n_zones[0], -1, self.n_y)).transpose((1, 2, 0)) * 418
            # Now for bikes
            y_pred_bike = y_pred[:, :, self.lat_bike_g, self.lng_bike_g, 1] * 84
        hours = np.empty((y_pred.shape[0], self.n_y), dtype=object)
        taxi = np.empty((y_pred.shape[0], self.n_y, n_zones[0]))
        bike = np.empty((y_pred.shape[0], self.n_y, n_zones[1]))
        # Now, prepare ground truth
        with tb.open_file(dset_taxi_count['time_series'][0], mode='r') as h5_taxi, \
             tb.open_file(dset_bike_count['time_series'][0], mode='r') as h5_bike:
            for horizon in range(self.n_y):
                y_l = start_id + self.n_x + self.shift + horizon - 1
                slc = slice(y_l, y_l + y_pred.shape[0])
                hours[:, horizon] = np.array([self.__idx_to_datetime(id_t, 2013) for id_t in range(slc.start, slc.stop)])
                taxi[:, horizon] = h5_taxi.get_node('/2013')[slc, self.zone_ids]
                bike[:, horizon] = h5_bike.get_node('/2013')[slc, self.zone_ids]
        return hours, (y_pred_taxi, y_pred_bike), (taxi, bike)

    def __test_date(self, model, date=None, zones_ids=None):
        if not date:
            date = self.helper.read('Date (dd/mm/yyyy)')
        is_map = True if 'map' in self.kind else False
        if is_map or len(self.zone_ids) == len(self.zones):
            test_zones_t = [756, 799, 140, 800, 10]
            test_zones_b = [592, 595, 140, 599, 708]
        elif len(self.zone_ids) > 5:
            test_zones = sorted(np.random.choice(len(self.zone_ids), 5, replace=False))
        else:
            test_zones = np.arange(len(self.zone_ids))
        d, m, y = date.split('/')
        hours, (y_pred_taxi, y_pred_bike), (taxi, bike) = self.__predict(model, *[int(n) for n in (d, m, y)])
        # Obtain min and max values for the limits of the plot
        #min_val = min(min(true.ravel()), min(pred.ravel())) - 0.05
        #max_val = max(max(true.ravel()), max(pred.ravel())) + 0.05
        # Create folder for plots
        plt_path = os.path.join(self.models_path, model.mod_name, 'day_plots')
        if not os.path.exists(plt_path):
            os.mkdir(plt_path)
        a = dt.datetime(*[int(n) for n in (y, m, d)], hour=0, minute=0)
        b = dt.datetime(*[int(n) for n in (y, m, d)], hour=23, minute=59)
        series_t, series_b = dict(), dict()
        for horizon in range(self.n_y):
            for id_z in test_zones_t: # For taxis
                mae = mean_absolute_error(taxi[:, horizon, id_z], y_pred_taxi[:, horizon, id_z])
                title = 'True vs predicted taxi trips for zone {} on {}, \nHorizon = {}min, R^2 = {:0.2}, MAE = {:0.2}'.format(
                    self.zones[id_z], date, (self.shift + horizon)*15,
                    r2_score(taxi[:, horizon, id_z], y_pred_taxi[:, horizon, id_z]), mae)
                series_t[title] = {'pred': y_pred_taxi[:, horizon, id_z],
                                   'true': taxi[:, horizon, id_z]}
            for id_z in test_zones_b: # For bikes
                mae = mean_absolute_error(bike[:, horizon, id_z], y_pred_bike[:, horizon, id_z])
                title = 'True vs predicted bike trips for zone {} on {}, \nHorizon = {}min, R^2 = {:0.2}, MAE = {:0.2}'.format(
                    str(self.xy_bike_g[id_z]), date, (self.shift + horizon)*15,
                    r2_score(bike[:, horizon, id_z], y_pred_bike[:, horizon, id_z]), mae)
                series_b[title] = {'pred': y_pred_bike[:, horizon, id_z],
                                   'true': bike[:, horizon, id_z]}
        self.plotter.plot_series(series_t, ('Hour', 'Taxi trip counts'), date_ticker=hours, scale=0.5,
                                 dims=(self.n_y, len(test_zones_t)), xlim=(a, b),
                                 out_path=plt_path + '/{}_series_taxi.png'.format(y+m+d))
        self.plotter.plot_series(series_b, ('Hour', 'Bike trip counts'), date_ticker=hours, scale=0.5,
                                 dims=(self.n_y, len(test_zones_b)), xlim=(a, b),
                                 out_path=plt_path + '/{}_series_bike.png'.format(y+m+d))

    def __robustness_test(self, model, up_to=0.5 + 0.01, n_reps=10):
        # Define parameters for the generator
        params = {'n_x': self.n_x,
                  'n_y': self.n_y,
                  'shift': self.shift,
                  'time_aware': self.time_aware,
                  'dataset_d': self.dataset_d,
                  'batch_size': self.batch_size,
                  'target_ids': self.zone_ids,
                  'X_shape': model.layers[0].input_shape[0] if 'map' in self.kind else model.layers[0].input_shape,
                  'y_shape': model.layers[-1].output_shape
                 }
        n_zones = (self.xy_taxi.shape[0], self.xy_bike_g.shape[0])
        # Define parameters for the original dataset (before interpolation)
        dset_taxi = {'time_series': ['../data/chicago/clean/15m_flat_taxi_norm_abs.h5'],
                     'weather': self.dataset_d['weather'], 'holidays': self.dataset_d['holidays']}
        dset_bike = {'time_series': ['../data/chicago/clean/15m_flat_bike_norm_abs.h5'],
                     'weather': self.dataset_d['weather'], 'holidays': self.dataset_d['holidays']}
        params_t, params_b = params.copy(), params.copy()
        params_t['dataset_d'], params_b['dataset_d'] = dset_taxi, dset_bike
        params_t['X_shape'], params_b['X_shape'] = (None, self.n_x, n_zones[0], 1), (None, self.n_x, n_zones[1], 1)
        params_t['y_shape'], params_b['y_shape'] = (None, self.n_y, n_zones[0], 1), (None, self.n_y, n_zones[1], 1)
        # Now for the original dataset with counts
        dset_taxi_count = {'time_series': ['../data/chicago/clean/15m_flat_taxi_count.h5']}
        dset_bike_count = {'time_series': ['../data/chicago/clean/15m_flat_bike_count.h5']}
        params_tc, params_bc = params_t.copy(), params_b.copy()
        params_tc['dataset_d'], params_bc['dataset_d'] = dset_taxi_count, dset_bike_count
        params_b['time_aware'], params_tc['time_aware'], params_bc['time_aware'] = False, False, False
        # Other required stuff
        reps = ['rep_{:02}'.format(idx) for idx in range(n_reps)]
        excl = np.arange(start=0.05, stop=up_to, step=0.05)
        rob_path = os.path.join(self.models_path, model.mod_name, 'robustness-test')
        if not os.path.exists(rob_path):
            os.mkdir(rob_path)
        rwc_taxi = self.__get_attr(model.mod_name, 'rmse_w_count_taxi')
        rwc_bike = self.__get_attr(model.mod_name, 'rmse_w_count_bike')
        # To keep the results
        arr_err_taxi = np.empty(shape=(len(excl), n_reps, self.n_y, n_zones[0]))
        arr_sks_taxi = np.empty(shape=(len(excl), n_reps, self.n_y))
        arr_err_bike = np.empty(shape=(len(excl), n_reps, self.n_y, n_zones[1]))
        arr_sks_bike = np.empty(shape=(len(excl), n_reps, self.n_y))
        with tb.open_file(dset_taxi_count['time_series'][0], mode='r') as h5_taxi, \
             tb.open_file(dset_bike_count['time_series'][0], mode='r') as h5_bike:
            for id_excl, p_excl in enumerate(excl):
                for id_rep, n_rep in enumerate(reps):
                    print('\n\t\t For {} with {}\% excluded'.format(n_rep, p_excl*100))
                    chosen_taxi = np.sort(np.random.choice(n_zones[0], size=int(n_zones[0] * (1 - p_excl)), replace=False))
                    chosen_bike = np.sort(np.random.choice(n_zones[1], size=int(n_zones[1] * (1 - p_excl)), replace=False))
                    sse_mod_taxi, sse_mod_bike = np.zeros((self.n_y, n_zones[0])), np.zeros((self.n_y, n_zones[1]))
                    n_samp = 0
                    taxi_gen = DataGenerator(extent=self.test_extent, **params_t)
                    bike_gen = DataGenerator(extent=self.test_extent, **params_b)
                    taxic_gen = DataGenerator(extent=self.test_extent, **params_tc)
                    bikec_gen = DataGenerator(extent=self.test_extent, **params_bc)
                    for n_batch, ((X_t, _), (X_b, _), (_, y_tc), (_, y_bc)) in enumerate(zip(taxi_gen, bike_gen, taxic_gen, bikec_gen)):
                        y_tc, y_bc = y_tc[..., 0], y_bc[..., 0]
                        bs = y_tc.shape[0]
                        # For taxi [batch, n_x, zone]
                        aux_t = X_t['time_series'][:, :, chosen_taxi]
                        aux_t = aux_t.reshape((-1, len(chosen_taxi))).T
                        aux_t = griddata(self.xy_taxi[chosen_taxi], aux_t, (self.X, self.Y),
                                         method='linear', fill_value=0).transpose((2, 1, 0))
                        aux_t = aux_t.reshape((bs, self.n_x, *self.map_shape))
                        X_t['time_series'] = np.expand_dims(aux_t, axis=-1)
                        # For bike
                        aux_b = X_b['time_series'][:, :, chosen_bike]
                        aux_b = aux_b.reshape((-1, len(chosen_bike))).T
                        aux_b = griddata(self.xy_bike_g[chosen_bike], aux_b, (self.X, self.Y),
                                         method='linear', fill_value=0).transpose((2, 1, 0))
                        aux_b = aux_b.reshape((bs, self.n_x, *self.map_shape))
                        X_b['time_series'] = np.expand_dims(aux_b, axis=-1)
                        # Combine them
                        X_t['time_series'] = np.concatenate([X_t['time_series'], X_b['time_series']], axis=-1)
                        # Predict and de-interpolate
                        y_pred = model.predict_on_batch(X_t)
                        # First, place map axes first and flatten along the remaining dimensions
                        y_pred_taxi = y_pred[:, :, :, :, 0].transpose((3, 2, 0, 1)).reshape((self.map_shape[1] * self.map_shape[0], -1))
                        # Then interpolate, undo the flatten and the transpose
                        y_pred_taxi = griddata(self.grid, y_pred_taxi, self.xy_taxi, method='linear',
                                               fill_value=0).reshape((n_zones[0], -1, self.n_y)).transpose((1, 2, 0)) * 418
                        # Now for bikes
                        y_pred_bike = y_pred[:, :, self.lat_bike_g, self.lng_bike_g, 1] * 84
                        # Evaluate error
                        sse_mod_taxi += ((y_tc - y_pred_taxi)**2).sum(axis=0)
                        sse_mod_bike += ((y_bc - y_pred_bike)**2).sum(axis=0)
                        n_samp += bs
                    # Compute and save the RMSE for the model
                    rmse_mod_taxi = np.sqrt(sse_mod_taxi / n_samp) * self.tnorm
                    rmse_mod_bike = np.sqrt(sse_mod_bike / n_samp) * self.bnorm
                    arr_err_taxi[id_excl, id_rep] = rmse_mod_taxi
                    arr_sks_taxi[id_excl, id_rep] = (1 - rmse_mod_taxi.mean(axis=1) / rwc_taxi.mean(axis=1)) * 100
                    arr_err_bike[id_excl, id_rep] = rmse_mod_bike
                    arr_sks_bike[id_excl, id_rep] = (1 - rmse_mod_bike.mean(axis=1) / rwc_bike.mean(axis=1)) * 100
        np.save(os.path.join(rob_path, 'rwc_per_horizon_taxi.npy'), arr_err_taxi)
        np.save(os.path.join(rob_path, 'skill_per_horizon_taxi.npy'), arr_sks_taxi)
        np.save(os.path.join(rob_path, 'rwc_per_horizon_bike.npy'), arr_err_bike)
        np.save(os.path.join(rob_path, 'skill_per_horizon_bike.npy'), arr_sks_bike)
        self.__plot_robustness(arr_sks_taxi, name='Taxi trips', tickers=[int(i*100) for i in excl],
                               plt_path=os.path.join(rob_path, model.mod_name + '_rob_test_taxi.pdf'))
        self.__plot_robustness(arr_sks_bike, name='Bicycle rides', tickers=[int(i*100) for i in excl],
                               plt_path=os.path.join(rob_path, model.mod_name + '_rob_test_bike.pdf'))

    def __plot_robustness(self, arr, name, tickers, plt_path):
        series_d = dict(zip(['{}min'.format(15 * (self.shift + h)) for h in range(self.n_y)],
                            (*arr.mean(axis=1).T,)))
        self.plotter.plot_series({'Skill with respect to the original model - {}'.format(name): series_d},
                                 ('Percentage of excluded zones', 'Worsening [%]'), tickers=tickers, style='o-',
                                 out_path=plt_path)


    #################################################
    #                 UPDATE OPTIONS                #
    #################################################

    def update_options(self):
        title = 'The options that can be changed are:'
        stop = False
        while not stop:
            opts = {'Time granularity: {}'.format(self.get_time_gran()): self.set_time_gran,
                    'City: {}'.format(self.get_city()): self.set_city,
                    'Dataset conversion: {}'.format(self.get_dset_conv()): self.set_dset_conv,
                    'Indexes of the zones: {} in total'.format(self.map_shape): self.set_zone_ids,
                    'Batch size: {}'.format(self.get_batch_size()): self.set_batch_size,
                    'n_x: {}'.format(self.get_n_x()): self.set_n_x,
                    'n_y: {}'.format(self.get_n_y()): self.set_n_y,
                    'shift: {}'.format(self.get_shift()): self.set_shift,
                    'Time aware: {}'.format(self.get_time_aware()): self.set_time_aware,
                    'Optimizer: {}'.format(self.models.get_optimizer()): self.models.set_optimizer,
                    'Loss: {}'.format(self.models.get_loss()): self.models.set_loss,
                    'Metrics: {}'.format(self.models.get_metrics()): self.models.set_metrics,
                    'Learning rate: {}'.format(self.models.get_lr()): self.models.set_lr,
                    'Back to main menu.': self.menu.exit,
                }
            opt = self.menu.run(list(opts.keys()), title=title)
            value = self.helper.read('Value')
            if opt:
                stop = opts[opt](value)


    #################################################
    #                 RESULTS TABLE                 #
    #################################################

    def results_table(self, fname='results.csv', auto=True):
        df = pd.DataFrame(columns=['Model', 'Target',
                                   'Time granularity',
                                   'n_x', 'n_y', 'shift',
                                   'MSE test', 'MAE test',
                                   'MMWC taxi', 'MMMWC taxi',
                                   'MMWC bike', 'MMMWC bike',
                                   'MRWC taxi', 'MMRWC taxi',
                                   'MMWC bike', 'MMMWC bike',
                                   'Epochs', 'Loss', 'Optimizer',
                                   'Learning rate', 'Train duration', 'Train duration [s]'])
        for idx, model in enumerate(os.listdir(self.models_path)):
            model_path = os.path.join(self.models_path, model, model + '.h5')
            if not os.path.exists(model_path):
                continue
            with tb.open_file(model_path, 'r') as h5_mod:
                node = h5_mod.root
                if 'test_mse' not in node._v_attrs:
                    continue
                name, zone = 'unknown', ''
                df.loc[idx] = [node._v_attrs['name'],
                               self.target_d[node._v_attrs['kind'] + '_' + node._v_attrs['dset_conv']] + zone,
                               node._v_attrs['time_gran'],
                               node._v_attrs['n_x'], node._v_attrs['n_y'], node._v_attrs['shift'],
                               node._v_attrs['test_mse'], node._v_attrs['test_mae'],
                               node._v_attrs['mae_w_count_taxi'].mean(axis=1).round(3),
                               node._v_attrs['mae_w_count_taxi'].mean(axis=1).mean(),
                               node._v_attrs['mae_w_count_bike'].mean(axis=1).round(3),
                               node._v_attrs['mae_w_count_bike'].mean(axis=1).mean(),
                               node._v_attrs['rmse_w_count_taxi'].mean(axis=1).round(3),
                               node._v_attrs['rmse_w_count_taxi'].mean(axis=1).mean(),
                               node._v_attrs['rmse_w_count_bike'].mean(axis=1).round(3),
                               node._v_attrs['rmse_w_count_bike'].mean(axis=1).mean(),
                               node._v_attrs['epochs'], node._v_attrs['loss'],
                               node._v_attrs['optimizer'].__name__, node._v_attrs['lr'],
                               node._v_attrs['train_duration'],
                               node._v_attrs['train_duration_s']
                              ]
        df.sort_values(['Time granularity', 'Target', 'n_x', 'n_y', 'shift', 'Model', 'Optimizer', 'Loss'], inplace=True)
        df.to_csv(os.path.join(self.models_path, fname), index=False)
        self.helper._print('{} saved at {}'.format(fname, self.models_path))
        if auto:
            self.helper._continue()


    #################################################
    #              GETTERS & SETTERS                #
    #################################################

    # Time granularity
    def get_time_gran(self):
        return self.time_gran
    def set_time_gran(self, x):
        curr_time_gran = self.time_gran
        self.time_gran = x
        self.dataset_d['time_series'] = [dset.replace(curr_time_gran, self.time_gran) for dset in self.dataset_d['time_series']]
        self.models_path = self.models_path.replace(curr_time_gran, self.time_gran)
        self.data_path = self.data_path.replace(curr_time_gran, self.time_gran)

    # n_x
    def get_n_x(self):
        return self.n_x
    def set_n_x(self, x):
        self.n_x = int(x)
        self.models.set_n_x(self.n_x)

    # n_y
    def get_n_y(self):
        return self.n_y
    def set_n_y(self, x):
        self.n_y = int(x)
        self.models.set_n_y(self.n_y)

    # shift
    def get_shift(self):
        return self.shift
    def set_shift(self, x):
        self.shift = int(x)

    # Map shape for convolutionals
    def get_kind(self):
        return self.kind
    def set_kind(self, x):
        if 'map' in x:
            # Load longitude and latitude for the interpolation
            st_path = os.path.join(self.other_path, 'zone-centroids-taxi.csv')
            self.xy_taxi = pd.read_csv(st_path).loc[:, ['longitude', 'latitude']].values
            self.lng_taxi = self.xy_taxi[:, 0]
            self.lat_taxi = self.xy_taxi[:, 1]
            st_path = os.path.join(self.other_path, 'station-locations-bike.csv')
            self.xy_bike = pd.read_csv(st_path).loc[:, ['lng', 'lat']].values
            self.lng_bike = self.xy_bike[:, 0]
            self.lat_bike = self.xy_bike[:, 1]
            st_path = os.path.join(self.other_path, 'grid-locations-bike.csv')
            self.xy_bike_g = pd.read_csv(st_path).loc[:, ['lng_int', 'lat_int']].values
            self.lng_bike_g = self.xy_bike_g[:, 0]
            self.lat_bike_g = self.xy_bike_g[:, 1]
            self.map_shape = tuple([int(aux) for aux in self.kind.split('_')[-2:]])
            offset = 0.002
            xrange = (min(self.lng_taxi.min(), self.lng_bike.min()), max(self.lng_taxi.max(), self.lng_bike.max()))
            yrange = (min(self.lat_taxi.min(), self.lat_bike.min()), max(self.lat_taxi.max(), self.lat_bike.max()))
            xstep = (xrange[1] - xrange[0] + 2 * offset) / self.map_shape[1]
            ystep = (yrange[1] - yrange[0] + 2 * offset) / self.map_shape[0]
            xnew = np.linspace(xrange[0] - offset + xstep/2, xrange[1] + offset - xstep/2, self.map_shape[1])
            ynew = np.linspace(yrange[0] - offset + ystep/2, yrange[1] + offset - ystep/2, self.map_shape[0])
            self.X, self.Y = np.meshgrid(xnew, ynew, indexing='ij')
            self.grid = np.stack((self.X.ravel(), self.Y.ravel())).T
        else:
            self.zone_ids = np.arange(801)
            self.map_shape = len(self.zone_ids)
        self.models.set_map_shape(self.map_shape)

    # City
    def get_city(self):
        return self.city
    def set_city(self, x):
        curr_city = self.city
        self.city = x
        self.other_path.replace(curr_city, self.city)
        self.data_path.replace(curr_city, self.city)
        self.dataset_d['time_series'] = [dset.replace(curr_city, self.city) for dset in self.dataset_d['time_series']]
        self.df_stats = pd.read_csv(os.path.join(self.other_path, 'metrics-per-zone.csv'), index_col=0)

    # Dataset conversion
    def get_dset_conv(self):
        return self.dset_conv
    def set_dset_conv(self, x):
        curr_dset_conv = self.dset_conv
        self.dset_conv = x
        self.target = self.target_d['{}_{}'.format(self.kind, self.dset_conv)]
        self.models.set_act_out('sigmoid' if 'norm' in self.dset_conv else 'linear')
        self.other_path.replace(curr_dset_conv, self.city)
        self.data_path.replace(curr_dset_conv, self.city)
        self.models_path.replace(curr_dset_conv, self.city)
        self.dataset_d['time_series'] = [dset.replace(curr_dset_conv, self.dset_conv) for dset in self.dataset_d['time_series']]

    # Zone indexes
    def get_zone_ids(self):
        return self.zone_ids
    def set_zone_ids(self, x):
        self.zone_ids = eval(x)
        self.map_shape = len(self.zone_ids)
        self.models.set_map_shape(self.map_shape[::-1])

    # Batch size
    def get_batch_size(self):
        return self.batch_size
    def set_batch_size(self, x):
        self.batch_size = int(x)

    # Time aware (input with time, day of the week...)
    def get_time_aware(self):
        return self.time_aware
    def set_time_aware(self, x):
        self.time_aware = eval(x)
        self.models.set_time_aware(self.time_aware)
