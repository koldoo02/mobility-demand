import numpy as np
import pandas as pd
import tables as tb
import datetime as dt
import time, os
from keras.utils import Sequence
from keras.callbacks import Callback


class HistoryCallback(Callback):
    '''
    Optional functions to define can be:
       - on_(epoch|batch|train)_(begin|end)
    Expected arguments:
       - (self, (epoch|batch), logs={})
    '''
    def __init__(self, mod_name, log_file):
        self.mod_name = mod_name
        self.log_file = log_file

    def on_train_begin(self, logs={}):
        self.epochs = []
        self.log_list = []
        self.times = []
        # Get current time in UTC+1
        now = time.strftime('%A, %d %b %Y %H:%M:%S', time.gmtime(time.time() + 3600 * 1))
        with open(self.log_file, 'a+') as f_log:
            f_log.write('\n\nStarting to train model {} on {}...'.format(self.mod_name, now))

    def on_epoch_begin(self, epoch, logs={}):
        self.init_time = time.time()
        with open(self.log_file, 'a+') as f_log:
            f_log.write('\nStarting epoch {}...'.format(epoch))

    def on_epoch_end(self, epoch, logs={}):
        end_time = round(time.time() - self.init_time)
        self.epochs.append(epoch)
        self.log_list.append(logs.copy())
        self.times.append(end_time)
        with open(self.log_file, 'a+') as f_log:
            f_log.write('\nIt took {}s'.format(end_time))

    def on_train_end(self, logs={}):
        hist = pd.DataFrame()
        hist['epoch'] = self.epochs
        hist['duration [s]'] = self.times
        # Iterate on log keys (typically: loss, val_loss...)
        for col in self.log_list[0].keys():
            hist[col] = [log[col] for log in self.log_list]
        history_file = self.log_file.replace('logs.txt', 'hist.csv')
        if os.path.exists(history_file):
            prev_hist = pd.read_csv(history_file)
            hist = pd.concat([prev_hist, hist])
        hist.set_index('epoch').to_csv(history_file, index=True)

    
class DataGenerator(Sequence):
    '''
    Data generator for Keras (fit_generator). Based on:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    
    def __init__(self, dataset_d, extent, n_x, n_y, shift, target_ids,
                 batch_size, X_shape, y_shape, time_aware=False, group='/2013'):
        '''Initialization of the generator object
        
        Keyword arguments:
        dataset_d -- dictionary with dataset paths (current keys: sources, weather and holidays)
        extent -- starting and finishing time indexes
        n_x -- length of the input window
        n_y -- length of the output window
        shift -- distance between input and output window (>=1)
        target_ids -- indexes of the columns/subset of items to work with
        batch_size -- size of the batch to be returned each time
        X_shape -- dict of expected shapes for X, corresponding to dataset_d
        y_shape -- expected shape for y
        time_aware -- whether to include time information in X
        group -- name of the HDF5 group where to retrieve data from
        '''
        # For files and paths
        self.time_series = dataset_d['time_series']
        self.weather = dataset_d['weather'] if 'weather' in dataset_d.keys() else None
        self.holidays = dataset_d['holidays'] if 'holidays' in dataset_d.keys() else None
        if self.holidays is not None:
            self.holidays = pd.read_csv(self.holidays[0], parse_dates=[1])
        self.group = group
        self.time_aware = time_aware
        # For time and horizons
        # Following the convention of np, the end_id is not reached
        self.start_id, self.end_id = extent
        self.n_instants = self.end_id - self.start_id
        self.n_x = n_x
        self.n_y = n_y
        self.shift = shift
        self.ts_pointer = self.start_id
        # Width of a (X + y) time window
        self.width = self.n_x + self.n_y + self.shift - 1
        # Total number of returned items by the generator
        self.n_preds = self.n_instants - self.width + 1
        # For training and prediction
        self.batch_size = batch_size
        self.curr_bs = self.batch_size
        # Number of items that are read each time
        self.n_to_read = self.batch_size + self.n_x - 1
        self.X_shape = X_shape
        self.y_shape = y_shape
        self.target_ids = target_ids
        self.batch_number = 1
        self.go = False
        #self.__check('From constructor')

    def __check(self, t):
        print()
        print('\t', t)
        print('\t     batch_number {}/{}'.format(self.batch_number, len(self)))
        print('\t     batch_size', self.batch_size)
        print('\t     curr_bs', self.curr_bs)
        print('\t     n_to_read', self.n_to_read)
        print('\t     ts_pointer', self.ts_pointer)
        print('\t     width', self.width)
        print('\t     n_preds', self.n_preds)
        print('\t     n_instants', self.n_instants)
        print('\t     start, end', self.start_id, self.end_id)
        print()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.ceil(self.n_preds / self.batch_size))

    def __getitem__(self, batch_n):
        '''
        Generate one batch of data
        Caveat: For some reason, batch_n is the batch number when used as a generator, but it is random when used from tf.
        '''
        #print('\t... now working on bn =', self.batch_number, '    tsp =', self.ts_pointer, end='\r')
        # If the previous call (was from peek_and_restore) or (finished an epoch) then (initialize generator)
        if (self.batch_number == 2 and not self.go) or (self.batch_number == len(self) + 1):
            self.__init_generator()
        # If (we are on the last batch) then (recalculate curr_bs and n_to_read)
        if self.batch_number * self.batch_size > self.n_preds:
            #self.__check('From _getitem_ {}'.format(batch_n))
            self.__prepare_for_last_batch()
        x_extent = (self.ts_pointer, self.ts_pointer + self.n_to_read)
        # Read chunk, from map or flat
        X_ts = np.empty((self.curr_bs, *self.X_shape[1:]))
        y = np.empty((self.curr_bs, *self.y_shape[1:]))
        for idx_dset, ts_dataset in enumerate(self.time_series):
            X_ts[..., idx_dset], y[..., idx_dset] = self.__data_generation(ts_dataset, *x_extent, batch_n)
        X = {'time_series': X_ts}
        # Weather:
        if self.weather is not None:
            n_repeat = 4
            # Equivalent to int(ceil(a / b)) <=> - ( - a // b)
            x_extent = (self.ts_pointer // n_repeat,
                        -(-(self.ts_pointer + self.n_to_read) // n_repeat))
            X['weather'] = self.__data_generation(self.weather[0], *x_extent, batch_n, n_repeat=4)[0]
        # Time:
        if self.time_aware:
            base = self.ts_pointer + self.n_x - 1
            x_extent = (base, base + self.curr_bs)
            X['time'] = self.__get_input_times(*x_extent)
        # Update the ts_pointer and batch_number for the next batch
        self.ts_pointer += self.curr_bs
        self.batch_number += 1
        return X, y

    def __init_generator(self):
        self.go = True
        self.ts_pointer = self.start_id
        self.batch_number = 1
        # And prepare for the next epoch:
        self.curr_bs = self.batch_size
        self.n_to_read = self.batch_size + self.n_x - 1

    def __prepare_for_last_batch(self):
        self.curr_bs = self.n_preds - (self.batch_number - 1) * self.batch_size
        self.n_to_read = self.curr_bs + self.n_x - 1

    def __idx_to_datetime(self, idx, freq, year=2013):
        # Frequency expressed in minutes
        freq = int(freq.replace('min', ''))
        base = dt.datetime(year, 1, 1, 0, 0)
        return base + dt.timedelta(minutes=int(freq * idx))

    def __get_input_times(self, x_l, x_r, freq='15min', mapper=dict(zip(range(7), [0]*5 + [1]*2))):
        '''
        Implemented to include:
        - time of day in [-1, 1],
        - time of week in [-1, 1],
        - time of year in [-1, 1],
        - week or weekend in {0, 1},
        - holiday in {0, 1}.
        '''        
        times = pd.date_range(start=self.__idx_to_datetime(x_l, freq),
                              end=self.__idx_to_datetime(x_r-1, freq), freq=freq)
        # Convert to seconds since epoch
        times_s = times.astype(int) // 1e9
        day = 24 * 60 * 60
        week = 7 * day
        year = 365.2425 * day
        time_ret = np.stack([np.sin(times_s * 2 * np.pi / day),
                             np.cos(times_s * 2 * np.pi / day),
                             np.sin(times_s * 2 * np.pi / week),
                             np.cos(times_s * 2 * np.pi / week),
                             np.sin(times_s * 2 * np.pi / year),
                             np.cos(times_s * 2 * np.pi / year),
                             times.weekday.map(mapper),
                             times.normalize().isin(self.holidays.Date).astype(int)],
                            axis=1)
        return time_ret

    def __data_generation(self, dset_path, x_l, x_r, batch_n, n_repeat=1):
        '''Generates data containing the required samples'''
        # It is based on the current batch size, which depends on the batch id
        with tb.open_file(dset_path, mode='r') as h5_file:
            X_slc = h5_file.get_node(self.group)[x_l:x_r, :][:, self.target_ids]
            # For different time granularities
            X_slc = np.repeat(X_slc, n_repeat, axis=0)
            if n_repeat > 1:
                # This is to adjust the time conversion (e.g. 1h -> 4 * 15min):
                base = self.ts_pointer % n_repeat
                X = np.stack([X_slc[i:i + self.n_x] for i in range(base, base + self.curr_bs)])
                return (X,)
            # Reshape, copying slices of lenght = n_x
            X = np.stack([X_slc[i:i + self.n_x] for i in range(self.curr_bs)])
            y_l, y_r = x_l + self.n_x + self.shift - 1, x_r + self.n_y + self.shift - 1
            y_slc = h5_file.get_node(self.group)[y_l:y_r, :][:, self.target_ids]
            y = np.stack([y_slc[i:i + self.n_y] for i in range(self.curr_bs)])
        return X, y
