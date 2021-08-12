from keras import layers, models, optimizers, utils
from keras import Model as kModel


class Models():
    '''
    Class that contains the hardcoded models.
    Also allows to load a pre-trained model from an HDF5 file.
    '''
    
    def __init__(self, n_x, n_y,
                 map_shape=801, act_out='sigmoid',
                 loss='mse', metrics=['mae', 'mse'],
                 optimizer=optimizers.Adadelta, lr=1.0):
        self.n_x = n_x
        self.n_y = n_y
        self.map_shape = map_shape
        self.act_out = act_out
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr = lr
        self.models_d = {'ST_MDF': self.__ST_MDF,       'ST_MDF_w': self.__ST_MDF_w,
                         'ST_MDF_t': self.__ST_MDF_t,   'ST_MDF_star': self.__ST_MDF_star,
                         'fc': self.__fc, 'FC': self.__FC, 'Fc': self.__Fc,
                         'lstm': self.__lstm,  'biLstm': self.__biLstm,  'rnn': self.__rnn,
                         'persistence': self.__persistence, 'naive': self.__naive
                        }

    def get_models_list(self):
        return list(self.models_d.keys())
    
    def model(self, mod_name):
        if mod_name in self.models_d.keys():
            return self.models_d[mod_name]()
        print('Model not found!')
        return None

    def load_model(self, model_path, do_compile=False):
        model = models.load_model(model_path)
        if do_compile:
                self.__compile_model(model)
        return model

    def plot_model(self, model, fpath):
        utils.plot_model(model, to_file=fpath,
                         show_shapes=True, show_layer_names=False)
        print('\n\tModel graph saved to {}'.format(fpath))

    def __compile_model(self, model):
        opt = self.optimizer(lr=self.lr) if self.lr > 0 else self.optimizer()
        model.compile(optimizer=opt, loss=self.loss, metrics=self.metrics)

    def __ST_MDF(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = layers.convolutional_recurrent.ConvLSTM2D(filters=3, kernel_size=(8, 8),
                                                             return_sequences=False)(input_maps)
        x0 = layers.MaxPool2D((4, 4))(x0)
        x0 = layers.Flatten()(x0)

        x3 = layers.Dense(1 * 43 * 28 * 1)(x0)
        x3 = layers.Reshape((1, 43, 28, 1))(x3)
        x3 = layers.convolutional.Conv3DTranspose(filters=2, kernel_size=(self.n_y, 6, 6), strides=(2,2,2))(x3)
        model = kModel(inputs=[input_maps], outputs=x3, name='ST_MDF')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __ST_MDF_w(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = layers.convolutional_recurrent.ConvLSTM2D(filters=3, kernel_size=(8, 8),
                                                             return_sequences=False)(input_maps)
        x0 = layers.MaxPool2D((4, 4))(x0)
        x0 = layers.Flatten()(x0)

        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.LSTM(16)(weather)

        x3 = layers.Concatenate()([x0, x2])
        x3 = layers.Dense(1 * 43 * 28 * 1)(x3)
        x3 = layers.Reshape((1, 43, 28, 1))(x3)
        x3 = layers.convolutional.Conv3DTranspose(filters=2, kernel_size=(self.n_y, 6, 6), strides=(2,2,2))(x3)
        model = kModel(inputs=[input_maps, weather], outputs=x3, name='ST_MDF_w')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __ST_MDF_t(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = layers.convolutional_recurrent.ConvLSTM2D(filters=3, kernel_size=(8, 8),
                                                             return_sequences=False)(input_maps)
        x0 = layers.MaxPool2D((4, 4))(x0)
        x0 = layers.Flatten()(x0)

        features = layers.Input(shape=(8, ), name='time')
        x1 = layers.Dense(16)(features)

        x3 = layers.Concatenate()([x0, x1])
        x3 = layers.Dense(1 * 43 * 28 * 1)(x3)
        x3 = layers.Reshape((1, 43, 28, 1))(x3)
        x3 = layers.convolutional.Conv3DTranspose(filters=2, kernel_size=(self.n_y, 6, 6), strides=(2,2,2))(x3)
        model = kModel(inputs=[input_maps, features], outputs=x3, name='ST_MDF_t')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __ST_MDF_star(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = layers.convolutional_recurrent.ConvLSTM2D(filters=3, kernel_size=(8, 8),
                                                             return_sequences=False)(input_maps)
        x0 = layers.MaxPool2D((4, 4))(x0)
        x0 = layers.Flatten()(x0)

        features = layers.Input(shape=(8, ), name='time')
        x1 = layers.Dense(16)(features)

        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.LSTM(16)(weather)

        x3 = layers.Concatenate()([x0, x1, x2])
        x3 = layers.Dense(1 * 43 * 28 * 1)(x3)
        x3 = layers.Reshape((1, 43, 28, 1))(x3)
        x3 = layers.convolutional.Conv3DTranspose(filters=2, kernel_size=(self.n_y, 6, 6), strides=(2,2,2))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='ST_MDF_star')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __fc(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = layers.Flatten()(input_maps)
        
        features = layers.Input(shape=(8, ), name='time')

        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.Flatten()(weather)

        x3 = layers.Concatenate()([x0, features, x2])
        x3 = layers.Dense(32)(x3)
        x3 = layers.Dense(self.n_y * self.map_shape[0] * self.map_shape[1] * 2)(x3)
        x3 = layers.Reshape((self.n_y, self.map_shape[0], self.map_shape[1], 2))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='fc')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __Fc(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = layers.Flatten()(input_maps)

        x3 = layers.Dense(32)(x0)
        x3 = layers.Dense(self.n_y * self.map_shape[0] * self.map_shape[1] * 2)(x3)
        x3 = layers.Reshape((self.n_y, self.map_shape[0], self.map_shape[1], 2))(x3)
        model = kModel(inputs=[input_maps], outputs=x3, name='Fc')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __FC(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = layers.Flatten()(input_maps)

        features = layers.Input(shape=(8, ), name='time')

        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.Flatten()(weather)

        x3 = layers.Concatenate()([x0, features, x2])
        x3 = layers.Dense(32)(x3)
        x3 = layers.Dense(32)(x3)
        x3 = layers.Dense(self.n_y * self.map_shape[0] * self.map_shape[1] * 2)(x3)
        x3 = layers.Reshape((self.n_y, self.map_shape[0], self.map_shape[1], 2))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='FC')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __lstm(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = layers.Reshape((self.n_x, self.map_shape[0] * self.map_shape[1] * 2))(input_maps)
        x3 = layers.LSTM(16, return_sequences=False)(x0)

        features = layers.Input(shape=(8, ), name='time')
        x1 = layers.Dense(16)(features)
        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.LSTM(16)(weather)
        x2 = layers.Flatten()(x2)

        x3 = layers.Concatenate()([x3, x1, x2])
        x3 = layers.Dense(32)(x3)
        x3 = layers.Dense(self.n_y * self.map_shape[0] * self.map_shape[1] * 2)(x3)
        x3 = layers.Reshape((self.n_y, self.map_shape[0], self.map_shape[1], 2))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='lstm')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __biLstm(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = layers.Reshape((self.n_x, self.map_shape[0] * self.map_shape[1] * 2))(input_maps)
        x3 = layers.Bidirectional(layers.LSTM(16, return_sequences=False))(x0)

        features = layers.Input(shape=(8, ), name='time')
        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.Flatten()(weather)

        x3 = layers.Concatenate()([x3, features, x2])
        x3 = layers.Dense(self.n_y * self.map_shape[0] * self.map_shape[1] * 2)(x3)
        x3 = layers.Reshape((self.n_y, self.map_shape[0], self.map_shape[1], 2))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='biLstm')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __rnn(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = layers.Reshape((self.n_x, self.map_shape[0] * self.map_shape[1] * 2))(input_maps)
        x3 = layers.SimpleRNN(16, return_sequences=False)(x0)

        features = layers.Input(shape=(8, ), name='time')
        x1 = layers.Dense(16)(features)
        weather = layers.Input(shape=(self.n_x, 8), name='weather')
        x2 = layers.LSTM(16)(weather)
        x2 = layers.Flatten()(x2)

        x3 = layers.Concatenate()([x3, x1, x2])
        x3 = layers.Dense(self.n_y * self.map_shape[0] * self.map_shape[1] * 2)(x3)
        x3 = layers.Reshape((self.n_y, self.map_shape[0], self.map_shape[1], 2))(x3)
        model = kModel(inputs=[input_maps, features, weather], outputs=x3, name='rnn')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __persistence(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = input_maps[:, -1:]
        x0 = layers.Concatenate(axis=1)([x0 for i in range(self.n_y)])
        model = kModel(inputs=[input_maps], outputs=x0, name='persistence')
        # Compile model and return
        self.__compile_model(model)
        return model

    def __naive(self):
        input_maps = layers.Input(shape=(self.n_x, *self.map_shape, 2), name='time_series')
        x0 = layers.Add()([input_maps[:, i] for i in range(self.n_x)]) / self.n_x
        x0 = layers.Reshape((1, *self.map_shape, 2))(x0)
        x0 = layers.Concatenate(axis=1)([x0 for i in range(self.n_y)])
        model = kModel(inputs=[input_maps], outputs=x0, name='naive')
        # Compile model and return
        self.__compile_model(model)
        return model

    # n_x
    def get_n_x(self):
        return self.n_x 
    def set_n_x(self, x):
        self.n_x = int(x)

    # Number of forecast horizons
    def get_n_y(self):
        return self.n_y
    def set_n_y(self, x):
        self.n_y = int(x)

    # Map shape
    def get_map_shape(self):
        return self.map_shape
    def set_map_shape(self, x):
        self.map_shape = x

    # Time aware
    def get_time_aware(self):
        return self.time_aware
    def set_time_aware(self, x):
        self.time_aware = x

    # Activation function for the last layer
    def get_act_out(self):
        return self.act_out
    def set_act_out(self, x):
        self.act_out = x

    # Optimizer
    def get_optimizer(self):
        return self.optimizer
    def set_optimizer(self, x):
        self.optimizer = eval('optimizers.' + x)

    # Loss
    def get_loss(self):
        return self.loss
    def set_loss(self, x):
        self.loss = x

    # Metrics
    def get_metrics(self):
        return self.metrics
    def set_metrics(self, x):
        self.metrics = eval(x)

    # Learning rate
    def get_lr(self):
        return self.lr
    def set_lr(self, x):
        self.lr = float(x)
