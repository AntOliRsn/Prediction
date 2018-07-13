import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras import backend as K
from keras.models import load_model
from models.base_NN import BaseModel


class FeedForward(BaseModel):
    def __init__(self, input_dim=96, output_dim=1, l_dims=[50], dropout_rates = [0], loss ='mean_squared_error', metrics=['mape','mae'], **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l_dims = l_dims
        self.dropout_rates = dropout_rates
        self.loss = loss
        self.metrics = metrics
        self.model = None

        self.build_model()

    def build_model(self, verbose=True):
        """

        :param verbose:
        :return:
        """

        x_input = Input(shape=(self.input_dim,), name='dense_input')
        x = x_input

        for idx, layer_dim in enumerate(self.l_dims):
            x = Dense(units=layer_dim, activation='relu', name="dense_hidden_{}".format(idx))(x)
            if self.dropout_rates[idx] !=0 :
                x = Dropout(self.dropout_rates[idx], name="dropout_hidden_{}".format(idx))(x)

        y_hat = Dense(units=self.output_dim, activation='linear', name='dense_output')(x)

        self.model = Model(inputs=x_input, outputs=y_hat)
        self.model.compile(optimizer='adam', loss=self.loss , metrics=self.metrics)

        # Store trainers
        self.store_to_save('model')

        if verbose:
            print("model: ")
            self.model.summary()

    def train(self, dataset_train, training_epochs=10, batch_size=20, callbacks = [], validation_data = None, verbose = True):
        """

        :param dataset_train:
        :param training_epochs:
        :param batch_size:
        :param callbacks:
        :param validation_data:
        :param verbose:
        :return:
        """

        self.training_epochs = training_epochs
        self.batch_size = batch_size

        model_hist = self.model.fit(dataset_train['x'], dataset_train['y'], batch_size=batch_size, epochs=training_epochs,
                             validation_data=validation_data,
                             callbacks=callbacks, verbose=True)

        return model_hist

    def analyze_history(self, dataset):

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        best_iter = np.argmin(self.history['val_loss'])
        min_val_loss = self.history['val_loss'][best_iter]

        summary_df = pd.DataFrame(columns=['name', 'layer_dims','dropout_rates','batchsize',
                                           'best_iter', 'train_mse',
                                           'train_mae', 'train_mape',
                                           'test_mse', 'test_mae',
                                           'test_mape'])

        summary = {'name': self.name,
                   'layer_dims': str(self.l_dims),
                   'dropout_rates': str(self.dropout_rates),
                   'batchsize': self.batch_size,
                   'best_iter': best_iter+1}

        path_best_model = os.path.join(self.out_dir, 'models', 'model-best.hdf5')

        if not os.path.exists(path_best_model):
            print('set model checkpoint as callback to get the best model')
            return
        else:
            best_model = load_model(path_best_model)

        y_hat_train = best_model.predict(dataset['train']['x'])
        y_train = dataset['train']['y']

        summary['train_mse'] = mean_squared_error(y_train, y_hat_train)
        summary['train_mae'] = mean_absolute_error(y_train, y_hat_train)
        summary['train_mape'] = mean_absolute_percentage_error(y_train, y_hat_train)

        y_hat_train = best_model.predict(dataset['test']['x'])
        y_train = dataset['test']['y']

        summary['test_mse'] = mean_squared_error(y_train, y_hat_train)
        summary['test_mae'] = mean_absolute_error(y_train, y_hat_train)
        summary['test_mape'] = mean_absolute_percentage_error(y_train, y_hat_train)

        summary_df = summary_df.append(summary, ignore_index=True)

        return summary_df, summary

    def main_train(self, dataset, training_epochs=100, batch_size=100, callbacks=[], verbose=True):
        super().main_train(dataset, training_epochs, batch_size, callbacks, verbose)

        summary_df, _ = self.analyze_history(dataset)
        if verbose:
            print(summary_df)

        summary_df.to_csv(os.path.join(self.out_dir, 'results', 'summary.csv'), sep=';')


