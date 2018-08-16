import copy

import numpy as np
from keras import optimizers
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Activation, Dense, Flatten, Dropout, Conv2D, MaxPool2D, Layer
from keras.models import Sequential, clone_model, Model
from typing import List, Dict, Union

import helpers
from program_variables.program_params import debug


class Network:
    (__x_train, __y_train), (__x_val, __y_val) = (None, None), (None, None)

    default_callbacks = [EarlyStopping(patience=5)]  # type: List[Callback]

    @staticmethod
    def prepare_data(dataset_name='colorflow'):
        """
        Prepares data with a given name, loads it to memory. and points variables to it. Possible arguments are:
            - colorflow
            - cifar
            - mnist
            - testing.

        :param dataset_name: Name of dataset to be used.
        """
        if Network.__x_val is None or Network.__y_val is None:
            (Network.__x_train, Network.__y_train), (Network.__x_val, Network.__y_val) = \
                helpers.prepare_data(dataset_name=dataset_name)
        else:
            (Network.__x_train, Network.__y_train), (_, __) = \
                helpers.prepare_data(dataset_name=dataset_name, first_time=False)

    def __init__(self, architecture, copy_model=None, opt='adam', lr=None, activation='relu', callbacks=None):
        # type: (Network, List, Model, Union[str, optimizers.Optimizer], float, str, List[Callback]) -> None
        """
        Creates a new instance of Network.

        :param architecture: A list description of the network.
        :param copy_model: A keras Model to make a copy of.
        :param opt: Optimizer used for given network/
        :param lr: Learning rate to be used with an optimizer.
        :param activation: Activation function to be used in a network.
        :param callbacks: Callbacks to be used while training a network.
        """
        assert hasattr(architecture, "__getitem__")
        # Check that architecture vector is first tuples (for convolutions)/MaxOuts,
        # and then integers for the dense layers or 'drop0.2' for a dropout layer
        dense_started = False
        drop_prev = True
        max_prev = True
        idx_to_remove = []

        for j in range(len(architecture)):
            i = architecture[j]
            i_type = helpers.arch_type(i)
            if i_type == 'conv':
                max_prev = False
                if dense_started:
                    if debug:
                        print(architecture)
                    raise TypeError(
                        'Architecture is not correctly formatted.\n'
                        'All Convolution layers should appear before Dense/Dropout layers.')

                if not (hasattr(i[0], "__getitem__") and isinstance(i[1], int)):
                    raise TypeError(
                        'Architecture is not correctly formatted.\n'
                        'The part of architecture which cause the problem is ' + str(i))

            elif i_type == 'dense':
                dense_started = True
                drop_prev = False

            elif i_type == 'max':
                if dense_started:
                    if debug:
                        print(architecture)
                    raise TypeError(
                        'Architecture is not correctly formatted.\n'
                        'All MaxPool layers should appear before Dense/Dropout layers.\n'
                        'The part of architecture which cause the problem is ' + str(i)
                    )
                if max_prev:
                    idx_to_remove = [j] + idx_to_remove
                max_prev = True

            elif i_type == 'drop':
                dense_started = True
                if drop_prev or j == len(architecture) - 1:
                    idx_to_remove = [j] + idx_to_remove
                else:
                    try:
                        if i.lower().startswith('dropout'):
                            val = float(i[7:])
                        else:
                            val = float(i[4:])
                        if val >= 1. or val <= 0.:
                            raise AttributeError(
                                'Architecute is not correctly formatted.\n'
                                'Dropout value should be in range (0.0, 1.0).\n'
                                'The part of architecture which cause the problem is ' + str(i)
                            )
                    except ValueError:
                        raise ValueError(
                            'Architecute is not correctly formatted.\n'
                            'Arguments for dropout layer should be in form of \'drop\' or \'dropout\' '
                            'followed by a float.\n'
                            'In example: dropout.2, drop0.5 are valid inputs.\n'
                            'The part of architecture which cause the problem is ' + str(i)
                        )
                drop_prev = True

            else:
                raise TypeError(
                    'Architecture is not correctly formatted.\n'
                    'Arguments should either be iterable, ints, \'max\', or \'drop0.x\','
                    'where 0.x can be any float fraction.\n'
                    'The part of architecture which cause the problem is ' + str(i)
                )

        for idx in idx_to_remove:
            architecture.pop(idx)

        self.callbacks = callbacks  # type: List[Callback]
        if callbacks is None:
            self.callbacks = Network.default_callbacks
        self.arch = architecture  # type: List
        self.act = activation  # type: str
        self.__model_created = False  # type: bool
        if isinstance(opt, optimizers.Optimizer):
            self.opt = opt  # type: optimizers.Optimizer
        else:
            self.opt = self.__optimizer(opt, lr=lr)  # type: optimizers.Optimizer
        self.__score = 0.  # type: float
        self.__prev_score = 0.  # type: float
        self.__prev_weights = None  # type: np.ndarray
        if copy_model is None:
            self.model = Sequential()  # type: Sequential
            self.__create_model()
        else:
            self.model = clone_model(copy_model)
            self.model.set_weights(copy_model.get_weights())
            if self.act != activation:
                for l in self.model.layers[1:-1]:  # type: Layer
                    if not isinstance(l, (Activation, MaxPool2D, Flatten, Dropout)):
                        l.activation = helpers.activations_function_calls[activation]
            self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])
            assert helpers.assert_model_arch_match(copy_model, self.arch)

    @staticmethod
    def __optimizer(opt_name, lr=None):
        # type: (str, float) -> optimizers.Optimizer
        """
        Given a name and learning rate returns a keras optimizer based on it.

        :param opt_name: Name of optimizer to use.
                Legal arguments are:\n
                #. adam
                #. nadam
                #. rmsprop
                #. adamax
                #. adagrad
                #. adadelta
        :param lr: Learning rate of an optimizer.
        :return: A new optimizer based on given name and learning rate.
        """

        opt_name = opt_name.lower()
        if lr is None:
            if opt_name == 'adam':
                return optimizers.Adam()
            elif opt_name == 'sgd':
                return optimizers.SGD(nesterov=True)
            elif opt_name == 'nadam':
                return optimizers.Nadam()
            elif opt_name == 'rmsprop':
                return optimizers.RMSprop()
            elif opt_name == 'adamax':
                return optimizers.Adamax()
            elif opt_name == 'adagrad':
                return optimizers.Adagrad()
            elif opt_name == 'adadelta':
                return optimizers.Adadelta()

        else:
            if opt_name == 'adam':
                return optimizers.Adam(lr=lr)
            elif opt_name == 'sgd':
                return optimizers.SGD(lr=lr, nesterov=True)
            elif opt_name == 'nadam':
                return optimizers.Nadam(lr=lr)
            elif opt_name == 'rmsprop':
                return optimizers.RMSprop(lr=lr)
            elif opt_name == 'adamax':
                return optimizers.Adamax(lr=lr)
            elif opt_name == 'adagrad':
                return optimizers.Adagrad(lr=lr)
            elif opt_name == 'adadelta':
                return optimizers.Adadelta(lr=lr)
        raise AttributeError('Invalid name of optimizer given.')

    def fit(self, epochs=20, batch_size=100, shuffle='batch', verbose=0):
        # type: (Network, int, int, str, int) -> None
        """
        Trains a network on a training set.
        For paramaters descriptions look at documentation for keras.models.Model.fit function.
        """
        self.__prev_score = self.__score
        self.__prev_weights = copy.deepcopy(self.model.get_weights())
        self.__score = 0.  # Resets score, so it will not collide w/ scoring it again (but w/ different weights).
        if Network.__x_train is None or Network.__x_val is None or Network.__y_train is None or Network.__y_val is None:
            Network.prepare_data()
        if debug:
            print(self.get_config())
        self.model.fit(
            x=Network.__x_train, y=Network.__y_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
            callbacks=self.callbacks, validation_data=(Network.__x_val, Network.__y_val), verbose=verbose
        )

    def score(self, f=None):
        # type: (Network, function) -> float
        """
        Scores a network on a given function/metric.

        :param f: Function to score a function on. If not given a Area Under ROC Curve is used as a metric.
        :return: Returns a score of this network on given function/metric.
        """
        import inspect

        if f is None:
            f = helpers.multi_roc_score

        args = inspect.getargspec(f)[0]
        if not ('y_true' in args and 'y_score' in args):
            raise AttributeError('Given function f, should have parameters y_true and y_score.')

        if self.__score == 0.0:
            self.__score = f(y_true=self.__y_val, y_score=self.model.predict(self.__x_val))

        if self.__score < self.__prev_score:
            self.__score = self.__prev_score
            self.__prev_score = 0.
            self.model.set_weights(self.__prev_weights)
            self.__prev_weights = None

        return self.__score

    def save(self, file_path, overwrite=True):
        # type: (Network, str, bool) -> None
        """
        Given path, saves a network.

        :param file_path: A path to which network should be saved.
        :param overwrite: If such file already exists, whether it should be overwritten or not.
        """
        self.model.save(filepath=file_path, overwrite=overwrite)

    def __create_model(self):
        """
        With already set architecture, translates it into actual keras model.
        Also compiles it, so that an actual model is ready to use.
        """
        if self.__model_created:
            raise Exception('Cannot recreate a new model in the same instance.')

        self.__model_created = True

        if Network.__x_train is None or Network.__x_val is None or Network.__y_train is None or Network.__y_val is None:
            raise Exception('Please prepare data before. creating a new model.')

        assert hasattr(self.arch, "__getitem__")
        assert isinstance(self.model, Sequential)
        self.model.add(Activation(activation='linear', input_shape=list(Network.__x_train[0].shape)))

        last_max_pool = True
        last_dropout = True
        for i in self.arch:
            new_layer = helpers.arch_to_layer(i, self.act)

            if isinstance(new_layer, Conv2D):
                last_max_pool = False
                self.model.add(new_layer)

            elif isinstance(new_layer, Dense):
                if len(self.model.output_shape) > 2:
                    self.model.add(Flatten())
                last_dropout = False
                self.model.add(new_layer)

            elif isinstance(new_layer, MaxPool2D):
                if not last_max_pool:
                    if self.model.output_shape[1] > 2:  # asserts that there's not too many maxpools
                        self.model.add(new_layer)
                        last_max_pool = True
                    else:
                        self.arch.remove(i)
                else:
                    self.arch.remove(i)

            elif isinstance(new_layer, Dropout):
                if not last_dropout:
                    if len(self.model.output_shape) > 2:
                        self.model.add(Flatten())
                    if i.lower().startswith('dropout'):
                        self.model.add(Dropout(rate=float(i[7:])))
                    else:
                        self.model.add(Dropout(rate=float(i[4:])))
                    last_dropout = True
                else:
                    self.arch.remove(i)

            else:
                raise TypeError('Architecture is not correctly formatted.')

        if len(self.model.output_shape) > 2:
            self.model.add(Flatten())
        self.model.add(Dense(units=len(Network.__y_train[0]), activation='sigmoid'))
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def get_config(self):
        # type: () -> Dict[str, ]
        """
        :return: A dictionary, which specifies configuration of this network. It contains:\n
            #. architecture - a list of layers in a network.
            #. optimizer - on which this network was trained.
            #. activation - activation function on all of layers of this network.
            #. score - score of this network, if function score was called beforehand. 0 otherwise.
            #. callbacks - list of callbacks used while training this network.
        """
        opt_name = str(self.opt.__class__)
        opt_name = opt_name[opt_name.index(".") + 1:]
        opt_name = opt_name[:opt_name.index("\'")]
        opt_name = opt_name[opt_name.index(".") + 1:]
        return {
            'architecture': self.arch,
            'optimizer': '{opt} with learning rate: {lr}'.format(
                opt=opt_name,
                lr="{:.2g}".format(self.opt.get_config()['lr'])
            ),
            'activation': self.act,
            'score': self.__score,
            'callbacks': self.callbacks
        }
