import random

from keras import optimizers
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Activation, Dense, Flatten, Dropout, Conv2D, MaxPool2D
from keras.models import Sequential
from typing import List, Dict, Union

import helpers


class Network:
    (__x_train, __y_train), (__x_val, __y_val) = (None, None), (None, None)

    @staticmethod
    def prepare_data(dataset_name='colorflow'):
        if Network.__x_val is None or Network.__y_val is None:
            (Network.__x_train, Network.__y_train), (Network.__x_val, Network.__y_val) =\
                helpers.prepare_data(dataset_name=dataset_name)
        else:
            (Network.__x_train, Network.__y_train), (_, __) = \
                helpers.prepare_data(dataset_name=dataset_name, first_time=False)

    def __init__(self, architecture, opt='adam', lr=None, activation='relu', callbacks=None):
        # type: (Network, List, Union[str, optimizers.Optimizer], float, str, List[Callback]) -> None
        assert hasattr(architecture, "__getitem__")

        # Check that architecture vector is first tuples (for convolutions)/MaxOuts,
        # and then integers for the dense layers or 'drop0.2' for a dropout layer
        dense_started = False
        for i in architecture:
            if hasattr(i, "__getitem__") and not isinstance(i, str):
                if dense_started:
                    print(architecture)
                    raise TypeError(
                        'Architecture is not correctly formatted.\n'
                        'All Convolution layers should appear before Dense/Dropout layers.')

                if not (hasattr(i[0], "__getitem__") and isinstance(i[1], int)):
                    raise TypeError(
                        'Architecture is not correctly formatted.\n'
                        'The part of architecture which cause the problem is ' + str(i))

            elif isinstance(i, int):
                dense_started = True

            elif isinstance(i, str) and i.lower() in ['max', 'maxout', 'maxpool']:
                if dense_started:
                    print(architecture)
                    raise TypeError(
                        'Architecture is not correctly formatted.\n'
                        'All MaxPool layers should appear before Dense/Dropout layers.\n'
                        'The part of architecture which cause the problem is ' + str(i)
                    )

            elif isinstance(i, str) and i.lower().startswith(('drop', 'dropout')):
                dense_started = True
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

            else:
                raise TypeError(
                    'Architecture is not correctly formatted.\n'
                    'Arguments should either be iterable, ints, \'max\', or \'drop0.x\','
                    'where 0.x can be any float fraction.\n'
                    'The part of architecture which cause the problem is ' + str(i)
                )

        self.callbacks = callbacks  # type: List[Callback]
        if callbacks is None:
            self.callbacks = [EarlyStopping(patience=5)]
        self.model = Sequential()  # type: Sequential
        self.arch = architecture  # type: List
        self.act = activation  # type: str
        self.model_created = False  # type: bool
        if isinstance(opt, optimizers.Optimizer):
            self.opt = opt
        else:
            self.opt = self.__optimizer(opt, lr=lr)  # type: optimizers.Optimizer
        self.__create_model()
        self.__score = 0.  # type: float

    @staticmethod
    def __optimizer(opt_name, lr=None):
        # type: (str, float) -> optimizers.Optimizer

        opt_name = opt_name.lower()
        if lr is None:
            if opt_name == 'adam':
                return optimizers.Adam()
            elif opt_name == 'sgd':
                return optimizers.SGD()
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
                return optimizers.SGD(lr=lr)
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

    def fit(self, epochs=1, batch_size=100, shuffle='batch', verbose=0):
        self.__score = 0.  # Resets score, so it will not collide w/ scoring it again (but w/ different weights).
        if Network.__x_train is None or Network.__x_val is None or Network.__y_train is None or Network.__y_val is None:
            Network.prepare_data()

        self.model.fit(
            x=Network.__x_train, y=Network.__y_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
            callbacks=self.callbacks, validation_data=(Network.__x_val, Network.__y_val), verbose=verbose
        )

    def score(self, f=None):
        # type: (Network, function) -> float
        import inspect

        if f is None:
            f = helpers.multi_roc_score

        args = inspect.getargspec(f)[0]
        if not ('y_true' in args and 'y_score' in args):
            raise AttributeError('Given function f, should have parameters y_true and y_score.')

        if self.__score == 0.0:
            self.__score = f(y_true=self.__y_val, y_score=self.model.predict(self.__x_val))

        return self.__score

    def __create_model(self):
        if self.model_created:
            raise Exception('Cannot recreate a new model in the same instance.')

        self.model_created = True

        if Network.__x_train is None or Network.__x_val is None or Network.__y_train is None or Network.__y_val is None:
            raise Exception('Please prepare data before. creating a new model.')

        assert hasattr(self.arch, "__getitem__")
        assert isinstance(self.model, Sequential)
        self.model.add(Activation(activation='linear', input_shape=list(Network.__x_train[0].shape)))

        last_max_pool = True
        last_dropout = True
        for i in self.arch:

            if hasattr(i, '__getitem__') and not isinstance(i, str):
                last_max_pool = False
                self.model.add(
                    Conv2D(filters=i[1], kernel_size=i[0], activation=self.act, kernel_initializer='he_uniform',
                           padding='same')
                )

            elif isinstance(i, int):
                if len(self.model.output_shape) > 2:
                    self.model.add(Flatten())
                last_dropout = False
                self.model.add(Dense(units=i, activation=self.act))

            elif isinstance(i, str) and i.lower() in ['m', 'max', 'maxout', 'maxpool']:
                if not last_max_pool:
                    if self.model.output_shape[1] > 2:  # asserts that there's not too many maxpools
                        self.model.add(MaxPool2D())
                        last_max_pool = True
                    else:
                        self.arch.remove(i)
                else:
                    self.arch.remove(i)

            elif isinstance(i, str) and i.lower().startswith(('drop', 'dropout')):
                if not last_dropout:
                    if len(self.model.output_shape) > 2:
                        self.model.add(Flatten())
                    if i.lower().startswith('dropout'):
                        self.model.add(Dropout(rate=float(i[7:])))
                    else:
                        self.model.add(Dropout(rate=float(i[4:])))

            else:
                raise TypeError('Architecture is not correctly formatted.')

        if len(self.model.output_shape) > 2:
            self.model.add(Flatten())
        self.model.add(Dense(units=len(Network.__y_train[0]), activation='tanh'))
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def add_layer(self, params):
        # type: (Network, Dict[str, List]) -> Network
        layer = random.randint(0, len(self.arch))

        conv_max = True
        dense_drop = True

        if layer != 0 and not \
                (hasattr(self.arch[layer - 1], "__getitem__") and not isinstance(self.arch[layer - 1], str)) or \
                (isinstance(self.arch[layer - 1], str)
                 and self.arch[layer - 1].lower() in ['max', 'maxout', 'maxpool']):
            conv_max = False

        elif layer <= len(self.arch) and not \
                (isinstance(self.arch[layer], int) or
                 isinstance(self.arch[layer], str) and
                 self.arch[layer].lower().startswith(('drop', 'dropout'))):
            dense_drop = False

        possible_layers = {}
        if conv_max:
            possible_layers['max'] = 'max'
            possible_layers['conv'] = \
                (random.choice(params['kernel_size']), random.choice(params['conv_filters']))
        if dense_drop:
            possible_layers['dense'] = random.choice(params['dense_size'])
            possible_layers['drop'] = 'drop' + str(random.choice(params['dropout']))
        added_layer = random.choice(possible_layers.values())
        return Network(architecture=self.arch[:layer] + [added_layer] + self.arch[layer:], opt=self.opt,
                       activation=self.act, callbacks=self.callbacks)

    def remove_layer(self, params):
        # type: (Network, Dict[str, List]) -> Network
        if len(self.arch) > 1:
            return self.add_layer(params)
        layer = random.randint(0, len(self.arch) - 1)
        return Network(architecture=self.arch[:layer] + self.arch[layer + 1:], opt=self.opt, activation=self.act,
                       callbacks=self.callbacks)

    def change_opt(self, params):
        # type: (Network, Dict[str, List]) -> Network
        return Network(architecture=self.arch, opt=random.choice(params['optimizer']),
                       lr=random.choice(params['optimizer_lr']), activation=self.act, callbacks=self.callbacks)

    def change_activation(self, params):
        # type: (Network, Dict[str, List]) -> Network
        return Network(architecture=self.arch, opt=self.opt, activation=random.choice(params['activation']),
                       callbacks=self.callbacks)
