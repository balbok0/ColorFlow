import random

import numpy as np
from keras import optimizers
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from keras.layers import Activation, Dense, Flatten, Dropout, Conv2D, MaxPool2D, Layer
from keras.models import Sequential, clone_model, Model
from typing import List, Dict, Union

import helpers
import net_mutator_vars as const
from local_vars import debug


class Network:
    (__x_train, __y_train), (__x_val, __y_val) = (None, None), (None, None)

    default_callbacks = [EarlyStopping(patience=5)]  # type: List[Callback]

    @staticmethod
    def prepare_data(dataset_name='colorflow'):
        if Network.__x_val is None or Network.__y_val is None:
            (Network.__x_train, Network.__y_train), (Network.__x_val, Network.__y_val) = \
                helpers.prepare_data(dataset_name=dataset_name)
        else:
            (Network.__x_train, Network.__y_train), (_, __) = \
                helpers.prepare_data(dataset_name=dataset_name, first_time=False)

    def __init__(self, architecture, copy_model=None, opt='adam', lr=None, activation='relu', callbacks=None):
        # type: (Network, List, Model, Union[str, optimizers.Optimizer], float, str, List[Callback]) -> None
        assert hasattr(architecture, "__getitem__")
        # Check that architecture vector is first tuples (for convolutions)/MaxOuts,
        # and then integers for the dense layers or 'drop0.2' for a dropout layer
        dense_started = False
        drop_prev = True
        max_prev = True
        idx_to_remove = []

        for j in range(len(architecture)):
            i = architecture[j]
            if hasattr(i, "__getitem__") and not isinstance(i, str):
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

            elif isinstance(i, int):
                dense_started = True
                drop_prev = False

            elif isinstance(i, str) and i.lower() in ['max', 'maxout', 'maxpool']:
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

            elif isinstance(i, str) and i.lower().startswith(('drop', 'dropout')):
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
        self.model_created = False  # type: bool
        if isinstance(opt, optimizers.Optimizer):
            self.opt = opt  # type: optimizers.Optimizer
        else:
            self.opt = self.__optimizer(opt, lr=lr)  # type: optimizers.Optimizer
        self.__score = 0.  # type: float
        self.__prev_score = 0.  # type: float
        if copy_model is None:
            self.model = Sequential()  # type: Sequential
            self.__create_model()
        else:
            assert helpers.assert_model_arch_match(copy_model, self.arch)
            self.model = clone_model(copy_model)
            self.model.set_weights(copy_model.get_weights())
            if self.act != activation:
                for l in self.model.layers[1:-1]:  # type: Layer
                    if not isinstance(l, (Activation, MaxPool2D, Flatten, Dropout)):
                        l.activation = helpers.activations_function_calls[activation]
            self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])

    @staticmethod
    def __optimizer(opt_name, lr=None):
        # type: (str, float) -> optimizers.Optimizer

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
        self.__prev_score = self.__score
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
        import inspect

        if f is None:
            f = helpers.multi_roc_score

        args = inspect.getargspec(f)[0]
        if not ('y_true' in args and 'y_score' in args):
            raise AttributeError('Given function f, should have parameters y_true and y_score.')

        if self.__score == 0.0:
            self.__score = f(y_true=self.__y_val, y_score=self.model.predict(self.__x_val))

        return self.__score

    def save(self, file_path, overwrite=True):
        # type: (Network, str, bool) -> None
        self.model.save(filepath=file_path, overwrite=overwrite)

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

    def add_layer(self, params):
        # type: (Network, Dict[str, List]) -> Network
        layer_idx = random.randint(0, len(self.arch))

        possible_layers = {}

        if helpers.get_number_of_weights(self.model) > const.max_n_weights or len(self.arch) + 1 > const.max_depth:
            self.remove_layer(params)

        if layer_idx == 0:
            possible_layers['conv'] = random.choice(params['kernel_size']), random.choice(params['conv_filters'])
        elif layer_idx == len(self.arch):
            possible_layers['dense'] = random.choice(params['dense_size'])
        else:
            prev_layer = self.arch[layer_idx - 1]
            next_layer = self.arch[layer_idx]
            if hasattr(prev_layer, '__getitem__') and \
                    (not isinstance(prev_layer, str) or (isinstance(prev_layer, str) and prev_layer.startswith('max'))):
                possible_layers['conv'] = (random.choice(params['kernel_size']), random.choice(params['conv_filters']))
                if not (isinstance(prev_layer, str) or (isinstance(next_layer, str) and next_layer.startswith('max'))):
                    possible_layers['max'] = 'max'

            check_if_flat = lambda x: isinstance(x, int) or (isinstance(x, str) and x.startswith('drop'))

            if check_if_flat(next_layer):
                possible_layers['dense'] = random.choice(params['dense_size'])
                if isinstance(next_layer, int) and check_if_flat(prev_layer) \
                        and not (isinstance(prev_layer, str) and prev_layer.startswith('drop')):
                    possible_layers['drop'] = 'drop' + str(random.choice(params['dropout']))

        layer_name = random.choice(possible_layers.values())

        new_arch = self.arch[:layer_idx] + [layer_name] + self.arch[layer_idx:]

        layer_idx += 1  # difference between self.arch and actual architecture. - First activation layer.
        if isinstance(layer_name, int) or (isinstance(layer_name, str) and layer_name.startswith('drop')):
            layer_idx += 1  # difference between self.arch and actual architecture. - Flatten layer.

        return Network(
            architecture=new_arch,
            copy_model=helpers.insert_layer(self.model, helpers.arch_to_layer(layer_name, self.act), layer_idx),
            opt=self.opt,
            activation=self.act,
            callbacks=self.callbacks
        )

    def remove_layer(self, params):
        # type: (Network, Dict[str, List]) -> Network
        if len(self.arch) <= 2:
            return self.add_layer(params)
        layer_idx = random.randint(1, len(self.arch) - 2)  # so that, Conv is always first, and Dense is always last.
        layer_name = self.arch[layer_idx]
        new_arch = self.arch[:layer_idx] + self.arch[layer_idx + 1:]

        layer_idx += 1  # difference between self.arch and actual architecture. - First activation layer.
        if isinstance(layer_name, int) or (isinstance(layer_name, str) and layer_name.startswith('drop')):
            layer_idx += 1  # difference between self.arch and actual architecture. - Flatten layer.

        return Network(
            architecture=new_arch,
            copy_model=helpers.remove_layer(self.model, layer_idx),
            opt=self.opt,
            activation=self.act,
            callbacks=self.callbacks
        )

    def change_opt(self, params):
        # type: (Network, Dict[str, List]) -> Network
        return Network(
            architecture=self.arch,
            copy_model=self.model,
            opt=random.choice(params['optimizer']),
            lr=random.choice(params['optimizer_lr']),
            activation=self.act,
            callbacks=self.callbacks
        )

    def change_activation(self, params):
        # type: (Network, Dict[str, List]) -> Network
        return Network(
            architecture=self.arch,
            copy_model=self.model,
            opt=self.opt,
            activation=random.choice(params['activation']),
            callbacks=self.callbacks
        )

    def change_lr_schedule(self, params):
        # type: (Network, Dict[str, List]) -> Network
        if random.choice(params['learning_decay_type']) == 'linear':
            def schedule(x):
                return self.opt.get_config()['lr'] - float(x * random.choice(params['learning_decay_rate']))
        else:
            def schedule(x):
                return self.opt.get_config()['lr'] - float(np.exp(-x * random.choice(params['learning_decay_rate'])))

        return Network(
            architecture=self.arch,
            copy_model=self.model,
            opt=self.opt,
            activation=self.act,
            callbacks=Network.default_callbacks + [LearningRateScheduler(schedule)]
        )

    def get_config(self):
        # type: () -> Dict[str, ]
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
