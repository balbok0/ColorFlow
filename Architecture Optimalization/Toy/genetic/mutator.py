import collections
import os
import random

import numpy as np
from typing import List, Dict

import log_save
from network import Network

'''
Actual Mutator class is in the bottom of this file!
It wraps __Mutator class!!!
'''


class __Mutator(object):

    def __init__(self, params=None):
        # type: (__Mutator, Dict[str, List]) -> None

        if params is None:
            params = {
                'kernel_size': [(3, 3), (5, 5), (7, 7)],
                'conv_filters': [8, 16],
                'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
                'dense_size': [32, 64, 128, 256],
                'optimizer': ['adam', 'sgd', 'nadam'],
                'optimizer_lr': [None, .0001, .0003, .001, .003, .01],
                'learning_decay_type': ['linear', 'exp'],
                'learning_decay_rate': [0.7, 0.8, 0.9],
                'activation': ['relu', 'sigmoid', 'tanh']
            }

        else:
            for i in ['kernel_size', 'conv_filers', 'dropout', 'dense_size', 'optimizer', 'optimizer_lr', 'activation']:
                assert i in list(params.keys()), "There should be %s in params keys." % i
                assert isinstance(params[i], list), "Key %s should be pointing to the list." % i

        self.params = params  # type: Dict[str, List]
        self.networks = []  # type: List[Network]

    def get_best_architecture(
            self, population_size=10, generations=20, saving_dir=None,
            epochs=2, batch_size=100, shuffle='batch', verbose=0, dataset='colorflow'
    ):
        # type: (__Mutator, int, int, str, int, int, str, int, str) -> str
        Network.prepare_data(dataset)

        if saving_dir is None:
            saving_dir = 'genetic_models/'

        if not saving_dir.endswith('/'):
            saving_dir += '/'

        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)

        assert population_size > 0
        assert generations > 0
        assert os.path.exists(saving_dir)

        for i in range(population_size):
            self.networks.append(self.__create_random_model())

        scores = []  # Needed to suppress warnings.

        log_save.print_message('')
        log_save.print_message('Started a new job.')

        for i in range(generations):

            Network.prepare_data(dataset)

            log_save.print_message('Starting training for generation %d' % (i + 1))

            for net in self.networks:
                net.fit(epochs=epochs, batch_size=batch_size, shuffle=shuffle, verbose=verbose)

            log_save.print_message('Finished training for generation %d' % (i + 1))

            tmp_scores = {}  # type: Dict[Network, float]
            best_net = None  # type: Network

            for net in self.networks:
                tmp_scores[net] = net.score()
                if best_net is None or tmp_scores[net] > tmp_scores[best_net]:
                    best_net = net

            scores = collections.OrderedDict(sorted(tmp_scores.items(), key=lambda t: t[1]))

            printing = {
                0: 'Generation {gen}. Best network scored: {score}'.format(gen=i+1, score=scores[best_net]),
                1: 'Generation {gen}. Best network, with architecture: {arch}, optimizer {opt}, and activation '
                   'function {act}. It\'s score is {score}.'.format(
                    gen=i + 1, arch=best_net.arch, opt=best_net.opt, act=best_net.act, score=scores[best_net]
                ),
                2: 'Generation {gen}. Best network, with architecture: {arch}, optimizer {opt}, and activation '
                   'function {act}. It\'s score is {score}.'.format(
                    gen=i + 1, arch=best_net.arch, opt=best_net.opt, act=best_net.act, score=scores[best_net]
                )
            }
            log_save.print_message(printing[verbose])

            # Save the best net of current generation.
            best_net.save(file_path='{dir}net_{num:03d}.h5'.format(dir=saving_dir, num=i+1))

            if not i + 1 == generations:

                # Remove bottom half of population.
                for _ in range(int(population_size/2)):
                    scores.popitem(last=False)

                self.networks = scores.keys()
                new_nets = []

                for net in self.networks:
                    new_nets.append(self.__mutate(net))

                for net in new_nets:
                    self.networks.append(net)

        best = scores.popitem()
        best_score = best[1]
        best_net = best[0]

        return ('Best network, with architecture: {arch}, optimizer {opt}, and activation function {act}.'
                'It\'s score is {score}.'.format(
                    arch=best_net.arch, opt=best_net.opt, act=best_net.act, score=best_score
                    )
                )

    def __create_random_model(self):
        # type: (__Mutator) -> Network
        architecture = [(random.choice(self.params['kernel_size']),
                         random.choice(self.params['conv_filters']))]

        # Two variables for probability of:
        #   r_min - adding any new layer,
        #   r_mid - distinguishing between two types of layers for each part of arch.
        r_min = .33
        r_mid = .66

        # Convolution/Maxout part of architecture
        r = random.random()
        while r > r_min:
            r = random.random()
            if r > r_mid:
                architecture.append('max')
            else:
                architecture.append(
                    (random.choice(self.params.get('kernel_size')), random.choice(self.params.get('conv_filters'))))
            r = random.random()

        # Dense/Dropout part of architecture
        architecture.append(random.choice(self.params.get('dense_size')))
        r = random.random()
        while r > r_min:
            r = random.random()
            if r > r_mid:
                architecture.append('drop%.2f' % random.choice(self.params.get('dropout')))
            else:
                architecture.append(random.choice(self.params.get('dense_size')))
            r = random.random()

        return Network(
            architecture=architecture, opt=random.choice(self.params.get('optimizer')),
            lr=random.choice(self.params.get('optimizer_lr')), activation=random.choice(self.params.get('activation'))
        )

    def __mutate(self, base_net, change_number_cap=3):
        # type: (Network, int) -> Network

        if self.params == {}:
            raise Exception('Mutator not initialized.')

        possible_changes = [Network.add_layer, Network.remove_layer, Network.change_opt, Network.change_activation,
                            Network.change_lr_schedule]

        # Number of changes is capped, and distributed exponentially.
        n_of_changes = int(1 + np.random.exponential())
        if n_of_changes > change_number_cap:
            n_of_changes = change_number_cap

        for i in range(n_of_changes):

            base_net = random.choice(possible_changes)(base_net, self.params)

        return base_net


class Singleton(type):
    """
    Singleton to keep the only one instance of given class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Mutator(__Mutator):
    """
    __Mutator wrapper into Singleton, by using metaclasses.
    """
    __metaclass__ = Singleton
