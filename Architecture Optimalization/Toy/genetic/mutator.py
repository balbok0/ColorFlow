import collections
import os
import random

import numpy as np
from keras import Model
from typing import List, Dict

import helpers_mutate
import log_save
from network import Network
from program_variables import program_params as const

'''
Actual Mutator class is in the bottom of this file!
It wraps __Mutator class!!!
'''


class __Mutator(object):

    def __init__(self, params=None):
        # type: (__Mutator, Dict[str, List]) -> None
        """
        Creates a new instance of Mutator.

        :param params: A dictionary of choices which can modify the network.
        """
        if params is None:
            params = const.mutations

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
        # type: (__Mutator, int, int, str, int, int, str, int, str) -> Model
        """
        Main function of Mutator.\n
        Trains neural networks, and evolves them through generations, in order to find architecture, optimezr, etc.
        which allow for the best results given a dataset.

        :param population_size: Number of neural networks in one genration.
        :param generations: Number of generations.
        :param saving_dir: Directory to which models are saved.
        :param epochs: Number of epochs each network in each generation is trained.
        :param batch_size: Look at keras.models.Model.fit function documentation.
        :param shuffle: Look at keras.models.Model.fit function documentation.
        :param verbose: Look at keras.models.Model.fit function documentation.
                            Additionally, it specifies printing properties between generations.
        :param dataset: Dataset to which neural network should be optimized.
        :return: Keras Model of neural network, which was the best in last generation, with already trained weights.
        """
        Network.prepare_data(dataset)

        if saving_dir is None:
            from program_variables.file_loc_vars import models_saving_dir
            saving_dir = models_saving_dir

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

        config = best_net.get_config()

        log_save.print_message(
            'Best network, with architecture: '
            '{arch}, optimizer {opt}, callbacks {call}, and activation function {act}. '
            'It\'s score is {score}.'.format(
                arch=config['architecture'],
                opt=config['optimizer'],
                call=config['callbacks'],
                act=config['activation'],
                score=best_score
            )
        )
        return best_net.model

    def __create_random_model(self):
        # type: (__Mutator) -> Network
        """
        Creates a random model, based of parameters choices given in this Mutator.

        :return: Random Network, with a random architecture, optimizers, etc.
        """
        architecture = [(random.choice(self.params['kernel_size']),
                         random.choice(self.params['conv_filters']))]

        n_layers = 1  # since there's always at least one dense layer.

        # Two variables for probability of:
        #   r_min - adding any new layer,
        #   r_mid - distinguishing between two types of layers for each part of arch.
        r_min = .33
        r_mid = .66

        # Convolution/Maxout part of architecture
        r = random.random()
        while r > r_min and n_layers < const.max_depth:
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
        while r > r_min and n_layers < const.max_depth:
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
        """
        Given a network, returns a new Network, with a random number of mutations (capped at given number).

        :param base_net: A network to which mutations should be based. It's not affected.
        :param change_number_cap: Maximal number of changes.
        :return: A new, mutated Network.
        """
        if self.params == {}:
            raise Exception('Mutator not initialized.')

        possible_changes = [helpers_mutate.add_layer, helpers_mutate.remove_layer, helpers_mutate.change_opt,
                            helpers_mutate.change_activation, helpers_mutate.change_lr_schedule]

        probabilities = [3, 2, 1, 1, 1]
        probabilities = np.divide(probabilities, 1. * np.sum(probabilities))  # Normalization, for probabilities.

        # Number of changes is capped, and distributed exponentially.
        n_of_changes = int(1 + np.random.exponential())
        if n_of_changes > change_number_cap:
            n_of_changes = change_number_cap

        for i in range(n_of_changes):
            base_net = np.random.choice(possible_changes, p=probabilities)(base_net, self.params)

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
