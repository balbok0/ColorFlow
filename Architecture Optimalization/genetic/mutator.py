import collections
import os
import random

import numpy as np
from keras import Model
from typing import *

from helpers import helpers, helpers_mutate
from helpers.helpers_data import Array_Type
import log_save
from network import Network
from program_variables import program_params as const


class Mutator(object):
    def __init__(self, population_size: int=10, starting_population: List[Network]=None, params: Dict=None):
        """
        Creates a new instance of Mutator.

        :param params: A dictionary of choices which can modify the network.
        """
        assert population_size > 0

        if params is not None:
            const.mutations.fset(params)
            for i in ['kernel_size', 'conv_filers', 'dropout', 'dense_size', 'optimizer', 'optimizer_lr', 'activation']:
                assert i in list(params.keys()), "There should be %s in params keys." % i
                assert isinstance(params[i], list), "Key %s should be pointing to the list." % i

        self.networks = starting_population or []  # type: List[Network]
        self.population_size = population_size

    def get_best_architecture(
            self, x, y, validation_data=None, validation_split=None,
            generations=20, save_each_generation=True, saving_dir=None,
            epochs=2, batch_size=32, shuffle='batch', verbose=0
    ):
        # type: (Array_Type, Array_Type, Tuple, float, int, bool, str, int, int, str, int) -> Model
        """
        Main function of Mutator.\n
        Trains neural networks, and evolves them through generations, in order to find architecture, optimezr, etc.
        which allow for the best results given a dataset.

        :param generations: Number of generations.
        :param save_each_generation: Saves the best network from each generation in specified 'saving_dir'
        :param saving_dir: Directory to which models are saved.
        :param epochs: Number of epochs each network in each generation is trained.
        :param batch_size: Look at keras.models.Model.fit function documentation.
        :param shuffle: Look at keras.models.Model.fit function documentation.
        :param verbose: Look at keras.models.Model.fit function documentation.
                            Additionally, it specifies printing properties between generations.
        :return: Keras Model of neural network, which was the best in last generation, with already trained weights.
        """
        assert generations > 0

        if validation_data is None:
            if validation_split is None:
                raise AttributeError('Either validation_data or validation split cannot be None.')
            else:
                split = int(validation_split * len(x))
                validation_data = (x[split:], y[split:])
                x = x[:split]
                y = y[:split]

        if len(x.shape[1:]) >= 2:
            const.max_layers_limit.fset(int(np.log2(x.shape[1])))

        if len(self.networks) < self.population_size:
            self.networks = self.networks[:]
            for i in range(len(self.networks), self.population_size):
                self.networks.append(self.__create_random_model(input_shape=x.shape[1:], output_shape=len(y[0])))

        if saving_dir is None:
            from program_variables.file_loc_vars import models_saving_dir
            saving_dir = models_saving_dir
        assert os.path.exists(saving_dir)

        if not saving_dir.endswith('/'):
            saving_dir += '/'

        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)

        scores = {}  # Needed to suppress warnings.

        log_save.print_message('')
        log_save.print_message('Started a new job.')

        for i in range(generations):
            if verbose > 0:
                print('\n\nGeneration %d' % (i + 1))

            log_save.print_message('Starting training for generation %d' % (i + 1))

            for _, net in enumerate(self.networks):
                print('Network {}/{}'.format(_, len(self.networks)))
                net.fit(x, y, validation_data, validation_split,
                        epochs=epochs, batch_size=batch_size, shuffle=shuffle, verbose=verbose)

            log_save.print_message('Finished training for generation %d' % (i + 1))

            tmp_scores = {}  # type: Dict[Network, float]
            best_net = None  # type: Network

            for net in self.networks:
                tmp_scores[net] = net.score(y_true=validation_data[1], y_score=net.predict(validation_data[0]))
                if best_net is None or tmp_scores[net] > tmp_scores[best_net]:
                    best_net = net

            scores = collections.OrderedDict(tmp_scores)

            config = best_net.get_config()

            printing = {
                0: 'Generation {gen}. Best network scored: {score}'.format(gen=i + 1, score=scores[best_net]),
                1: 'Generation {gen}. Best network, with architecture: {arch}, optimizer {opt}, and activation '
                   'function {act}. It\'s score is {score}.'.format(
                    gen=i + 1,
                    arch=config['architecture'],
                    opt=config['optimizer'],
                    act=config['activation'],
                    score=scores[best_net]
                ),
                2: 'Generation {gen}. Best network, with architecture: {arch}, optimizer {opt}, and activation '
                   'function {act}. It\'s score is {score}.'.format(
                    gen=i + 1,
                    arch=config['architecture'],
                    opt=config['optimizer'],
                    act=config['activation'],
                    score=scores[best_net]
                )
            }
            log_save.print_message(printing[verbose])

            # Save the best net of current generation.
            if save_each_generation or i + 1 == generations:
                best_net.save(file_path='{dir}net_{num:03d}.h5'.format(dir=saving_dir, num=i + 1))

            if not i + 1 == generations:
                self.__evolve_networks(scores, self.population_size, input_shape=x.shape[1:], output_shape=len(y[0]))

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

    def __evolve_networks(self, scores, population_size, input_shape, output_shape):
        # type: (collections.OrderedDict[Network, float], int, List, int) -> None
        """
        Modifies list of networks in such a way, that bottom half is removed,
        and new networks are added instead of them, all based on networks from top half.

        :param scores: An OrderedDict, which points from the network to its score.
        :param population_size: Size, which population of networks should have at the end.
        """

        # Remove bottom half of population.
        for _ in range(int(population_size / 2)):
            scores.popitem(last=False)

        self.networks = list(scores.keys())
        p = np.exp(list(scores.values()))
        p = np.divide(p, sum(p))

        new_nets = []
        called_pairs = []  # type: List[List[Network, Network]]

        tmp_p = p
        tmp_nets = self.networks

        for _ in range(int((population_size - len(self.networks)) / 2)):
            if len(tmp_nets) >= 2:
                pair = np.random.choice(tmp_nets, size=2, replace=False, p=tmp_p)
                for i_pair in called_pairs:
                    if all(elem in pair for elem in i_pair):
                        for j in pair:
                            print([type(i) for i in tmp_p])
                            print(tmp_nets.index(j))
                            idx = tmp_p[tmp_nets.index(j)]
                            print(idx)
                            tmp_p = tmp_p[:idx] + tmp_p[idx:]
                            tmp_nets = tmp_nets[:j] + tmp_nets[idx:]
                        tmp_p = np.divide(p, sum(p))
                        _ -= 1

                else:
                    called_pairs += [pair, pair[::-1]]
                    new_nets += self.__mutate(pair[0], pair[1], input_shape=input_shape, output_shape=output_shape)
            elif tmp_nets:
                new_nets += [self.__mutate_random(tmp_nets[0], input_shape=input_shape, output_shape=output_shape)]
            tmp_p = p
            tmp_nets = self.networks

        for net in new_nets:
            self.networks.append(net)

    def __create_random_model(self, input_shape, output_shape):
        # type: (List, int) -> Network
        """
        Creates a random model, based of parameters choices given in this Mutator.

        :return: Random Network, with a random architecture, optimizers, etc.
        """
        if const.input_dim.fget() < 3:
            architecture = [random.choice(const.mutations.fget()['dense_size'])]
        else:
            architecture = [(random.choice(const.mutations.fget()['kernel_size']),
                            random.choice(const.mutations.fget()['conv_filters'])),
                            random.choice(const.mutations.fget()['dense_size'])]

        n_layers = 2  # since there's always at least one conv, dense layer.

        # r_min - adding any new sequence.,
        r_min = .5

        if const.input_dim.fget() >= 3:
            # Convolution/Maxout part of architecture
            r = random.random()
            while (r > r_min and n_layers < const.max_depth != 0) or n_layers < const.min_depth != 0:
                n_layers = len(architecture)
                architecture = helpers_mutate.add_arch_conv_max(architecture)
                r = random.random()

        # Dense/Dropout part of architecture
        r = random.random()
        while (r > r_min and n_layers < const.max_depth != 0) or n_layers < const.min_depth != 0:
            n_layers = len(architecture)
            architecture = helpers_mutate.add_arch_dense_drop(architecture)
            r = random.random()

        return Network(
            architecture=architecture,
            input_shape=input_shape,
            output_shape=output_shape,
            opt=random.choice(const.mutations.fget()['optimizer']),
            lr=random.choice(const.mutations.fget()['optimizer_lr']),
            activation=random.choice(const.mutations.fget()['activation'])
        )

    def __mutate(self, base_net_1, base_net_2, input_shape, output_shape, change_number_cap=3):
        # type: (Network, Network, List, int, int) -> List[Network]
        """
        Creates and returns two new Networks, based on passed in parent Networks.

        :param base_net_1: A first parent network on which mutation is based.
        :param base_net_2: A second parent network on which mutation is based.
        :param change_number_cap: Cap number of a random changes in case of random mutations.
        :return: List of 2 Networks, which are based on passed in parent Networks.
        """

        if random.random() < const.parent_to_rand_chance:
            return [
                self.__mutate_random(base_net_1,
                                     change_number_cap=change_number_cap,
                                     input_shape=input_shape,
                                     output_shape=output_shape),
                self.__mutate_random(base_net_2,
                                     change_number_cap=change_number_cap,
                                     input_shape=input_shape,
                                     output_shape=output_shape)
            ]

        elif random.random() < 0.5:
            return self._mutate_parent(base_net_1, base_net_2, input_shape, output_shape)
        else:
            return self._mutate_parent_2(base_net_1, base_net_2, input_shape, output_shape)

    def _mutate_parent(self, base_net_1, base_net_2, input_shape, output_shape):
        # type: (Network, Network, List, int) -> List[Network]
        """
        Creates two new Networks, both based on combination of given parents.

        :param base_net_1: A first parent network on which mutation is based.
        :param base_net_2: A second parent network on which mutation is based.
        :return: List of 2 Networks, both of which have features of both parent Networks.
        """
        dense_idx_1, weight_idx_1 = helpers.find_first_dense(base_net_1.model)
        dense_idx_2, weight_idx_2 = helpers.find_first_dense(base_net_2.model)
        dense_idx_1 -= 2
        dense_idx_2 -= 2

        conv_1 = Network(
            architecture=base_net_1.arch[:dense_idx_1] + base_net_2.arch[dense_idx_2:],
            input_shape=input_shape,
            output_shape=output_shape,
            opt=base_net_2.opt,
            activation=base_net_2.act,
            callbacks=base_net_2.callbacks
        )

        conv_2 = Network(
            architecture=base_net_2.arch[:dense_idx_2] + base_net_1.arch[dense_idx_1:],
            input_shape=input_shape,
            output_shape=output_shape,
            opt=base_net_1.opt,
            activation=base_net_1.act,
            callbacks=base_net_1.callbacks
        )

        conv_1.model.set_weights(  # Set Conv-Max weights
            base_net_1.model.get_weights()[:weight_idx_1] + conv_1.model.get_weights()[weight_idx_1:]
        )
        conv_1.model.set_weights(  # Set Dense-Drop weights
            conv_1.model.get_weights()[:weight_idx_1 + 1] + base_net_2.model.get_weights()[weight_idx_2 + 1:]
        )

        conv_2.model.set_weights(  # Set Conv-Max weights
            base_net_2.model.get_weights()[:weight_idx_2] + conv_2.model.get_weights()[weight_idx_2:]
        )
        conv_2.model.set_weights(  # Set Dense-Drop weights
            conv_2.model.get_weights()[:weight_idx_2 + 1] + base_net_1.model.get_weights()[weight_idx_1 + 1:]
        )
        return [conv_1, conv_2]

    def _mutate_parent_2(self, base_net_1, base_net_2, input_shape, output_shape):
        # type: (Network, Network, List, int) -> List[Network]
        """
        Creates two new Networks, both based on combination of given parents.
        More random than :ref:`main _mutate_parent<mutator.__Mutator#_mutate_parent>`.

        :param base_net_1: A first parent network on which mutation is based.
        :param base_net_2: A second parent network on which mutation is based.
        :return: List of 2 Networks, both of which have features of both parent Networks.
        """
        new_nets = []
        for _ in range(2):
            max_seq_start_idx = 0
            drop_seq_start_idx = helpers.find_first_dense(base_net_2.model)[0] - 2
            idx = 0
            max_seq_idx = []
            drop_seq_idx = []

            for l in base_net_1.arch:
                if helpers.arch_type(l) == 'max':
                    max_seq_idx.append((0, max_seq_start_idx, idx))
                    max_seq_start_idx = idx + 1
                elif helpers.arch_type(l) == 'drop':
                    drop_seq_idx.append((0, drop_seq_start_idx, idx))
                    drop_seq_start_idx = idx + 1
                idx += 1

            n_max_seq = [len(max_seq_idx)]
            n_drop_seq = [len(drop_seq_idx)]

            idx = 0
            max_seq_start_idx = 0
            drop_seq_start_idx = helpers.find_first_dense(base_net_2.model)[0] - 2

            for l in base_net_2.arch:
                if helpers.arch_type(l) == 'max':
                    max_seq_idx.append((1, max_seq_start_idx, idx))
                    max_seq_start_idx = idx + 1
                elif helpers.arch_type(l) == 'drop':
                    drop_seq_idx.append((1, drop_seq_start_idx, idx))
                    drop_seq_start_idx = idx + 1
                idx += 1

            n_max_seq = random.choice(n_max_seq + [len(max_seq_idx) - n_max_seq[0], int(len(max_seq_idx) / 2)])
            n_max_seq = max(1, n_max_seq)
            n_drop_seq = random.choice(n_drop_seq + [len(drop_seq_idx) - n_drop_seq[0], int(len(drop_seq_idx) / 2)])
            n_drop_seq = max(1, n_drop_seq)

            archs = [base_net_1.arch, base_net_2.arch]
            new_arch = []

            max_idxs = np.random.choice(max_seq_idx, size=n_max_seq, replace=n_max_seq > len(max_seq_idx))
            drop_idxs = np.random.choice(drop_seq_idx, size=n_drop_seq, replace=n_drop_seq > len(drop_seq_idx))

            for i in max_idxs:
                a = archs[i[0]]
                new_arch += a[i[1]:i[2] + 1]

            for i in drop_idxs:
                a = archs[i[0]]
                new_arch += a[i[1]:i[2] + 1]

            new_net = Network(
                architecture=new_arch,
                callbacks=random.choice([base_net_1.callbacks, base_net_2.callbacks]),
                opt=random.choice([base_net_1.opt, base_net_2.opt]),
                activation=random.choice([base_net_1.act, base_net_2.act]),
                input_shape=input_shape,
                output_shape=output_shape,
            )

            nets = [base_net_1, base_net_2]  # type: List[Network]

            idx = 1
            for i in max_idxs:
                a = nets[i[0]]
                for j in range(i[1] + 1, i[2] + 1):
                    kernel_filter = a.model.get_layer(index=j).get_weights()[1]
                    new_weights = [new_net.model.get_layer(index=idx).get_weights()[0], kernel_filter]
                    new_net.model.get_layer(index=idx).set_weights(new_weights)
                    idx += 1
                idx += 1  # for MaxPool

            idx += 1

            for i in drop_idxs:
                a = nets[i[0]]
                for j in range(i[1] + 2, i[2] + 2):
                    w_a = a.model.get_layer(index=j).get_weights()
                    w_n = new_net.model.get_layer(index=idx).get_weights()

                    new_weights = w_a[0][:len(w_n[0])]
                    if len(w_a[0]) < len(w_n[0]):
                        new_weights = np.concatenate((new_weights, w_n[0][len(new_weights):]), axis=0)
                    new_weights = [new_weights, w_a[1]]

                    new_net.model.get_layer(index=idx).set_weights(new_weights)
                    idx += 1
                idx += 1  # for Dropout

            new_nets += [new_net]
        return new_nets

    def __mutate_random(self, base_net, input_shape, output_shape, change_number_cap=3):
        # type: (Network, List, int, int) -> Network
        """
        Given a network, returns a new Network, with a random number of mutations (capped at given number).

        :param base_net: A network to which mutations should be based. It's not affected.
        :param change_number_cap: Maximal number of changes.
        :return: A new, mutated Network.
        """

        possible_changes = [
            helpers_mutate.add_dense_drop,
            helpers_mutate.add_conv_max,
            helpers_mutate.remove_dense_drop,
            helpers_mutate.remove_conv_max,
            helpers_mutate.change_opt,
            helpers_mutate.change_activation,
            helpers_mutate.change_lr_schedule
        ]

        probabilities = [10, 9, 7, 7, 4, 5, 2]
        probabilities = np.divide(probabilities, 1. * np.sum(probabilities))  # Normalization, for probabilities.

        # Number of changes is capped, and distributed exponentially.
        n_of_changes = int(1 + np.random.exponential())
        if n_of_changes > change_number_cap:
            n_of_changes = change_number_cap

        for i in range(n_of_changes):
            base_net = np.random.choice(possible_changes, p=probabilities)(base_net, input_shape, output_shape)

        return base_net
