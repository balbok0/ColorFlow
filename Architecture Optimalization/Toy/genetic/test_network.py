from mutator import Mutator
from network import Network

# Start
# When a new requirement is added tbd on the beginning of workings of mutator/network, please include that fn call
# here, so that it's not repeated in every test.
Network.prepare_data('testing')
m = Mutator()
n = Network(architecture=[((3, 3), 32), ((3, 3), 32), ((3, 3), 32), 'max', 30, 30, 'drop0.4', 10])


# End of Start.


def check_drop_last_in_arch():
    n2 = Network(architecture=[((3, 3), 32), 'max', ((5, 5), 6), 32, 'drop0.4'], opt='sgd', activation='relu')
    assert not (isinstance(n2.arch[-1], str) and n2.arch[-1].startswith('drop'))


def check_nans():
    from mutator import Mutator
    _m = Mutator()
    print(_m.get_best_architecture(verbose=1, dataset='testing', population_size=6, generations=20))


def check_add_conv_max_sequence():
    from helpers_mutate import add_conv_max
    n2 = Network(architecture=[((3, 3), 32), ((3, 3), 32), ((3, 3), 32), 'max', 30, 30, 'drop0.4', 10])
    print(add_conv_max(n2, m.params, 3).arch)


def check_add_dense_drop_seq():
    from helpers_mutate import add_dense_drop
    n2 = Network(architecture=[((3, 3), 32), ((3, 3), 32), ((3, 3), 32), 'max', 30, 30, 'drop0.4', 10])
    print(add_dense_drop(n2, m.params).arch)


def check_rmv_conv_max_sequence():
    from helpers_mutate import remove_conv_max
    n2 = Network(architecture=[
        ((3, 3), 32), 'max', ((3, 3), 32), ((3, 3), 32), 'max', ((3, 3), 32), ((3, 3), 32), 'max', 10, 'drop0.2', 10
    ])
    print(remove_conv_max(n2, m.params).arch)


def main():
    check_rmv_conv_max_sequence()


if __name__ == '__main__':
    main()
