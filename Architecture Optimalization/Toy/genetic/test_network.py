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
    print('check_drop_last_in_arch start')
    n2 = Network(architecture=[((3, 3), 32), 'max', ((5, 5), 6), 32, 'drop0.4'], opt='sgd', activation='relu')
    assert not (isinstance(n2.arch[-1], str) and n2.arch[-1].startswith('drop'))
    print('check_drop_last_in_arch end')
    print('')


def check_nans():
    pass
    # from mutator import Mutator
    # print('check_nans start')
    # _m = Mutator()
    # print(_m.get_best_architecture(verbose=1, dataset='testing', population_size=6, generations=20))
    # print('check_nans end')
    # print('')


def check_add_conv_max_seq():
    from helpers_mutate import add_conv_max
    n2 = Network(architecture=[((3, 3), 32), ((3, 3), 32), ((3, 3), 32), 'max', 30, 30, 'drop0.4', 10])
    print(add_conv_max(n2, m.params, 3).arch)


def check_add_dense_drop_seq():
    from helpers_mutate import add_dense_drop
    print('check_add_dense_drop_seq start')
    n2 = Network(architecture=[((3, 3), 32), ((3, 3), 32), ((3, 3), 32), 'max', 30, 30, 'drop0.4', 10])
    print(add_dense_drop(n2, m.params).arch)
    n2 = Network(architecture=[((7, 7), 16), 'max', 128])
    print(add_dense_drop(n2, m.params).arch)
    print('check_add_dense_drop_seq end')
    print('')


def check_priv_dense_drop_seq():
    from helpers_mutate import __add_dense_drop
    print('check_add_dense_drop_seq start')
    n2 = Network(
        [((3, 3), 16), ((7, 7), 16), 'max', ((3, 3), 16), ((3, 3), 16), ((3, 3), 16), 'max', ((3, 3), 8),
         32, 'drop0.30', 128, 'drop0.30', 32]
    )
    print(__add_dense_drop(n2, 11, 32, 'drop0.70'))
    print('check_priv_add_dense_drop_seq end')
    print('')


def check_rmv_conv_max_seq():
    from helpers_mutate import remove_conv_max
    n2 = Network(architecture=[
        ((3, 3), 32), 'max', ((3, 3), 32), ((3, 3), 32), 'max', ((3, 3), 32), ((3, 3), 32), 'max', 10, 'drop0.2', 10
    ])
    print(remove_conv_max(n2, m.params).arch)


def check_rmv_dense_drop_seq():
    from helpers_mutate import remove_dense_drop
    n2 = Network(architecture=[
        ((3, 3), 32), 'max', ((3, 3), 32), ((3, 3), 32), 'max', 10, 'drop0.2', 20, 'drop0.3', 10
    ])
    print(remove_dense_drop(n2, m.params).arch)


def main():
    """
    Always calls all the functions in this file, in alphabetical order. Does not call itself.
    """
    import sys, inspect

    current_module = sys.modules[__name__]
    all_functions = inspect.getmembers(current_module, inspect.isfunction)
    for key, value in all_functions:
        if inspect.getargspec(value)[0] == [] and not key == 'main':
            value()


if __name__ == '__main__':
    main()
