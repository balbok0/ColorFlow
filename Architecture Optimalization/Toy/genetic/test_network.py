from network import Network


def start():
    # When a new requirement is added tbd on the beginning of workings of mutator/network, please include that fn call
    # here, so that it's not repeated in every test.
    Network.prepare_data('testing')


def check_drop_last_in_arch():
    n = Network(architecture=[((3, 3), 32), 'max', ((5, 5), 6), 32, 'drop0.4'], opt='sgd', activation='relu')
    assert not (isinstance(n.arch[-1], str) and n.arch[-1].startswith('drop'))


def check_nans():
    from mutator import Mutator
    _m = Mutator()
    print(_m.get_best_architecture(verbose=1, dataset='testing', population_size=6, generations=20))


def main():
    start() # Do not delete. It's needed for correct test.
    check_drop_last_in_arch()


if __name__ == '__main__':
    main()
