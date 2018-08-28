def main():
    from mutator import Mutator
    m = Mutator()
    print(m.get_best_architecture(verbose=1, dataset='colorflow', population_size=20, generations=20))


if __name__ == '__main__':
    main()
