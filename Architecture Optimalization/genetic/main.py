def main():
    from mutator import Mutator
    m = Mutator()
    print(m.get_best_architecture(verbose=1, epochs=1, dataset='testing', population_size=3, generations=20))


if __name__ == '__main__':
    main()
