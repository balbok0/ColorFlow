# Network parameters
"""
- When any of them is set to 0, then this boundary is ignored.
"""
max_n_weights = 7500000
max_depth = 10
min_depth = 3

# Mutation parameters
_mutations = {
                'kernel_size': [(3, 3), (5, 5)],
                'conv_filters': [8, 16],
                'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
                'dense_size': [32, 64, 128, 256],
                'optimizer': ['adam', 'sgd', 'nadam'],
                'optimizer_lr': [None, .0001, .0003, .001, .003, .01],
                'learning_decay_type': ['linear', 'exp'],
                'learning_decay_rate': [0.7, 0.8, 0.9],
                'activation': ['relu', 'sigmoid', 'tanh']
            }


@property
def mutations():
    return _mutations


@mutations.setter
def mutations(val):
    global _mutations
    _mutations = val


# Above this threshold mutations are parent ones, below are random. Range is (0, 1). Used in Mutator.
# If set to 1, all mutations are random. If set to 0, all mutations are parent.
parent_to_rand_chance = 0.6667

# Dataset variables
n_train = 50000

# Development variables
debug = True
deep_debug = False
