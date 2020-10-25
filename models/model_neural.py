
###############################################################################

# MODEL CREATION

import numpy as np

norm_big = lambda size : np.random.normal(loc=0, scale=20, size=size)
norm_small = lambda size : np.random.normal(loc=0, scale=2, size=size)
norm_abs_small = lambda size : np.abs(norm_small(size))
relu = lambda x : (np.abs(x) + x) / 2
##############################################################################

N_nodes = 10

# REQUIRED VARIABLES

model_names = ['input_weight', 'input_bias', 'output_weight', 'output_bias']
model_sizes = [(N_nodes, None), (N_nodes, 1), (1, N_nodes), (1, 1)]
model_limits = [None for _ in model_names]
model_initializers = [norm_small for _ in model_names]
model_dtypes = ['float32' for _ in model_names]
model_mutators = [norm_small for _ in model_names]

evaluator = lambda W_in, B_in, w_out, b_out, R : np.argmax(w_out
                                                           @ relu(W_in @ R
                                                                  + B_in)
                                                           + b_out)

###############################################################################

def get_model_settings():

    return [model_names, model_sizes, model_limits,
            model_initializers, model_dtypes, model_mutators, ], evaluator

#EOF