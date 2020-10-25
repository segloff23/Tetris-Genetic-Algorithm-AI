
###############################################################################

# MODEL CREATION

import numpy as np

norm_big = lambda size : np.random.normal(loc=0, scale=20, size=size)
norm_small = lambda size : np.random.normal(loc=0, scale=2, size=size)
norm_abs_small = lambda size : np.abs(norm_small(size))

##############################################################################

# REQUIRED VARIABLES

model_names = ['weight', 'exponent', 'displacement']
model_sizes = [(None, 1), (None, 1), (None, 1)]
model_limits = [None, (0, 5), None]
model_initializers = [norm_big, norm_abs_small, norm_small]
model_dtypes = ['float32', 'float32', 'float32']
model_mutators = [norm_small, norm_abs_small, norm_small]

evaluator = lambda w, e, d, R : np.argmax(np.sum(
                                    w * np.power(np.abs(R - d), e), axis=0))

###############################################################################

def get_model_settings():

    return [model_names, model_sizes, model_limits,
            model_initializers, model_dtypes, model_mutators, ], evaluator

#EOF