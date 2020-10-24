
###############################################################################

# MODEL CREATION

import numpy as np

norm_big = lambda size : np.random.normal(loc=0, scale=20, size=size)
norm_small = lambda size : np.random.normal(loc=0, scale=2, size=size)

##############################################################################

# REQUIRED VARIABLES
model_names = ['weight']
model_sizes = [(None, 1)]
model_limits = [None]
model_initializers = [norm_big]
model_dtypes = ['float32']
model_mutators = [norm_small]

evaluator = lambda w, R : np.argmax(np.sum(w * R, axis=0))

###############################################################################

def get_model_settings():

    return [model_names, model_sizes, model_limits,
            model_initializers, model_dtypes, model_mutators, ], evaluator

#EOF