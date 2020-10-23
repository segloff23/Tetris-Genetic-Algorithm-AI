# Tetris-Genetic-Algorithm-AI
Work in Progress: A Tetris AI developed using evolutionary algorithms. Sample ecosystems evolved over five generations included.

Created in Python 3.6

Required modules:

numpy

matplotlib
Optional modules for visualization:

cv2
Optional modules for data tracking:

pytorch
Currently all parameters are adjusted within the header of each script.

Main scripts:

eam_train: Evolve a new population with the given parameters

eam_test: Test the best individual of a saved generation

eam_animate: Test, create plots, or generate a video (.mp4) of the best indiviudal's play

Supporting files:

tetris: Holds the environment for running games, along with two support functions to simulate games with or without multiprocessing (recommended)

heuristics: Calculates certain ratings, like the pile height or number of holes, to be used by the population to evaluate the optimal board

ecosystem: Holds a set of indiviudals with methods to evolve them within a certain tetris environment

Set up the following directory map to use:

--main

  --ecosystems

  --logs

  --videos

  -__init__.py

  -eam_animate.py

  -eam_test.py

  -eam_train.py

  -ecosystem.py

  -heuristics.py

  -tetris.py
