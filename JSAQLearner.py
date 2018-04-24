from Agent import *
from DrawAveStepsGrapth import *
import numpy as np
from numpy.random import randint
from datetime import datetime
import csv
import time

class JSAQLearner(Agent):
    def __init__(self, id, nstate, naction, params, env):
        super().__init__(id, nstate, naction, params, env)

    @classmethod
    def mixed(cls):
        return True

    @classmethod
    def joint(cls):
        return True

