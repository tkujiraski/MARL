import sys
import numpy as np
import time

from MultiAgentRL import MultiAgentRL
from DrawAveStepsGrapth import *
from Agent import *

class Environment():
    def __init__(self, learner, params):
        pass

    def env_init(self):
        pass

    def env_update(self, agents):
        pass

    def observe(self, agent):
        pass

    def check_goal(self):
        pass

    def getOthersState(self, id):
        pass

    def getNewState(self, old_state):
        # 汎用的ではないので修正必要
        pass

    def getOthersOldState(self, id):
        # id以外のエージェントの１つ前の状態のリストを返す。状態は１次元でもリストで定義する
        pass
