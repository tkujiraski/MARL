import numpy as np
from QTable import *
import random

class Agent:
    # 強化学習のエージェントの基本的な機能の実装
    # 単体ではe-GreedyなQ-Learningを実行
    def __init__(self, id, nstate, naction, params, env):
        self.id = id
        self.nstate = nstate
        self.naction = naction
        self.params = params
        self.eps = params['eps']
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.env = env
        self.init_qvalue = []
        if 'init_qvalue'+str(id) in params:
            self.init_qvalue = params['init_qvalue'+str(id)]
        self.action = -1
        self.state = []
        self.old_s = []
        self.r = 0
        self.earned_reward = 0
        self.q = QTable(nstate,naction, self.init_qvalue)
        return

    # アルゴリズムによって上書きする
    @classmethod
    def mixed(cls):
        return False

    @classmethod
    def joint(cls):
        return False

    def initState(self):
        # 状態を初期化する
        self.earned_reward = 0

    def selectAct(self):
        # デフォルトではe-GreedyでargmaxQ(a|s)
        rnd = np.random.rand()
        if rnd < self.eps:
            # ランダムな行動選択
            self.action = random.randint(0, self.naction - 1)
        else:
            self.action = self.q.getMaxAction(self.state)
        return

    def sparse_interaction(self):
        return

    def updateQ(self):
        # デフォルトではQ-Learning,PHC,WoLF-PHCに共通の実装
        self.q.qvalue[tuple(self.old_s+[self.action])] += self.alpha * (self.r + self.gamma*self.q.getMaxQ(self.state)-self.q.qvalue[tuple(self.old_s+[self.action])])
        return

    def updatePi(self):
        # Policyを更新する方法を選択
        # デフォルトでは空
        return

    def get_state(self):
        # 現在の状態を返す
        return self.state

    def get_augmented_states(self):
        return []
