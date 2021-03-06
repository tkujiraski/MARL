from Agent import *
import numpy as np

# CQ-Learning用のデータを準備するために、報酬の平均を記録するクラス
class ERLearner(Agent):
    def __init__(self, id, nstate, naction, params, env):
        super().__init__(id, nstate, naction, params, env)
        self.N = params['window_size']
        self.reward = np.zeros(nstate+[naction,self.N])
        self.count = np.zeros(nstate + [naction], dtype=int)

    def sparse_interaction(self):
        # ここで報酬を記録する
        self.reward[tuple(self.old_s+[self.action,(self.count[tuple(self.old_s+[self.action])] % self.N)])] = self.r
        self.count[tuple(self.old_s + [self.action])] += 1
        return

    def end_episode(self, ep):
        # 一定間隔ごとにQ値とER値とCount値を書き出す
        ep += 1
        if ep % 1000 == 0:
            self.q.saveQvalue(self.filename + '_qvalue_ep' + str(ep))
            self.saveER(self.filename + '_ER_ep' + str(ep))
            self.saveCount(self.filename + '_count_ep' + str(ep))

    def setFilename(self,filename):
        self.filename = filename

    def saveER(self,filename):
        ER = self.reward.mean(axis=len(self.nstate)+1)
        np.save(filename + ".npy", ER)

    def saveCount(self, filename):
        np.save(filename + ".npy", self.count)


