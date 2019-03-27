from Agent import *
import numpy as np
import math
import csv

# 学習の進度によるOptimal Qとの差の変化を計算するクラス
class CompareLearner(Agent):
    def __init__(self, id, nstate, naction, params, env):
        super().__init__(id, nstate, naction, params, env)
        self.N = params['window_size']
        self.reward = np.zeros(nstate+[naction,self.N])
        self.count = np.zeros(nstate + [naction], dtype=int)
        self.optQ = np.load(params["OptQ"])
        self.rmse = []

    def sparse_interaction(self):
        # ここで報酬を記録する
        self.reward[tuple(self.old_s+[self.action,(self.count[tuple(self.old_s+[self.action])] % self.N)])] = self.r
        self.count[tuple(self.old_s + [self.action])] += 1
        return

    def end_episode(self, ep):
        # 一定間隔ごとに現在のQ値とOptimalQ値を比較した結果を書き出す
        e = ep // 1000
        ep += 1
        if ep % 1000 == 0:
            self.diff = self.q.qvalue - self.optQ
            self.rmse.append(0.0)
            for i in range(self.diff.shape[0]):
                tmp = self.diff[i][0]*self.diff[i][0] + self.diff[i][1]*self.diff[i][1] + self.diff[i][2]*self.diff[i][2] + self.diff[i][3]*self.diff[i][3]
                self.rmse[e] = self.rmse[e] + tmp
            self.rmse[e] = math.sqrt(self.rmse[e])
        if ep == self.params["maxEpisodes"]:
            with open(self.params["filename"], 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(self.rmse)

    def setFilename(self,filename):
        self.filename = filename

    def saveER(self,filename):
        ER = self.reward.mean(axis=len(self.nstate)+1)
        np.save(filename + ".npy", ER)

    def saveCount(self, filename):
        np.save(filename + ".npy", self.count)
