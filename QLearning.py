import QTable
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

class QLearning:
    def __init__(self, nstate, naction, eps, gamma, alpha, maxEpisode, maxSteps, initFunc, act, check_goal, vals):
        """
        Q Learningのパラメータを設定する
        :param: nstate:
        :param: naction:
        :param eps:
        :param gamma:
        :param maxEpisode:
        :param maxSteps:
        :param initFunc:
        :param act
        :param check_goal
        """
        self.nstate = nstate
        self.naction = naction
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.maxEpisode = maxEpisode
        self.stepsForGoal = []
        self.earnedReward = []
        self.maxSteps = maxSteps
        self.initFunc = initFunc
        self.act = act
        self.check_goal = check_goal
        self.vals = vals
        self.q = QTable.QTable(nstate, naction)
        self.adic = [] #辞書から配列に変更

    def alabel(self,adic):
        self.adic = adic

    def learn(self):
        for ep in range(self.maxEpisode):
            print('#episode %d' % ep)
            # 状態sを初期化
            self.state = self.initFunc()
            earnedReward = 0

            for step in range(self.maxSteps):

                # 状態sでの行動aを選択
                rnd = np.random.rand()
                if rnd < self.eps:
                    # ランダムな行動選択
                    a = random.randint(0,self.naction-1)
                else:
                    a = self.q.getMaxAction(tuple(self.state))

                # 行動aを実行し、r,s'を観測
                old_s = self.state.copy()
                r = self.act(self.state, a) #self.state自体が書き換わる
                earnedReward += r
                ##print('------')
                ##print(r)
                ##print(old_s,next_s)
                ##print('------')

                # QTableを更新
                self.q.qvalue[tuple(old_s+[a])] += self.alpha * (r + self.gamma*self.q.getMaxQ(self.state)-self.q.qvalue[tuple(old_s+[a])])

                # 状態sを更新
                # self.state = next_s すでにactで更新されている

                # sが終端状態ならばエピソードを終了
                if self.check_goal(self.state):
                    print('終了状態へ到達 %d step' % step)
                    break
            self.stepsForGoal.append(step)
            self.earnedReward.append(earnedReward)
        print('学習終了')

    def plot_learning_curve(self):
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(range(self.maxEpisode), self.stepsForGoal, c='red')
        ax.plot(range(self.maxEpisode), self.earnedReward, c='blue')
        plt.show()

    def save_learning_curve(self, filename):
        with open(filename, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(self.nstate+[self.naction, self.eps, self.gamma, self.alpha, self.maxEpisode, self.maxSteps]+self.vals)
            writer.writerow(self.stepsForGoal)

    def replay(self):
        """最大のQ値選択のみでGoalまでのアクション系列を返す"""
        self.state = self.initFunc()
        self.states = [self.state.copy()]
        self.actions = []
        for step in range(self.maxSteps):
            # 状態sでの行動aをQ値の最大値で選択
            a = self.q.getMaxAction(tuple(self.state))
            if self.adic:
                self.actions.append(self.adic[a])
            else:
                self.actions.append(a)

            # 行動aを実行し、r,s'を観測
            r = self.act(self.state, a)  # self.state自体が書き換わる
            self.states.append(self.state.copy())
            # sが終端状態ならばreplayを終了
            if self.check_goal(self.state):
                break
        return self.states, self.actions