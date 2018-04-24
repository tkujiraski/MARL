from CQLearner import *

class PCQLearner(CQLearner):
    def __init__(self, id, nstate, naction, params, env):
        super().__init__(self, id, nstate, naction, params, env)

    # ここを変える必要あり。q.getMaxQの代わりに、次のステップも干渉する確率を踏まえて推定する。後回し
    def updateQ(self):
        if self.update:
            all = self.W_count[tuple(self.state)].sum()
            for other in self.others:
                # 回数を集計
                aug = 0
                for i in range(self.naction):
                    aug += self.Waug_count[tuple(self.state+other+[i])]
                self.Qaug[tuple(self.old_s + other)].q[self.action] = (1 - self.alpha) * self.Qaug[tuple(self.old_s + other)].q[self.action] + self.alpha * (self.r + self.gamma * self.q.getMaxQ(self.state))
        return
