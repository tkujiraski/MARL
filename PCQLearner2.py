from CQLearner import *

class PCQLearner2(CQLearner):
    def __init__(self, id, nstate, naction, params, env):
        super().__init__(id, nstate, naction, params, env)

    # ここを変える必要あり。次の遷移状態での(s_k, s_l)が拡張されているかどうかで決める
    def updateQ(self):
        if self.update:
            for other in self.others:
                new_other = self.env.multi_q.getNewState(other)
                if tuple(self.state + new_other) in self.Qaug:
                    self.Qaug[tuple(self.old_s + other)].q[self.action] = (1 - self.alpha) * self.Qaug[tuple(self.old_s + other)].q[self.action] + self.alpha * (self.r + self.gamma * self.Qaug[tuple(self.state+new_other)].getMaxQ())
                else:
                    self.Qaug[tuple(self.old_s + other)].q[self.action] = (1 - self.alpha) * self.Qaug[tuple(self.old_s + other)].q[self.action] + self.alpha * (self.r + self.gamma * self.q.getMaxQ(self.state))
        return
