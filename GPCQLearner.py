from GCQLearner import *

class GPCQLearner(GCQLearner):
    def __init__(self, id, nstate, naction, params, env):
        super().__init__(id, nstate, naction, params, env)

    # PCQLearner2と同じ実装。多重継承できないんだっけ？
    # 多重継承はできるが、super()が２回呼ばれている可能性があるので、後で精査してから
    def updateQ(self):
        if self.update:
            for other in self.others:
                new_other = self.env.multi_q.getNewState(other)
                if tuple(self.state + new_other) in self.Qaug:
                    self.Qaug[tuple(self.old_s + other)].q[self.action] = (1 - self.alpha) * self.Qaug[tuple(self.old_s + other)].q[self.action] + self.alpha * (self.r + self.gamma * self.Qaug[tuple(self.state+new_other)].getMaxQ())
                else:
                    self.Qaug[tuple(self.old_s + other)].q[self.action] = (1 - self.alpha) * self.Qaug[tuple(self.old_s + other)].q[self.action] + self.alpha * (self.r + self.gamma * self.q.getMaxQ(self.state))
        return