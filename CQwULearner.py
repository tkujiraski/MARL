from CQLearner import *
from scipy import stats

class CQwULearner(CQLearner):
    def updateQ(self):
        if self.update:
            # 2つ以上のエージェントと同時に干渉した場合、実際には行動選択に使っていないQaugもアップデートしている(合理的)
            for other in self.others: # 行動を選んだ時の他のエージェントの状態
                q_id = self.id
                q_action = self.action
                q_state = self.old_s
                q_next_state = self.state
                q_target = (self.r + self.gamma*self.q.getMaxQ(self.state))
                q_now = self.Qaug[tuple(self.old_s+other)].q[self.action]
                self.Qaug[tuple(self.old_s+other)].q[self.action] = (1-self.alpha)*self.Qaug[tuple(self.old_s+other)].q[self.action] +self.alpha*q_target
        else:
            # 拡張されていない状態のQ値もアップデート
            self.q.qvalue[tuple(self.old_s + [self.action])] += self.alpha * (self.r + self.gamma * self.q.getMaxQ(self.state) - self.q.qvalue[tuple(self.old_s + [self.action])])
        return