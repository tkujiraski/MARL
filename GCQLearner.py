from CQLearner import *

class GCQLearner(CQLearner):
    def __init__(self, id, nstate, naction, params, env):
        super().__init__(id, nstate, naction, params, env)

    def selectAct(self):
        # 単独のQテーブルを使う場合はGreedyに動いてみる
        self.others = []
        rnd = np.random.rand()
        if tuple(self.state) in self.s_vec:
            # 干渉しているエージェントの状態のリストを返す
            others = self.env.getOthersState(self.id) # 行動は今の状態に基づく
            for d in self.s_vec[tuple(self.state)]:
                if d in others:
                    self.others.append(d)

            if len(self.others)==0:
                # Select a_k(t) according to Q_k
                self.action = self.q.getMaxAction(self.state)
            elif len(self.others)==1:
                # Select a_k(t) according to Q_k~aug
                if rnd < self.eps:
                    self.action = random.randint(0, self.naction - 1)
                else:
                    self.action = self.Qaug[tuple(self.state+self.others[0])].getMaxAction()
            else:
                # Select a_k(t) according to Q_k~aug
                # 候補となるQ_k~augが複数ある場合(近くにエージェントが複数いる場合）どうするか？

                self.action = atode
        else:
            self.action = self.q.getMaxAction(self.state)

        return