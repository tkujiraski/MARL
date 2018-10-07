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
                self.action = self.q.getMaxAction(self.state) # greedyに行動する
            elif len(self.others)==1:
                # Select a_k(t) according to Q_k~aug
                if rnd < self.eps:
                    self.action = random.randint(0, self.naction - 1)
                else:
                    self.action = self.Qaug[tuple(self.state+self.others[0])].getMaxAction()
            else:
                # Select a_k(t) according to Q_k~aug
                # 候補となるQ_k~augが複数ある場合(近くにエージェントが複数いる場合）どうするか？
                # a) 一番近いエージェントのQを考慮する → ドメイン依存な処理になってしまう
                # b) ランダムに選ぶ
                # c) 最大のQ値を選ぶ(最も考慮すべき状況・行動と考えられる？)

                # まずb)を実装してみる。D論ではc)との比較もしても良いかも
                if rnd < self.eps:
                    self.action = random.randint(0, self.naction - 1)
                else:
                    q_selected = random.randint(0, len(self.others)-1)
                    self.action = self.Qaug[tuple(self.state+self.others[q_selected])].getMaxAction()
        else:
            self.action = self.q.getMaxAction(self.state)

        return