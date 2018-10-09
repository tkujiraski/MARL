from Agent import *
from scipy import stats

class CQLearner(Agent):
    @classmethod
    def mixed(cls):
        return False # Mixedポリシーではない。オーバーライド不要だが明示しておく

    def __init__(self, id, nstate, naction, params, env):
        super().__init__(id, nstate, naction, params, env)
        self.N = params['window_size']
        self.p_th = params['p_threshold']

        # 単独で学習したデータ
        self.ER = self._readER(params['ER'+str(id)])

        # 拡張された状態に関する変数
        self.s_vec = {} # 拡張された状態すべてを保持。s_kをキーとするs_lの配列を要素とする辞書。この実装はN=2までしか使えない
        self.Qaug = {} # 拡張された状態、行動についての情報(AugumetedQ)の辞書。キーはs_k,s_a
        self.Waug = {} # 拡張された
        self.Waug_count = {} #
        self.others = []
        self.update = False

        # 自分自身の状態に関する報酬の統計値
        self.W = np.zeros(self.nstate+[self.naction, self.N]) # 状態、行動ごとの報酬の履歴(Sliding Window)
        self.W_count = np.zeros(self.nstate+[self.naction], dtype=int) # Wをサンプリングした回数

    def selectAct(self):
        # ここで実装するということは、単独で判断ができるということ
        # CQ-Learningの場合、単独では行動ポリシーの学習が収束している想定なので、探索はしない
        # if s_k(t) is part of a augmented state s_k_vec and the information of s_k_vec is present in the system state s(t)
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
                if rnd < self.eps:
                    # ランダムな行動選択
                    self.action = random.randint(0, self.naction - 1)
                else:
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
            if rnd < self.eps:
                # ランダムな行動選択
                self.action = random.randint(0, self.naction - 1)
            else:
                self.action = self.q.getMaxAction(self.state)

        return

    # 行動した後に呼ばれるので過去の状態を使う
    def sparse_interaction(self):
        self.update = False # デフォルトでは行動は学習済みのQ値に従う
        # Store <s_k(t), a_k(t), r_k(t)> in W_k(s_k,a_k)
        tp = tuple(self.old_s+[self.action])
        self.W[tuple(self.old_s+[self.action,self.W_count[tp] % self.N])] = self.r
        self.W_count[tp] += 1

        # if Student t-test.. 99% certainty
        if self.W_count[tp] >= self.N:
            W_k = self.W[tp]
        else:
            W_k = self.W[tp][:self.W_count[tp]]
        ER_k = self.ER[tp]
        t, p = stats.ttest_1samp(W_k,ER_k)
        if len(W_k) == self.N and p<self.p_th:
            # Store <s_k(t),a_k(t),s_l(t),r_k(t)> in W_k(s_k,a_k,s_l) for all other agents l
            others = self.env.getOthersOldState(self.id)
            for other in others:
                tpaug = tuple(self.old_s+other+[self.action])
                if not tpaug in self.Waug:
                    self.Waug[tpaug] = np.zeros(self.N)
                    self.Waug_count[tpaug] = 0
                self.Waug[tpaug][self.Waug_count[tpaug] % self.N] = self.r
                self.Waug_count[tpaug] += 1
            # for all extra state information s_i about another agent l present in s(t) do
            for other in others:
                # アルゴリズムには無いが、毎回検定するのは無駄なので、すでに(s_k, other)がs_vecにあれば検定しない
                if tuple(self.old_s) in self.s_vec and other in self.s_vec[tuple(self.old_s)]:
                    pass
                else:
                    # if Student t-test rejects h0: W_k(s_k,a_k,s_l) ...
                    if self.Waug_count[tpaug] >= self.N:
                        Waug_k = self.Waug[tpaug]
                    else:
                        Waug_k = self.Waug[tpaug][0:self.Waug_count[tpaug]]
                    t, p = stats.ttest_1samp(Waug_k, ER_k)
                    if len(Waug_k) == self.N and p<self.p_th:
                        # augment s_k with s_l to s_vec and add it to Q_k^aug
                        if not tuple(self.old_s) in self.s_vec:
                            self.s_vec[tuple(self.old_s)] = []
                        self.s_vec[tuple(self.old_s)].append(other)
                        # Q値は学習済みのもので初期化
                        self.Qaug[tuple(self.old_s+other)] = AugmentedQ(self.naction, self.q.getQvector(self.old_s))
                        print("Augmented({0},{1}) num={2}".format(self.old_s, other, len(self.Qaug)))

        # if s_k(t) is not part of any s_vec or the information of s_vec is not in s then
        if tuple(self.old_s) in self.s_vec and len(self.others)>0:
            self.update = True

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
        return

    def _readER(self,filename):
        return np.load(filename+".npy")

    def get_augmented_states(self):
        return self.Qaug

class AugmentedQ():
    # 記録構造は、アクセス方法を見てから決める
    def __init__(self,naction,initQ):
        self.naction = naction
        self.q = initQ.copy() # 初期値としてQ(s_k, a)をコピーする
        return

    def getMaxAction(self):
        return self.q.argmax()

    def getMaxQ(self):
        return self.q.max()














