from GPCQLearner import *

class RGPCQLearner(GPCQLearner):
    def __init__(self, id, nstate, naction, params, env):
        super().__init__(id, nstate, naction, params, env)

    # 行動した後に呼ばれるので過去の状態を使う
    def sparse_interaction(self):
        self.update = False  # デフォルトでは行動は学習済みのQ値に従う
        # Store <s_k(t), a_k(t), r_k(t)> in W_k(s_k,a_k)
        tp = tuple(self.old_s + [self.action])
        self.W[tuple(self.old_s + [self.action, self.W_count[tp] % self.N])] = self.r
        self.W_count[tp] += 1

        # if Student t-test.. 99% certainty
        if self.W_count[tp] >= self.N:
            W_k = self.W[tp]
        else:
            W_k = self.W[tp][:self.W_count[tp]]
        ER_k = self.ER[tp]
        t, p = stats.ttest_1samp(W_k, ER_k)
        if len(W_k) == self.N and p < self.p_th:
            # Store <s_k(t),a_k(t),s_l(t),r_k(t)> in W_k(s_k,a_k,s_l) for all other agents l
            others = self.env.multi_q.getOthersOldState(self.id)
            for other in others:
                tpaug = tuple(self.old_s + other + [self.action])
                if not tpaug in self.Waug:
                    self.Waug[tpaug] = np.zeros(self.N)
                    self.Waug_count[tpaug] = 0
                self.Waug[tpaug][self.Waug_count[tpaug] % self.N] = self.r
                self.Waug_count[tpaug] += 1
            # for all extra state information s_i about another agent l present in s(t) do
            for other in others:
                tpaug = tuple(self.old_s + other + [self.action])  # 3体で上手くいっていないのはこのせい？
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
                    if len(Waug_k) == self.N and p < self.p_th:
                        # augment s_k with s_l to s_vec and add it to Q_k^aug
                        if not tuple(self.old_s) in self.s_vec:
                            self.s_vec[tuple(self.old_s)] = []
                        self.s_vec[tuple(self.old_s)].append(other)
                        # Q値は学習済みのもので初期化
                        self.Qaug[tuple(self.old_s + other)] = AugmentedQ(self.naction,
                                                                          self.q.getQvector(self.old_s))
                        print("Augmented({0},{1}) num={2}".format(self.old_s, other, len(self.Qaug)))
        # RGPCQでの変更部分
        elif self.isGreedy:
            if tuple(self.state) in self.s_vec:
                # t+1で干渉しているエージェントの状態のリストを返す
                interferences_tplus1 = []
                others = self.env.multi_q.getOthersState(self.id)  # 行動は今の状態に基づく
                for d in self.s_vec[tuple(self.state)]:
                    if d in others:
                        interferences_tplus1.append(d)
                # t-1で同じ状態だったか？
                if self.state == self.oldold_s:
                    # t-1で干渉していたエージェントの状態のリストを返す
                    interferences_t_1 = []
                    others = self.env.multi_q.getOthersOldOldState(self.id)
                    for d in self.s_vec[tuple(self.oldold_s)]:
                        if d in others:
                            interferences_t_1.append(d)
                    # リストに共通な状態があればtでの状態を拡張する
                    # 状態がリストなのでsetが使えない？
                    tobeAug = []
                    for s1 in interferences_tplus1:
                        for s2 in interferences_t_1:
                            if s1 == s2:
                                tobeAug.append(s1)
                    # tの状態に対して、t-1,t+1で干渉している状態を拡張する
                    for aug in tobeAug:
                        if tuple(self.old_s) in self.s_vec and aug in self.s_vec[tuple(self.old_s)]:
                            pass
                        else:
                            if not tuple(self.old_s) in self.s_vec:
                                self.s_vec[tuple(self.old_s)] = []
                            self.s_vec[tuple(self.old_s)].append(aug)
                            # Q値は学習済みのもので初期化
                            self.Qaug[tuple(self.old_s + aug)] = AugmentedQ(self.naction, self.q.getQvector(self.old_s))
                            print("Augmented by RGPCQ ({0},{1}) num={2}".format(self.old_s, aug, len(self.Qaug)))
        # 変更部分終わり

        # if s_k(t) is not part of any s_vec or the information of s_vec is not in s then
        if tuple(self.old_s) in self.s_vec and len(self.others) > 0:
            self.update = True