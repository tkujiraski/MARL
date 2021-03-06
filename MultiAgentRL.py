import matplotlib.pyplot as plt

class MultiAgentRL():
    # タスクに依存しない処理を記述する。Super Agentは対象外。
    # 環境を共有する複数のエージェントが、知覚できる状態に基づいてポリシーを学習する
    # QLearning/PHC/Wolf-PHCの実装をサポート
    def __init__(self, agents, params, env_init, env_update, observe, check_goal):
        # アルゴリズムと独立した設定
        self.agents = agents
        self.maxEpisodes = params['maxEpisodes']
        self.maxSteps = params['maxSteps']
        # アルゴリズムではなく、タスクで変わる処理
        self.env_init = env_init
        self.env_update = env_update
        self.observe = observe
        self.check_goal = check_goal
        # Log用
        self.stepsForGoal = []
        self.adic = [] # 辞書から配列に変更

    def alabel(self,adic):
        self.adic = adic

    def learn(self):
        q = []
        q.append([])
        q.append([])
        for ep in range(self.maxEpisodes):
            print('#episode %d' % ep)
            # 状態sを初期化
            self.env_init()

            # 初期状態の観測
            for agent in self.agents:
                agent.state = self.observe(agent)

            # 各ステップでの行動選択・行動・Qテーブル更新
            for step in range(self.maxSteps):
                # 各エージェントごとの行動を選ぶ
                self._selectActions()

                # 行動する前に現在の状態をold_sに保存
                for agent in self.agents:
                    agent.oldold_s = agent.old_s.copy() # RGPCQで使用
                    agent.old_s = agent.state.copy()

                # 選択された行動を実行し、状態遷移と、報酬の決定を行う
                self.env_update(self.agents) # 全てのエージェントの行動が選択された後で行動を行い環境を更新

                # 各エージェントが知覚する状態を取得する
                for agent in self.agents:
                    agent.state = self.observe(agent)

                # Sparse Interaction関連の処理
                self._sparse_interaction()

                # 各エージェントのQTableやポリシーの更新
                self._update()

                # sが終端状態もしくは最大ステップに到達したらエピソードを終了
                if self.check_goal():
                    print('終了状態へ到達 %d step' % step)
                    self.stepsForGoal.append(step)
                    break
                if step == 1000:
                    print('over 1000')

                if step == self.maxSteps - 1:
                    self.stepsForGoal.append(step)
                    break

            self._end_episode(ep)

    # アルゴリズムごとにOverrideされる処理
    # デフォルトは完全独立なQ-Learning
    def _selectActions(self):
        for agent in self.agents:
            # デフォルトは、それぞれのエージェントに独立して選択させる実装
            agent.selectAct()  # 行動を選ぶ agent.actionに選択された行動が入る

    # デフォルトはそれぞれのエージェントに処理させる
    def _sparse_interaction(self):
        for agent in self.agents:
            agent.sparse_interaction()

    # デフォルトはエージェントにQ値のみ更新をさせる
    def _update(self):
        for agent in self.agents:
            agent.updateQ()

    def _end_episode(self, ep):
        for agent in self.agents:
            agent.end_episode(ep)

    def getOthersState(self, id):
        # id以外のエージェントの状態のリストを返す。状態は１次元でもリストで定義する
        ret = []
        for i in range(len(self.agents)):
            if i != id:
                ret.append(self.agents[i].state.copy())
        return ret

    def getNewState(self, old_state):
        # 1つの前の状態(mixedではない)を入力として、現在の状態(mixedではない)を返す

        # 対応するエージェントを探す
        for i in range(len(self.agents)):
            if self.agents[i].old_s == old_state:
                return self.agents[i].state
        raise Exception("対応するエージェントがない")

    def getOthersOldState(self, id):
        # id以外のエージェントの１つ前の状態のリストを返す。状態は１次元でもリストで定義する
        ret = []
        for i in range(len(self.agents)):
            if i != id:
                ret.append(self.agents[i].old_s.copy())
        return ret

    def getOthersOldOldState(self, id):
        # id以外のエージェントの２つ前の状態のリストを返す。状態は１次元でもリストで定義する
        ret = []
        for i in range(len(self.agents)):
            if i != id:
                ret.append(self.agents[i].oldold_s.copy())
        return ret

    def plot_learning_curve(self):
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(range(self.maxEpisodes), self.stepsForGoal, c='black')
        plt.show()

    def replay(self):
        # 状態sを初期化
        self.env_init()

        # greedyに行動
        for agent in self.agents:
            agent.eps = 0.0

        # 初期状態の観測
        for agent in self.agents:
            agent.state = self.observe(agent)

        # 各ステップでの行動選択・行動・Qテーブル更新
        for step in range(self.maxSteps):
            # 各エージェントごとの行動を選ぶ
            self._selectActions()

            # 行動する前に現在の状態をold_sに保存
            for agent in self.agents:
                agent.old_s = agent.state.copy()

            # 選択された行動を実行し、状態遷移と、報酬の決定を行う
            self.env_update(self.agents)  # 全てのエージェントの行動が選択された後で行動を行い環境を更新

            # 各エージェントが知覚する状態を取得する
            for agent in self.agents:
                agent.state = self.observe(agent)

            # Sparse Interaction関連の処理
            self._sparse_interaction()

            # updateはなし

            # sが終端状態もしくは最大ステップに到達したらエピソードを終了
            if self.check_goal():
                print('終了状態へ到達 %d step' % step)
                break
            if step == self.maxSteps - 1:
                break