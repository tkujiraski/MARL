from Environment import *

class MultiPursuit:
    # 完全独立での学習と、マルチエージェントでのQ-Learning/PHC/WoLF-PHCでの学習をサポート。JAL(一般的)はCentralPursuitで扱う
    def __init__(self, learner, params):
        self.learner = learner  # 学習アルゴリズムの指定
        self.params = params
        self.num_of_agents = params['num_of_agents']

        # Pursuitならではの記述
        self.mv = [[0, -1], [1, 0], [0, 1], [-1, 0]]  # 0:left 1:down 2:right 3:up
        self.size = params['size'] # エリアの広さ size=5なら 5x5
        self.walk = params['walk']
        self.wall = params['wall']
        self.collision = params['collision']
        self.erate = params['erate'] # ターゲットが移動に失敗する確率
        self.touch = params['touch'] # いずれかがタッチしているときの報酬
        self.goal = params['goal'] # 全てのエージェントがタッチしたときのゴール報酬
        self.targetMove = params['targetMove']  # ターゲットが移動するかどうか
        self.startPos = params['startPos'] # 初期位置の配置のリスト

        # 移動するためのフィールドを定義
        self.map = self._create_maze(self.size)

        # どの学習系を使うかで変わる設定
        #  エージェントが把握する基本的な状態数はmixed policyかどうかで変わる
        self.nstate = [((self.size-1)*2+1)*((self.size-1)*2+1)] # ターゲットとの位置の差(dx,dy)の組合せ数
        if learner.mixed():
            for i in range(self.num_of_agents-1):
                self.nstate = self.nstate + [((self.size-1)*2+1)*((self.size-1)*2+1)]
        # 共通処理
        self.agents = []
        if learner.joint():
            self.agents.append(learner(0, self.nstate, pow(len(self.mv),self.num_of_agents), params, self))
        else:
            for i in range(self.num_of_agents):
                self.agents.append(learner(i, self.nstate, len(self.mv), params, self))
        self.multi_q = MultiAgentRL(self.agents, params, self.env_init, self.env_update, self.observe, self.check_goal)
        self.state = []

        # 学習パラメータ
        self.eps = params['eps']
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.maxEpisodes = params['maxEpisodes']
        self.maxSteps = params['maxSteps']

    def env_init(self):
        # 環境の初期化
        # ターゲットは最後のエージェントとして位置等を記録する
        self.agentPosition = {}
        self.agentOldPosition = {}
        self.route = {}

        # エージェントとターゲットの位置初期化
        if self.startPos == []:
            for i in range(self.num_of_agents+1):
                self.agentPosition[i] = self._random_position(i) # 重複しないように配置
        else:
            for i in range(self.num_of_agents+1):
                self.agentPosition[i] = self.startPos[i].copy()

        # 位置の記録開始
        for i in range(self.num_of_agents+1):
            self.route[i] = [self.agentPosition[i].copy()]

        # 各エージェントの初期化
        if self.learner.joint():
            self.agents[0].initState()
        else:
            # ターゲットはエージェントではないので+1は付けない
            for i in range(self.num_of_agents):
                self.agents[i].initState()

        # 初期状態を表現
        self.state = self._make_state_expression()

        self.isGoal = False

    def env_update(self, agents):
        if self.num_of_agents == 1:
            self.agentOldPosition[0] = self.agentPosition[0].copy()
            self.agentOldPosition[1] = self.agentPosition[1].copy() # ターゲットの位置

            # jointやmixedであることはありえないとして実装
            next_pos0, penalty0 = self._move(0,self.agentPosition[0],self.agents[0].action)
            if self.targetMove:
                self.agents[1].action = np.random.randint(len(self.mv))
                next_pos1, _ = self._move(1,self.agentPosition[1], self.agents[1].action)
            else:
                next_pos1 = self.agentPosition[1].copy()

            if next_pos0 == next_pos1 or (self.agentPosition[0] == next_pos1 and self.agentPosition[1] == next_pos0): # 交叉しても動けない
                pass
            else:
                self.agentPosition[0] = next_pos0
                self.agentPosition[1] = next_pos1

            if self._isTouch(self.agentPosition[0], self.agentPosition[1]):
                # ターゲットにタッチした場合
                self.isGoal = True
                self.agents[0].r = self.goal
            else:
                self.agents[0].r = penalty0
        else:
            # N>=1は共通とした
            # ①確定リスト（場所、ID）、未移動リスト（場所、ID）、行きたい場所リスト（場所、ID)
            next_pos = {}  # エージェントiが動きたい位置
            not_fixed_pos ={} # 確定していないエージェントの位置
            fixed_pos = {} # 確定したエージェントの位置
            penalty = {}

            # ②まずは行先を調べて未移動リストと行きたい場所リストに登録
            # エージェントの行先
            for i in range(self.num_of_agents):
                self.agentOldPosition[i] = self.agentPosition[i].copy()
                if self.learner.joint():
                    # Joint Learnerの場合、agents[0].actionに全てのエージェントのアクションの値が含まれている
                    next_pos1, penalty1 = self._move(i, self.agentPosition[i], self._getAction(self.agents[0].action, len(self.mv), self.num_of_agents, i))
                else:
                    next_pos1, penalty1 = self._move(i, self.agentPosition[i], self.agents[i].action)
                penalty[i] = penalty1
                next_pos[i] = next_pos1
                not_fixed_pos[i] = self.agentOldPosition[i]
            # ターゲットの行先
            self.agentOldPosition[self.num_of_agents] = self.agentPosition[self.num_of_agents].copy()
            if self.targetMove:
                self.agents[self.num_of_agents].action = np.random.randint(len(self.mv))
                next_pos[self.num_of_agents], _ = self._move(self.num_of_agents,self.agentPosition[self.num_of_agents], self.agents[self.num_of_agents].action)
            else:
                next_pos[self.num_of_agents] = self.agentPosition[self.num_of_agents].copy()
            not_fixed_pos[self.num_of_agents] = self.agentOldPosition[self.num_of_agents]

            if self.targetMove:
                self.agents[self.num_of_agents].action = np.random.randint(len(self.mv))
                next_pos1, _ = self._move(self.num_of_agents, self.agentPosition[self.num_of_agents], self.agents[self.num_of_agents].action)
            else:
                next_pos1 = self.agentPosition[self.num_of_agents].copy()
            next_pos[self.num_of_agents] = next_pos1
            not_fixed_pos[self.num_of_agents] = self.agentOldPosition[self.num_of_agents]

            # 　③壁への衝突チェック
            #　　未移動リストと行先リストが同じ場所であればwallということで確定リストに追加し、
            #　　未移動リストと行きたい先リストから削除。
            for id in set(not_fixed_pos): # keyのsetが返る
                l = not_fixed_pos[id]
                if l == next_pos[id]:
                    fixed_pos[id] = l
                    del not_fixed_pos[id]
                    del next_pos[id]

            #　④同じ場所に動こうとしているものを確定
            #　　　行きたい場所リストで同じ場所に動こうとしているものが複数あれば、行きたい場所リストと、未移動
            #　　　リストから削除し、確定リストに追加
            #      エージェントとターゲットの衝突も一旦ペナルティとしておいて、touchの場合は後で修正
            for id in set(next_pos):
                # 最初のsetから途中で削除されるので[]でアクセスするとエラーになるのでチェックしてから
                if next_pos.get(id):
                    l = next_pos[id]
                    same = [id]
                    for id2 in set(next_pos):
                        if id != id2 and l == next_pos[id2]:
                            same.append(id2)
                            fixed_pos[id2] = not_fixed_pos[id2]
                            del not_fixed_pos[id2]
                            del next_pos[id2]
                            penalty[id2] = self.collision
                    if len(same) > 1:
                        fixed_pos[id] = not_fixed_pos[id]
                        del not_fixed_pos[id]
                        del next_pos[id]
                        penalty[id] = self.collision

            #　⑤場所を交換しようとしているものを確定
            #　　　行きたい場所が未移動リストにあり、その未移動リストのIDの行きたい場所が自分の場所なら、両方の
            #　　　IDを行きたい場所から削除し、未移動リストを確定リストに移動
            #      エージェントとターゲットの衝突も一旦ペナルティとしておいて、touchの場合は後で修正
            for id in set(next_pos):
                if next_pos.get(id):
                    for id2 in set(not_fixed_pos):
                        if next_pos[id] == not_fixed_pos[id2] and next_pos[id2] == not_fixed_pos[id]: # 自分自身の行きたい場所が自分の場所なら③で確定されているはずなので、id!=id2
                            fixed_pos[id] = not_fixed_pos[id]
                            fixed_pos[id2] = not_fixed_pos[id2]
                            del not_fixed_pos[id]
                            del not_fixed_pos[id2]
                            del next_pos[id]
                            del next_pos[id2]
                            penalty[id] = self.collision
                            penalty[id2] = self.collision
                            break # 一つ見つかればもう見つからない

            #　⑥動けるエージェントと動けないエージェントを確定を未移動リストが無くなるまで繰り返す
            #　　　行きたい場所リストに対して、行先が確定リストになく、未移動リストにもなければ、
            #　　　行きたい場所リストから削除し、確定リストに追加。同じIDの未移動リストからも削除
            #　　　行先が確定リストにあれば、未移動リストから確定リストに移動させて、行きたい場所リストを削除
            #      エージェントとターゲットの衝突も一旦ペナルティとしておいて、touchの場合は後で修正
            while(True):
                for id in set(next_pos):
                    if (not self._in(next_pos[id],not_fixed_pos)) and (not self._in(next_pos[id],fixed_pos)):
                        # 動けること確定
                        fixed_pos[id] = next_pos[id]
                        del next_pos[id]
                        del not_fixed_pos[id]
                    elif self._in(next_pos[id],fixed_pos):
                        # 動けないこと確定
                        fixed_pos[id] = not_fixed_pos[id]
                        del next_pos[id]
                        del not_fixed_pos[id]
                        penalty[id] = self.collision
                if len(not_fixed_pos)==0:
                    break

            # ⑦確定した場所に動かす
            for i in range(self.num_of_agents+1):
                self.agentPosition[i] = fixed_pos[i]
                self.route[i].append(fixed_pos[i])

            # ペナルティの修正とゴールチェック
            self.isGoal = True
            for i in range(self.num_of_agents):
                if self._isTouch(self.agentPosition[i], self.agentPosition[self.num_of_agents]):
                    # ターゲットにタッチしている場合は、衝突した結果としてもtouchの報酬
                    penalty[i] = self.touch
                else:
                    self.isGoal = False
            if self.isGoal:
                for i in range(self.num_of_agents):
                    penalty[i] = self.goal
            if self.learner.joint():
                p = 0
                for i in range(self.num_of_agents):
                    p = p + penalty[i]
                self.agents[0].r = p / self.num_of_agents # 平均報酬
            else:
                for i in range(self.num_of_agents):
                    self.agents[i].r = penalty[i]

        # 台数に関係ない処理
        for i in range(self.num_of_agents+1):
            self.route[i].append(self.agentPosition[i])

        if self.learner.joint():
            self.agents[0].earned_reward += self.agents[0].r
        else:
            for i in range(self.num_of_agents):
                self.agents[i].earned_reward += self.agents[i].r

        # 次の状態をセットする
        self.state = self._make_state_expression()

    def observe(self,agent):
        # 視界の設定を行う場合はこの関数をOverride
        if self.learner.mixed():
            return self.state.copy()
        else:
            return self._get_state(agent.id)

    def check_goal(self):
        return self.isGoal

    # ここから下の実装は、環境に依存しないのでは？
    # 別のClassで実装されるべき
    """def getOthersState(self, id):
        # id以外のエージェントの状態のリストを返す。状態は１次元でもリストで定義する
        ret = []
        for i in range(self.num_of_agents):
            if i != id:
                ret.append(self._get_state(i))
        return ret

    def getNewState(self, old_state):
        # 1つの前の状態(mixedではない)を入力として、現在の状態(mixedではない)を返す

        # 対応するエージェントを探す
        oldpos = self._oldstate_to_oldposition(old_state)
        for i in range(self.num_of_agents):
            if self.agentOldPosition[i] == oldpos:
                break
        return self._get_state(i)

    def getOthersOldState(self, id):
        # id以外のエージェントの１つ前の状態のリストを返す。状態は１次元でもリストで定義する
        ret = []
        for i in range(self.num_of_agents):
            if i != id:
                ret.append(self._get_old_state(i))
        return ret"""

    def _get_state(self, id):
        # 各エージェントはターゲットとの差しか見えない
        dy = self.agentPosition[self.num_of_agents][0] - self.agentPosition[id][0] + self.size - 1
        dx = self.agentPosition[self.num_of_agents][1] - self.agentPosition[id][1] + self.size - 1
        return [dy * ((self.size-1)*2+1) + dx]

    def _oldstate_to_oldposition(self, state):
        state = state[0] # これが呼ばれるときはjoint stateではないのでOK?
        dy = state // ((self.size-1)*2+1)
        dx = state % ((self.size-1)*2+1)
        posy = self.agentOldPosition[self.num_of_agents][0] + self.size - 1 - dy
        posx = self.agentOldPosition[self.num_of_agents][1] + self.size - 1 - dx
        return [posy, posx]

    def _get_old_state(self, id):
        dy = self.agentOldPosition[self.num_of_agents][0] - self.agentOldPosition[id][0] + self.size - 1
        dx = self.agentOldPosition[self.num_of_agents][1] - self.agentOldPosition[id][1] + self.size - 1
        return [dy * ((self.size-1)*2+1) + dx]

    def _create_maze(self, size):
        # size x size の周囲に壁のある空間をMazeと同じ形式で作成
        maze = []
        # 上の壁
        maze.append([[0, 1, 0, 0, 1]])
        for i in range(size-2):
            maze[0].append([0, 0, 0, 0, 1])
        maze[0].append([0, 0, 0, 1, 1])
        # 途中の壁
        for j in range(size-2):
            maze.append([[0, 1, 0, 0, 0]])
            for i in range(size-2):
                maze[j+1].append([0, 0, 0, 0, 0])
            maze[j+1].append([0, 0, 0, 1, 0])
        # 下の壁
        maze.append([[0, 1, 1, 0, 0]])
        for i in range(size-2):
            maze[size-1].append([0, 0, 1, 0, 0])
        maze[size-1].append([0, 0, 1, 1, 0])
        return np.array(maze)

    def _random_position(self, id):
        # idより小さいエージェントと重複しない位置を返す
        isFind = False
        while not isFind:
            isFind = True
            y = np.random.randint(self.size)
            x = np.random.randint(self.size)

            for i in range(id):
                if y == self.agentPosition[i][0] and x == self.agentPosition[i][1]:
                    isFind = False
                    break
        return [y, x]

    def _isTouch(self, pos1, pos2):
        if abs(pos1[0]-pos2[0])+abs(pos1[1]-pos2[1]) == 1:
            return True
        else:
            return False

    def _make_state_expression(self):
        state = []
        for id in range(self.num_of_agents):
            state = state + self._get_state(id)
        return state

    def _move(self,i,pos,action):
        # 特定の位置で特定の行動をした場合の、他のエージェントを考慮しない移動位置を返す
        # 実際に移動はしない
        # リストを返しているがタプルの方が良いのでは？

        if self.map[pos[0],pos[1],action+1] == 0:
            penalty = self.walk
            newpos = [0,0]
            newpos[0] = pos[0] + self.mv[action][0]
            newpos[1] = pos[1] + self.mv[action][1]
        else:
            penalty = self.wall
            newpos = pos.copy()
        return newpos, penalty

    def _getAction(self, action, n_act, n_agent, id):
        # action = a0 * n_act^(n_agent-0) + a1 * n_act^(n_agent-1) + a_n_agent * n_act^(n_agent-n_agent)となっていることを仮定
        a = action // pow(n_act,n_agent-id-1)
        n_act_id = a % n_act
        return n_act_id

    def _in(self,pos,dict):
        for id in set(dict):
            if pos == dict[id]:
                return True
        return False

if __name__ == '__main__':
    params = {'num_of_agents':1,
              'size':7,
              'erate':0.3,
              'walk':-1,
              'wall':-1,
              'collision':-10,
              'touch':0,
              'goal':0,
              'targetMove':False,
              'startPos':[],
              'eps':0.3,
              'gamma':0.8,
              'alpha':0.1,
              'maxEpisodes':10000,
              'maxSteps':10000}
    learner = Agent

    start = time.time()
    m = MultiPursuit(learner, params)
    m.multi_q.learn()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

