from MultiAgentRL import MultiAgentRL
from DrawAveStepsGrapth import *
from Agent import *
from JSQLearner import *
from JSAQLearner import *
from CQLearner import *
from PCQLearner import *
from PCQLearner2 import *
from GCQLearner import *
from GPCQLearner import *
from Tunnel2Goal import *
import sys
import numpy as np
import time

class MultiMaze():
    def __init__(self, learner, params, maze):
        self.learner = learner
        self.params = params
        self.num_of_agents = maze.num_of_agents

        # Mazeならではの処理
        self.mv = [[0, -1],[1, 0],[0, 1],[-1, 0]] # 0:left 1:down 2:right 3:up
        self.walk = params['walk']
        self.wall = params['wall']
        self.collision = params['collision']
        self.maze = maze
        #self.goal = maze.goal
        #self.start = maze.start
        self.yx2state = {} # 座標から状態番号への変換
        # 迷路のチェック
        self.mazestate = 0
        for y in range(len(maze.map)):
            for x in range(len(maze.map[0])):
                if maze.map[y,x,0] == 0:
                    self.yx2state[(y,x)] = self.mazestate
                    self.mazestate += 1
        if maze.state != self.mazestate:
            print("迷路の状態数が一致しません"+str(maze.state)+'/'+str(self.mazestate))
            sys.exit(1)
        self.agentPosition = {}
        self.agentOldPosition = {}

        # どの学習系を使うかで変わる設定
        #  エージェントが把握する基本的な状態数はmixed policyかどうかで変わる
        self.nstate = [self.mazestate] # 自分の位置。ゴール位置は動かないので状態にする必要がない
        if learner.mixed():
            for i in range(self.num_of_agents-1):
                self.nstate = self.nstate + [self.mazestate]

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
        # 環境の初期化。
        for i in range(self.num_of_agents):
            self.agentPosition[i] = self.maze.start[i]
            self.agents[i].route = [self.maze.start[i]]
        # 初期状態を表現
        self.state = self._make_state_expression()
        self.isGoal = False

    def env_update(self, agents):
        # 全てのエージェントの行動が選択されているので、それに応じて状態を更新して、各エージェントの報酬を決める
        # Mero2009でゲームの設定を調べる！

        # 衝突時の処理をどうするか？
        # 移動先が一緒の場合、衝突として両方が元の位置に戻るとすると、戻った位置に移動しようと
        # していたエージェントと衝突する。順番に処理してすべての干渉を考慮できるか？
        # 例えば狭い道を４つのエージェントが列をなしている場合、全てのエージェントが左を選択すれば、
        # 隊列は左に行ける。しかし一番左のロボットが左以外を選んで、残りが左を選んだ場合は、結局
        # どのロボットも動けない。干渉があった場合は、干渉の影響を伝搬させる必要がある。
        # 干渉しているものを先に処理して、その処理によって干渉が生じたら干渉のキューに入れる。

        # 1つと2つの場合限定で実装してみる
        if self.num_of_agents == 1:
            # jointやmixedであることはありえないとして実装
            next_pos, penalty = self._move(0,self.agentPosition[0], self.agents[0].action)
            self.agentOldPosition[0] = self.agentPosition[0].copy()
            self.agentPosition[0] = next_pos
            if next_pos == self.maze.goal[0]:
                self.isGoal = True
                self.agents[0].r = self.goal
            else:
                self.agents[0].r = penalty
        else:
            # N>=2は共通とした
            # ①確定リスト（場所、ID）、未移動リスト（場所、ID）、行きたい場所リスト（場所、ID)
            next_pos = {}  # エージェントiが動きたい位置
            not_fixed_pos ={} # 確定していないエージェントの位置
            fixed_pos = {} # 確定したエージェントの位置

            # ②まずは行先を調べてゴールに到着しているものは確定リストに、そうでないものは未移動リストと行きたい場所リストに登録
            for i in range(self.num_of_agents):
                self.agentOldPosition[i] = self.agentPosition[i].copy()
                if self.learner.joint():
                    # Joint Learnerの場合、agents[0].actionに全てのエージェントのアクションの値が含まれている
                    next_pos1, penalty1 = self._move(i, self.agentPosition[i], self._getAction(self.agents[0].action, len(self.mv), i))
                else:
                    next_pos1, penalty1 = self._move(i, self.agentPosition[i], self.agents[i].action)
                self.agents[i].r = penalty1
                # ゴールに到達していれば確定
                if next_pos1 == self.maze.goal[i]:
                    fixed_pos[i] = next_pos1
                # 到達していなければnext_posとnot_fixed_posに登録
                else:
                    next_pos[i] = next_pos1
                    not_fixed_pos[i] = self.agentOldPosition[i]

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
                            self.agents[id2].r = self.collision
                    if len(same) > 1:
                        fixed_pos[id] = not_fixed_pos[id]
                        del not_fixed_pos[id]
                        del next_pos[id]
                        self.agents[id].r = self.collision

            #　⑤場所を交換しようとしているものを確定
            #　　　行きたい場所が未移動リストにあり、その未移動リストのIDの行きたい場所が自分の場所なら、両方の
            #　　　IDを行きたい場所から削除し、未移動リストを確定リストに移動
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
                            self.agents[id].r = self.collision
                            self.agents[id2].r = self.collision
                            break # 一つ見つかればもう見つからない

            #　⑥動けるエージェントと動けないエージェントを確定を未移動リストが無くなるまで繰り返す
            #　　　行きたい場所リストに対して、行先が確定リストになく、未移動リストにもなければ、
            #　　　行きたい場所リストから削除し、確定リストに追加。同じIDの未移動リストからも削除
            #　　　行先が確定リストにあれば、未移動リストから確定リストに移動させて、行きたい場所リストを削除
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
                        self.agents[id].r = self.collision
                if len(not_fixed_pos)==0:
                    break

            # ⑦確定した場所に動かす
            self.isGoal = True
            for i in range(self.num_of_agents):
                self.agentPosition[i] = fixed_pos[i]
                self.agents[i].route.append(fixed_pos[i])
                if self.agentPosition[i] != self.maze.goal[i]:
                    self.isGoal = False

        if self.learner.joint():
            self.agents[0].earned_reward += self.agents[0].r
        else:
            for i in range(self.num_of_agents):
                self.agents[i].earned_reward += self.agents[i].r

        # 次の状態をセットする
        self.state = self._make_state_expression()

    def observe(self, agent):
        if self.learner.mixed():
            return self.state.copy()
        else:
            return [self.yx2state[tuple(self.agentPosition[agent.id])]]

    def check_goal(self):
        return self.isGoal

    def getOthersState(self, id):
        # id以外のエージェントの状態のリストを返す。状態は１次元でもリストで定義する
        ret = []
        for i in range(self.num_of_agents):
            if i != id:
                ret.append([self.yx2state[tuple(self.agentPosition[i])]])
        return ret

    def getNewState(self, old_state):
        # 1つの前の状態を入力として、現在の状態を返す
        # 同じ場所には１つのエージェントしかいない前提
        for i in range(self.num_of_agents):
            if old_state == self.yx2state[tuple(self.agentOldPosition[i])]:
                return self.yx2state[tuple(self.agentPosition[i])]
        return -1 # ここにはこないはず。来たら例外になるはず

    def getOthersOldState(self, id):
        # id以外のエージェントの１つ前の状態のリストを返す。状態は１次元でもリストで定義する
        ret = []
        for i in range(self.num_of_agents):
            if i != id:
                ret.append([self.yx2state[tuple(self.agentOldPosition[i])]])
        return ret

    def _make_state_expression(self):
        state = []
        for i in range(self.num_of_agents):
            state += [self.yx2state[tuple(self.agentPosition[i])]]
        return state

    def _move(self,i,pos,action):
        # 特定の位置で特定の行動をした場合の、他のエージェントを考慮しない移動位置を返す
        # 実際に移動はしない
        # ゴールに辿り着いたエージェントは動かない
        # リストを返しているがタプルの方が良いのでは？
        if pos == self.maze.goal[i]:
            penalty = 0 # 先についたエージェントに報酬を与えなくて良いのか？与え続けるのはおかしい気がする
            newpos = pos.copy()
        elif self.maze.map[pos[0],pos[1],action+1] == 0:
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
        a = action // pow(n_act,n_agent-id)
        n_act_id = a % n_act
        return n_act_id

    def _in(self,pos,dict):
        for id in set(dict):
            if pos == dict[id]:
                return True
        return False

if __name__ == '__main__':
    params = {
        'walk': -1,
        'wall': -1,
        'collision': -10,
        'goal': 10,
        'eps': 0.1,
        'gamma': 1.0,
        'alpha': 0.1,
        'maxEpisodes': 200000,
        'maxSteps': 300000,
        'window_size': 20,
        'p_threshold': 0.01}

    maze = Tunnel2Goal
    params['init_qvalue0'] = 'result/maze/Tunnel2Goal_qvalue0_e0.3g1.0ep200000'
    params['init_qvalue1'] = 'result/maze/Tunnel2Goal_qvalue1_e0.3g1.0ep200000'
    params['ER0'] = 'result/maze/Tunnel2Goal_ER0_e0.3g1.0ep200000'
    params['ER1'] = 'result/maze/Tunnel2Goal_ER1_e0.3g1.0ep200000'

    start = time.time()
    m = MultiMaze(CQLearner,params,maze())
    m.multi_q.learn()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")




