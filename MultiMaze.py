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
        self.goal = maze.goal
        self.start = maze.start
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
        self.goal = False

        # 学習パラメータ
        self.eps = params['eps']
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.maxEpisodes = params['maxEpisodes']
        self.maxSteps = params['maxSteps']

    def env_init(self):
        # 環境の初期化。
        for i in range(self.num_of_agents):
            self.agentPosition[i] = self.start[i]
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
        elif self.num_of_agents == 2:
            # N=2の場合。
            # 自分から突っ込んでいっても、壁にぶつかって結果としてぶつかってもペナルティは同じとする(collisionのペナルティのみ)
            self.agentOldPosition[0] = self.agentPosition[0].copy()
            self.agentOldPosition[1] = self.agentPosition[1].copy()
            if self.learner.joint():
                next_pos0, penalty0 = self._move(0,self.agentPosition[0],self.agents[0].action // len(self.mv))
                next_pos1, penalty1 = self._move(1,self.agentPosition[1], self.agents[0].action % len(self.mv)) # 判断するのは１つだけ
            else:
                next_pos0, penalty0 = self._move(0,self.agentPosition[0],self.agents[0].action)
                next_pos1, penalty1 = self._move(1,self.agentPosition[1], self.agents[1].action)

            if next_pos0 == self.maze.goal[0] and next_pos1 == self.maze.goal[1]:
                # 両者がゴールした場合
                self.agentPosition[0] = next_pos0
                self.agentPosition[1] = next_pos1
                self.isGoal = True
                if self.learner.joint():
                    self.agents[0].r = self.goal
                else:
                    self.agents[0].r = self.goal
                    self.agents[1].r = self.goal
            else:
                if next_pos0 == next_pos1: # 同じ場所に動こうとすると元の位置に戻る(動けない)
                    penalty0 = self.collision
                    penalty1 = self.collision
                else:
                    self.agentPosition[0] = next_pos0
                    self.agentPosition[1] = next_pos1
                if self.learner.joint():
                    self.agents[0].r = (penalty0+penalty1)/2
                else:
                    self.agents[0].r = penalty0
                    self.agents[1].r = penalty1
            #print('({0},{1}),({2},{3})'.format(self.agentPosition[0][0],self.agentPosition[0][1],self.agentPosition[1][0],self.agentPosition[1][1]),end="")
        else:
            pass


        """# まずは動かしてみる
        next_pos = {} # エージェントiが動きたい位置
        collision = {} # ある位置(y,x)に動こうとしているエージェントのリスト
        fixed = {} # ある位置(y,x)に動くことが確定したエージェント
        for i in range(self.num_of_agents):
            next_pos[i] = self.agentPosition[i] + self.mv[self.agents[i].action]
            # 干渉しているエージェントを位置ごとにリストアップ
            if next_pos[i] not in collision:
                collision[tuple(next_pos[i])] = [i]
            else:
                collision[tuple(next_pos[i])].append(i)

        # 干渉しているエージェントは元に戻して確定
        for pos in collision:
            if len(pos)>1:
                # 干渉している
                for i in pos:
                    fixed[self.agentPosition[i]] = i
                    pos.remove(i)
                    del next_pos[i]

        # 干渉によって戻った場所に動こうとしていたエージェントを元に戻して確定
        for pos in next_pos: # まだ動けていないエージェントに対して
            if pos in fixed:
                fixed[]"""

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
        if pos == [1,4] and action == 1:
            pass
        # 特定の位置で特定の行動をした場合の、他のエージェントを考慮しない移動結果を返す
        # ゴールに辿り着いたエージェントは動かない
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


if __name__ == '__main__':
    params = {
        'num_of_agents':2,
        'walk': -1,
        'wall': -1,
        'collision': -20,
        'goal':10,
        'eps': 0.3,
        'gamma': 0.8,
        'alpha': 0.1,
        'maxEpisodes': 20000,
        'maxSteps': 3000,
        'window_size': 20,
        'p_threshold': 0.01
              }

    start = time.time()
    a = Tunnel2Goal()
    m = MultiMaze(JSQLearner,params,a)
    m.multi_q.learn()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")




