import time
from datetime import datetime

import numpy as np

from DrawAveStepsGrapth import *
from MultiAgentRL import MultiAgentRL
from PCQLearner import *
from old.AreaFullView import *


class MultiPursuit:
    # 完全独立での学習と、マルチエージェントでのQ-Learning/PHC/WoLF-PHCでの学習をサポート。JAL(一般的)はCentralPursuitで扱う
    def __init__(self, learner, params):
        self.params = params
        # エージェント数のチェック
        self.num_of_agents = params['num_of_agents']
        if self.num_of_agents > 4:
            raise
        self.mv = {0: [0, -1], 1: [1, 0], 2: [0, 1], 3: [-1, 0], 4: [0, 0]}
        # Pursuitならではの記述
        # 0:left 1:down 2:right 3:up
        self.ysize = params['ysize'] # GridWorldのY方向の長さ
        self.xsize = params['xsize'] # GridWorldのX方向の長さ
        self.erate = params['erate'] # ターゲットが移動に失敗する確率
        self.walk = params['walk']
        self.wall = params['wall']
        self.collision = params['collision']
        self.touch = params['touch']
        self.capture = params['capture']
        self.targetMove = params['targetMove']  # ターゲットが移動するかどうか
        self.targetPos = params['targetPos'] # ターゲットの初期位置を指定するか、ランダムか
        self.area = AreaFullView(self.ysize,self.xsize)
        self.agentPosition = {}

        # どの学習系を使うかで変わる設定
        #  エージェントが把握する基本的な状態数はmixed policyかどうかで変わる
        self.nstate = [self.ysize, self.xsize, self.ysize, self.xsize]
        if learner.mixed():
            for i in range(self.num_of_agents-1):
                self.nstate = self.nstate + [self.ysize, self.xsize]
        # 共通処理
        agents = []
        for i in range(self.num_of_agents):
            agents.append(learner(i+1, self.nstate, len(self.mv), params, self)) #エージェントのidは1,2,3,・・・
        self.multi_q = MultiAgentRL(agents, params, self.env_init, self.env_update, self.observe, self.check_goal)
        self.learner = learner
        # 学習パラメータ
        self.eps = params['eps']
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.maxEpisodes = params['maxEpisodes']
        self.maxSteps = params['maxSteps']

        # 記録用
        self.target_loc = []
        self.multi_q.alabel(['左','下','右','上','-'])

    def env_init(self):
        # 環境の初期化。Areaの初期化とターゲット及びエージェントの配置
        self.area.reset_state() #ゼロクリア。配置無し
        if self.targetPos == []:
            self.target = self.area.random_position()
        else:
            self.target = self.targetPos.copy()
        self.area.setId(self.target,128)
        self.target_loc = []
        # エージェントの配置
        for agent in self.multi_q.agents:
            self.agentPosition[agent] = self.area.random_position()
            self.area.setId(self.agentPosition[agent], agent.id)
        # 初期状態を表現
        self.state = self._make_state_expression()

    def env_update(self, agents):
        # 全てのエージェントの行動が選択されているので、それに応じて状態を更新して、各エージェントの報酬を決め
        # touchの報酬があると、各エージェントの報酬が異なり完全協調ではなくなる。touchは特別な報酬を与えない？

        # ターゲットの位置を記録
        self.target_loc.append(self.target.copy())
        # ターゲットの移動
        if self.targetMove:
            ta = np.random.randint(len(self.mv))
            err = np.random.rand()
            if err > self.erate:
                self._moveTarget(ta)
        else:
            self._moveTarget(4) # 常に動かない
        ty , tx = self.area.search_target()

        # それぞれのエージェントの行動結果。ここでは番号の若いエージェントの行動が優先
        for agent in self.multi_q.agents:
            # 行動の結果positionが変わり、それぞれの報酬が決まる
            r = 0
            result = self._moveAgent(agent) # 0:移動成功 -1:壁 1,2,3,4,128:エージェントもしくはターゲットと衝突
            if result == 0:
                r += self.walk
            elif result == -1:
                r += self.wall
            elif result > 0 and result <=4:
                r += self.collision
            elif result == 128:
                r += self.walk #向こうから近づいてくれた場合も移動と同じペナルティにしている

            # ターゲットと隣接した場合の報酬
            dy = ty - self.agentPosition[agent][0]
            dx = tx - self.agentPosition[agent][1]
            if (abs(dx) == 1 and dy == 0) or (abs(dy) == 1 and dx == 0):
                r += self.touch
            agent.r = r

        # 全てのエージェントがターゲットを取り囲んだ場合の報酬
        if self._is_target_caputured():
            for agent in self.multi_q.agents:
                agent.r += self.capture # 衝突しながら捕獲してもプラスにはなる
                agent.earned_reward += agent.r

        # 次の状態をセットする
        self.state = self._make_state_expression()

    def observe(self,agent):
        # 視界の設定を行う場合はこの関数をOverride
        if self.learner.mixed():
            return self.state.copy()
        else:
            # 各エージェントはターゲットと自分しか見えない
            return [self.state[0], self.state[1], self.agentPosition[agent][0], self.agentPosition[agent][1]]

    def check_goal(self):
        # 複数のエージェントがターゲットを取り囲んだ場合終了
        if self._is_target_caputured():
            return True
        else:
            return False

    def _moveTarget(self, dir):
        # ターゲットを移動させる
        # self.targetを書き換える。
        prev_position = self.target.copy()
        self.target[0] += self.mv[dir][0]
        self.target[1] += self.mv[dir][1]
        # 一歩進んで壁にぶつかったら戻る
        if self.target[0] < 0 or self.target[0] >= self.area.maxy or self.target[1] < 0 or self.target[1] >= self.area.maxx:
            self.target = prev_position
        id = self.area.getId(self.target)
        # 他のエージェントやターゲットがいた場合移動しない
        if id != 0 and id != 128:
            self.target = prev_position
        # Areaを更新する
        self.area.setId(prev_position, 0)
        self.area.setId(self.target, 128)

    def _moveAgent(self, agent):
        # 移動先の状態を返す(0:空白　1:壁 それ以外:オブジェクトID) ->必要か考える
        # self.agentPositionを書き換える。
        # self.stateは変わらない

        ret = 0 #成功
        prev_position = self.agentPosition[agent].copy()
        self.agentPosition[agent][0] += self.mv[agent.action][0]
        self.agentPosition[agent][1] += self.mv[agent.action][1]
        # 一歩進んで壁にぶつかったら戻る
        if self.agentPosition[agent][0] < 0 or self.agentPosition[agent][0] >= self.area.maxy or self.agentPosition[agent][1] < 0 or self.agentPosition[agent][1] >= self.area.maxx:
            self.agentPosition[agent] = prev_position
            ret = -1 #壁に衝突
        id = self.area.getId(self.agentPosition[agent])
        # 他のエージェントやターゲットがいた場合移動しないで、すでにいるオブジェクトのIDを返す
        if id != 0 and id != agent.id:
            ret = id #干渉
            self.agentPosition[agent] = prev_position
        # Areaを更新する
        self.area.setId(prev_position,0)
        self.area.setId(self.agentPosition[agent], agent.id)

        return ret

    def _is_target_caputured(self):
        sum = self.area.getId([self.target[0]-1,self.target[1]])+self.area.getId([self.target[0]+1,self.target[1]])+self.area.getId([self.target[0],self.target[1]-1])+self.area.getId([self.target[0],self.target[1]+1])
        id_sum = (self.num_of_agents+1)*self.num_of_agents//2 # Hunterのidは1,2,3,4と振られ、その合計がn(n+1)/2ならすべてのエージェントが接している
        if sum == id_sum:
            return True
        else:
            return False

    def _make_state_expression(self):
        # ターゲットの位置と各エージェントの位置で状態を定義
        ty, tx = self.area.search_target()
        state = [ty, tx]
        for agent in self.multi_q.agents:
            state = state + [self.agentPosition[agent][0], self.agentPosition[agent][1]]
        return state

def print_list(s,s2,tloc,a1,a2):
    for idx, val in enumerate(s):
        print(val, s2[idx], tloc[idx], a1[idx], a2[idx])

if __name__ == '__main__':
    params = {'num_of_agents':2,
              'xsize':7,
              'ysize':7,
              'erate':0.3,
              'walk':-1,
              'wall':-3,
              'collision':-5,
              'touch':0,
              'capture':10,
              'targetMove':True,
              'targetPos':[],
              'eps':0.3,
              'gamma':0.8,
              'alpha':0.1,
              'maxEpisodes':10000,
              'maxSteps':3000}
    learner = Agent

    start = time.time()
    m = MultiPursuit(learner, params)
    m.multi_q.learn()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    filename = 'log/'+__file__.split('/')[-1]+datetime.now().strftime("%Y%m%d_%H%M%S")+'.csv'
    m.save_learning_curve(filename)

    drg = DrawAveStepsGraph([filename], 100, params['maxEpisodes'])
