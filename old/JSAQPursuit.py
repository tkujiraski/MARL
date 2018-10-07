import QLearning
from DrawAveStepsGrapth import *
import numpy as np
from numpy.random import randint
from datetime import datetime
import csv
import time

class JSAQPursuit:
    """
    状態を共有した２～４のエージェントがランダムに動く獲物を追跡する問題

    状態：エージェントとターゲットの現在の位置(y[0-6],x[0-6])、視界のファクターは無し
    アクション：上下左右への移動と停止の５つ
    """
    def __init__(self, n, ysize, xsize, erate, eps, gamma, alpha, maxEpisodes, maxSteps, touch, capture):
        """
        Persuitの初期化
        :param erate: 獲物が移動に失敗する率
        """
        self.numOfAgents = n
        self.erate = erate
        self.dif = [[0,-1],[0,1],[-1,0],[1,0],[0,0]]
        self.ysize = ysize
        self.xsize = xsize
        self.touch = touch
        self.capture = capture
        dim = [self.ysize, self.xsize]
        for i in range(n):
            dim = dim + [self.ysize, self.xsize]
        self.ql = QLearning.QLearning(dim, pow(5,n), eps, gamma, alpha, maxEpisodes, maxSteps, self.inif, self.act, self.checkg, [touch, capture] )
        self.ql.alabel = self.mkActLabel(n) # 辞書ではなく配列に変更

    def mkActLabel(self, n):
        acts = ['左', '右', '上', '下', '-']
        if n == 1:
            return acts
        else:
            labels = self.mkActLabel(n - 1)
            newlabel = []
            for a in acts:
                for l in labels:
                    newlabel.append(a + l)
            return newlabel

    def inif(self):
        """地図上でのスタート地点を表す状態値を返す"""
        # 獲物の位置を決定
        self.target = [randint(self.ysize),randint(self.xsize)]
        # ハンターの位置を決定
        hunters = []
        for i in range(self.numOfAgents):
            while True:
                hunter = [randint(self.ysize), randint(self.xsize)]
                if hunter not in hunters and hunter != self.target:
                    hunters.append(hunter.copy())
                    break

        self.target_loc  = [self.target.copy()]
        state = []
        for s in hunters:
            state.extend(s)
        return self.target + state

    def act(self, s, a):
        """状態sでアクションaを行った場合の次の状態にsを書き換え、報酬を返す"""
        hunters = []
        ha = []
        for i in range(self.numOfAgents):
            hunters.append([s[i*2+2],s[i*2+3]]) #各ハンターの位置
            ha.append(a // pow(5,self.numOfAgents-i-1)) #各ハンターの行動
            a = a % pow(5,self.numOfAgents-i-1)
        # 獲物の行動。ハンターの位置には移動できない。実際にはハンターの位置に移動できる場合、すでに捕まっている
        ta = randint(5)
        err = np.random.rand()
        if err > self.erate:
            self.target, _ = self.move(self.target, ta, hunters)
        s[0] = self.target[0]
        s[1] = self.target[1]
        self.target_loc.append(self.target.copy())

        # ハンターの行動
        cond = [0,0,0,0]
        for i in range(self.numOfAgents):
            ng = hunters.copy()
            del ng[i]
            ng = [self.target] + ng # 獲物の移動が優先。
            hunters[i], cond[i] = self.move(hunters[i], ha[i], ng)
            # 状態更新
            s[i*2+2] = hunters[i][0]
            s[i*2+3] = hunters[i][1]

        # どちらかが隣接すればtouch, 捕獲すればcapture,それ以外は-1
        r = self.reward(hunters,self.target,cond)

        return r

    def reward(self, hunters, t,cond):
        # MultiPursuitと合わせる 通常-1, 両方隣接 10
        n1 = [t[0]-1,t[1]]
        n2 = [t[0]+1,t[1]]
        n3 = [t[0],t[1]-1]
        n4 = [t[0],t[1]+1]
        n = [n1,n2,n3,n4]
        sum = 0
        r = 0
        for i in range(self.numOfAgents):
            if hunters[i] in n:
                sum += 1
        if 0 in cond[0:self.numOfAgents]:
            r += -1
        if 1 in cond[0:self.numOfAgents]:
            r += -3
        if 2 in cond[0:self.numOfAgents]:
            r += -5
        if sum == self.numOfAgents:
            return r + self.touch*self.numOfAgents + self.capture
        elif sum > 0:
            return r + self.touch
        else:
            return r

    def move(self, loc, a, ng):
        """
        位置locから行動aを取った場合の更新された位置と移動ステータスを返す。
        範囲外への移動や、位置ngへの移動を試みると元の位置にとどまる

        移動ステータス
            0: 正常移動
            1: 壁に衝突
            2: 競合で移動できず
        """
        cond = 0
        new_y = loc[0] + self.dif[a][0]
        new_x = loc[1] + self.dif[a][1]
        if new_y < 0:
            cond = 1 #壁に衝突
            new_y = 0
        elif new_y >= self.ysize:
            cond = 1  # 壁に衝突
            new_y = self.ysize - 1
        if new_x < 0:
            cond = 1  # 壁に衝突
            new_x = 0
        elif new_x >= self.xsize:
            cond = 1  # 壁に衝突
            new_x = self.xsize - 1
        new_loc = [new_y, new_x]
        if new_loc in ng:
            cond = 2 # 競合のため移動できず
            return loc.copy(), cond
        else:
            return new_loc, cond

    def checkg(self,state):
        """stateが終端状態がどうかを判定する"""
        hunters = []
        for i in range(self.numOfAgents):
            hunters.append([state[i*2+2],state[i*2+3]]) #各ハンターの位置
        t = self.target
        n1 = [t[0]-1,t[1]]
        n2 = [t[0]+1,t[1]]
        n3 = [t[0],t[1]-1]
        n4 = [t[0],t[1]+1]
        n = [n1,n2,n3,n4]
        sum = 0
        for i in range(self.numOfAgents):
            if hunters[i] in n:
                sum += 1
        if sum == self.numOfAgents:
            return True
        else:
            return False

def print_list(s,tloc,a):
    a += ['']
    for idx, val in enumerate(s):
        print(val, tloc[idx], a[idx])

def save_movement(n, ysize, xsize, touch, capture, s, tloc, a, filename):
    a += ['']
    with open(filename,'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        row = [n, ysize, xsize, touch, capture]
        writer.writerow(row)
        for idx, val in enumerate(s):
            row = []
            for i in range(n):
                row = row + [val[i*2+2],val[i*2+3]]
            row = row + [tloc[idx][0],tloc[idx][1]]
            writer.writerow(row)

if __name__ == '__main__':
    trial = 1
    num = 3
    ysize = 7
    xsize = 7
    erate = 0.3
    eps = 0.05
    gamma = 0.8
    alpha = 0.1
    maxEpisodes = 200000
    maxSteps = 3000
    params = {"eps":eps, "gamma":gamma, "alpha":alpha, "maxEpisodes":maxEpisodes, "maxSteps":maxSteps}
    # touch = -1.0 / num
    touch = 0
    capture = 10
    filename = 'log/'+__file__.split('/')[-1]+datetime.now().strftime("%Y%m%d_%H%M%S")+'.csv'
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        header = ["JSAQLearner"]
        header += [num, ysize, xsize, erate, params]
        row = header
        writer.writerow(row)
        avetime = 0
        for i in range(trial):
            start = time.time()
            m = JSAQPursuit(num,ysize,xsize,erate,eps, gamma, alpha, maxEpisodes, maxSteps, touch, capture)
            m.ql.learn()
            elapsed_time = time.time() - start
            print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
            avetime += elapsed_time
            writer.writerow(m.ql.stepsForGoal)
        avetime /= trial
        print("average time:{0}".format(avetime)+"[sec]")

    drg = DrawAveStepsGraph([filename],100,params['maxEpisodes'])


    """for i in range(5):
        ss, acs = m.ql.replay()
        print_list(ss, m.target_loc, acs)
        save_movement(num, xsize, ysize, touch, capture, ss, m.target_loc, acs, 'log/move_'+__file__.split('/')[-1]+datetime.now().strftime("%Y%m%d_%H%M%S")+'_'+str(i)+'.csv')
        print('')"""