from MultiPursuit import *
from DrawAveStepsGrapth import *
import csv
import time
import numpy as np
from datetime import datetime

# 学習方法
from Agent import *
from JSQLearner import *
from JSAQLearner import *
from CQLearner import *
from PCQLearner import *
from PCQLearner2 import *
from GCQLearner import *
from GPCQLearner import *
from RGPCQLearner import *
from GPCQwULearner import *
from CQwULearner import *

if __name__ == '__main__':
    params = {
        'num_of_agents':2,
        'size':7,
        'erate':0.3,
        'walk':-1,
        'wall':-1,
        'collision':-10,
        'touch':0,
        'goal':0,
        'targetMove':False,
        'startPos':[],
        'eps':0.1,
        'gamma':1.0,
        'alpha':0.1,
        'maxEpisodes':10000,
        'maxSteps':10000,
        'window_size': 20,
        'p_threshold': 0.01}

    allignment = [
        [[0, 2], [0, 4], [4, 3]],
        [[1, 3], [2, 3], [5, 3]],
        [[4, 1], [5, 2], [2, 4]],
        [[0, 5], [6, 5], [3, 3]],
        [[3, 0], [4, 0], [3, 4]],
        [[0, 3], [1, 3], [2, 3], [5, 3]],
        [[4, 1], [5, 2], [2, 0], [2, 4]],
        [[0, 5], [6, 5], [3, 0], [3, 3]],
        [[2, 0], [4, 0], [6, 0]],
        [[1, 3], [2, 3], [3, 3], [5, 3]]
    ]

    noa = [2, 2, 2, 2, 2, 3, 3, 3, 2, 3]




    startPos = 8
    params['num_of_agents'] = noa[startPos]
    params['startPos'] = allignment[startPos]
    for i in range(noa[startPos]):
        # 事前学習内容は共有する
        params['init_qvalue'+str(i)] = 'prelearning/goal0/Pursuit' + str(params['size']) + '_e0.8g1.0_qvalue_ep10000'
        params['ER'+str(i)] = 'prelearning/goal0/Pursuit' + str(params['size']) + '_e0.8g1.0_ER_ep10000'

    trial = 50
    np.random.seed(seed=1)

    #learner = Agent
    #learner = JSQLearner
    #learner = JSAQLearner
    #learner = CQLearner
    #learner = GCQLearner
    #learner = PCQLearner2
    #learner = GPCQLearner
    #learner = RGPCQLearner
    #learner = GPCQwULearner
    learner = CQwULearner

    filename = 'log/'+learner.__name__+'Pursuit'+str(params['size'])+'start{}e{}g{}wall{}ep{}trial{}.csv'.format(startPos,params['eps'],params['gamma'],params['wall'],params['maxEpisodes'],trial)
    filename_aug = 'log/'+learner.__name__+'Pursuit'+str(params['size'])+'start{}e{}g{}wall{}ep{}trial{}_aug.csv'.format(startPos,params['eps'],params['gamma'],params['wall'],params['maxEpisodes'],trial)
    with open(filename, 'w') as f:
        with open(filename_aug, 'w') as f_aug:
            writer = csv.writer(f, lineterminator='\n')
            writer_aug = csv.writer(f_aug, lineterminator='\n')
            header = [learner.__name__, 'Pursuit'+str(params['size'])]
            header += [params]
            row = header
            writer.writerow(row)
            writer_aug.writerow(row)
            avetime = 0
            for i in range(trial):
                start = time.time()
                m = MultiPursuit(learner, params)
                m.multi_q.learn()
                elapsed_time = time.time() - start
                print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
                avetime += elapsed_time
                writer.writerow(m.multi_q.stepsForGoal)
                # エピソードを全て終えた後の拡張状態
                writer_aug.writerow(m.agents[0].get_augmented_states()) # Agent/JSQ/JSAQでは実装されていないのでエラーになるはず
            avetime /= trial
            print("average time:{0}".format(avetime)+"[sec]")
            print(header)
    """# Greedyに動かして経路を出力. ←　意味が無い
    m.multi_q.replay()
    if learner.joint():
        print(m.route[0],m.agents[0].get_log())
    else:
        for i in range(m.num_of_agents):
            print(i,m.route[i],m.agents[i].get_log())"""

    drg = DrawAveStepsGraph([filename],[learner.__name__],100,params['maxEpisodes'])