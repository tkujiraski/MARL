from MultiMaze import *
from DrawAveStepsGrapth import *
import csv
import time
import numpy as np
from datetime import datetime
from Tunnel2Goal import *
from ISR import *
from CIT import *
from CMU import *
from TunnelToGoal3 import *
from TunnelToGoal4 import *

if __name__ == '__main__':
    params = {
        'walk': -1,
        'wall': -1,
        'collision': -10,
        'goal': 0,
        'eps': 0.1,
        'gamma': 1.0,
        'alpha': 0.1,
        'maxEpisodes': 10000,
        'maxSteps': 1000,
        'window_size': 20,
        'p_threshold': 0.01}

    #maze = Tunnel2Goal
    maze = ISR
    #maze = CIT
    #maze = CMU
    #maze = TunnelToGoal3
    #maze = TunnelToGoal4

    maze_instance = maze()

    trial = 10
    np.random.seed(seed=1)

    #learner = Agent
    #learner = JSQLearner
    #learner = JSAQLearner
    #learner = CQLearner
    #learner = GCQLearner
    #learner = PCQLearner2
    learner = GPCQLearner

    for ep in range(9):
        episode = (ep+1)*1000
        # 事前学習ファイルの読み込み
        for i in range(maze_instance.num_of_agents):
            params['init_qvalue' + str(i)] = 'prelearning/goal0/'+maze.__name__+'_e0.3g1.0_id'+str(i)+'_qvalue_ep' + str(episode)
            params['ER' + str(i)] = 'prelearning/goal0/'+maze.__name__+'_e0.3g1.0_id'+str(i)+'_ER_ep' + str(episode)

        #filename = 'log/'+__file__.split('/')[-1]+datetime.now().strftime("%Y%m%d_%H%M%S")+'.csv'
        filename = 'log/'+learner.__name__+maze.__name__+'e{}g{}wall{}step{}pre{}goal{}trial{}.csv'.format(params['eps'],params['gamma'],params['wall'],params['maxSteps'],episode,params['goal'],trial)
        filename_aug = 'log/'+learner.__name__+maze.__name__+'e{}g{}wall{}step{}pre{}goal{}trial{}_aug.csv'.format(params['eps'],params['gamma'],params['wall'],params['maxSteps'],episode,params['goal'],trial)
        with open(filename, 'w') as f:
            with open(filename_aug, 'w') as f_aug:
                writer = csv.writer(f, lineterminator='\n')
                writer_aug = csv.writer(f_aug, lineterminator='\n')
                header = [learner.__name__, maze.__name__]
                header += [params]
                row = header
                writer.writerow(row)
                writer_aug.writerow(row)
                avetime = 0
                for i in range(trial):
                    start = time.time()
                    m = MultiMaze(learner, params, maze_instance)
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
        # Greedyに動かして経路を出力
        m.multi_q.replay()
        for i in range(m.num_of_agents):
            print(i,m.route[i],m.agents[i].get_log())

        #drg = DrawAveStepsGraph([filename],[learner.__name__],100,params['maxEpisodes'])