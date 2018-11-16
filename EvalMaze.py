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
        'goal': 10,
        'eps': 0.1,
        'gamma': 1.0,
        'alpha': 0.1,
        'maxEpisodes': 10000,
        'maxSteps': 100000,
        'window_size': 20,
        'p_threshold': 0.01}

    # 迷路を変えるときは下記のparamsもコメントアウトを切り替える

    maze = Tunnel2Goal
    params['init_qvalue0'] = 'result/maze/Tunnel2Goal_qvalue0_e0.3g1.0ep200000'
    params['init_qvalue1'] = 'result/maze/Tunnel2Goal_qvalue1_e0.3g1.0ep200000'
    params['ER0'] = 'result/maze/Tunnel2Goal_ER0_e0.3g1.0ep200000'
    params['ER1'] = 'result/maze/Tunnel2Goal_ER1_e0.3g1.0ep200000'

    """maze = ISR
    params['init_qvalue0'] = 'result/maze/ISR_qvalue0_e0.3g1.0ep200000'
    params['init_qvalue1'] = 'result/maze/ISR_qvalue1_e0.3g1.0ep200000'
    params['ER0'] = 'result/maze/ISR_ER0_e0.3g1.0ep200000'
    params['ER1'] = 'result/maze/ISR_ER1_e0.3g1.0ep200000'"""

    """maze = CIT
    params['init_qvalue0'] = 'result/maze/CIT_qvalue0_e0.3g1.0ep200000'
    params['init_qvalue1'] = 'result/maze/CIT_qvalue1_e0.3g1.0ep200000'
    params['ER0'] = 'result/maze/CIT_ER0_e0.3g1.0ep200000'
    params['ER1'] = 'result/maze/CIT_ER1_e0.3g1.0ep200000'"""

    """maze = CMU
    params['init_qvalue0'] = 'result/maze/CMU_qvalue0_e0.3g1.0ep200000'
    params['init_qvalue1'] = 'result/maze/CMU_qvalue1_e0.3g1.0ep200000'
    params['ER0'] = 'result/maze/CMU_ER0_e0.3g1.0ep200000'
    params['ER1'] = 'result/maze/CMU_ER1_e0.3g1.0ep200000'"""

    """maze = TunnelToGoal3
    params['init_qvalue0'] = 'result/maze/TunnelToGoal3_qvalue0_e0.3g1.0ep200000'
    params['init_qvalue1'] = 'result/maze/TunnelToGoal3_qvalue1_e0.3g1.0ep200000'
    params['init_qvalue2'] = 'result/maze/TunnelToGoal3_qvalue2_e0.3g1.0ep200000'
    params['ER0'] = 'result/maze/TunnelToGoal3_ER0_e0.3g1.0ep200000'
    params['ER1'] = 'result/maze/TunnelToGoal3_ER1_e0.3g1.0ep200000'
    params['ER2'] = 'result/maze/TunnelToGoal3_ER2_e0.3g1.0ep200000'"""

    """maze = TunnelToGoal4
    params['init_qvalue0'] = 'result/maze/TunnelToGoal4_qvalue0_e0.3g1.0ep200000'
    params['init_qvalue1'] = 'result/maze/TunnelToGoal4_qvalue1_e0.3g1.0ep200000'
    params['init_qvalue2'] = 'result/maze/TunnelToGoal4_qvalue2_e0.3g1.0ep200000'
    params['init_qvalue3'] = 'result/maze/TunnelToGoal4_qvalue3_e0.3g1.0ep200000'
    params['ER0'] = 'result/maze/TunnelToGoal4_ER0_e0.3g1.0ep200000'
    params['ER1'] = 'result/maze/TunnelToGoal4_ER1_e0.3g1.0ep200000'
    params['ER2'] = 'result/maze/TunnelToGoal4_ER2_e0.3g1.0ep200000'
    params['ER3'] = 'result/maze/TunnelToGoal4_ER3_e0.3g1.0ep200000'"""

    trial = 1
    np.random.seed(seed=1)

    #learner = Agent
    #learner = JSQLearner
    #learner = JSAQLearner
    learner = CQLearner
    #learner = GCQLearner
    #learner = PCQLearner2
    #learner = GPCQLearner


    #filename = 'log/'+__file__.split('/')[-1]+datetime.now().strftime("%Y%m%d_%H%M%S")+'.csv'
    filename = 'log/'+learner.__name__+maze.__name__+'e{}g{}wall{}step{}trial{}.csv'.format(params['eps'],params['gamma'],params['wall'],params['maxSteps'],trial)
    filename_aug = 'log/'+learner.__name__+maze.__name__+'e{}g{}wall{}step{}trial{}_aug.csv'.format(params['eps'],params['gamma'],params['wall'],params['maxSteps'],trial)
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
                m = MultiMaze(learner, params,maze())
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

    drg = DrawAveStepsGraph([filename],[learner.__name__],100,params['maxEpisodes'])