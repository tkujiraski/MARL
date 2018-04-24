from MultiMaze import *
from DrawAveStepsGrapth import *
import csv
import time
from datetime import datetime
from Tunnel2Goal import *
from ISR import *
from CIT import *

if __name__ == '__main__':
    params = {
        'walk': -1,
        'wall': -1,
        'collision': -10,
        'goal': 10,
        'eps': 0.1,
        'gamma': 1.0,
        'alpha': 0.1,
        'maxEpisodes': 20000,
        'maxSteps': 300000,
        'window_size': 20,
        'p_threshold': 0.01}

    params['init_qvalue0'] = 'result/maze/Tunnel2Goal_qvalue0_e0.3g1.0ep200000'
    params['init_qvalue1'] = 'result/maze/Tunnel2Goal_qvalue1_e0.3g1.0ep200000'
    params['ER0'] = 'result/maze/Tunnel2Goal_ER0_e0.3g1.0ep200000'
    params['ER1'] = 'result/maze/Tunnel2Goal_ER1_e0.3g1.0ep200000'

    """params['init_qvalue0'] = 'result/maze/ISR_qvalue0_e0.3g1.0ep200000'
    params['init_qvalue1'] = 'result/maze/ISR_qvalue1_e0.3g1.0ep200000'
    params['ER0'] = 'result/maze/ISR_ER0_e0.3g1.0ep200000'
    params['ER1'] = 'result/maze/ISR_ER1_e0.3g1.0ep200000'"""

    """params['init_qvalue0'] = 'result/maze/CIT_qvalue0_e0.3g1.0ep200000'
    params['init_qvalue1'] = 'result/maze/CIT_qvalue1_e0.3g1.0ep200000'
    params['ER0'] = 'result/maze/CIT_ER0_e0.3g1.0ep200000'
    params['ER1'] = 'result/maze/CIT_ER1_e0.3g1.0ep200000'"""

    trial = 50
    maze = Tunnel2Goal
    #maze = ISR
    #maze = CIT

    #learner = Agent
    #learner = JSQLearner
    #learner = JSAQLearner
    #learner = CQLearner
    #learner = GCQLearner
    #learner = PCQLearner2
    learner = GPCQLearner


    #filename = 'log/'+__file__.split('/')[-1]+datetime.now().strftime("%Y%m%d_%H%M%S")+'.csv'
    filename = 'log/'+learner.__name__+maze.__name__+'e{}g{}wall{}step{}trial{}.csv'.format(params['eps'],params['gamma'],params['wall'],params['maxSteps'],trial)
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        header = [learner.__name__, maze.__name__]
        header += [params]
        row = header
        writer.writerow(row)
        avetime = 0
        for i in range(trial):
            start = time.time()
            m = MultiMaze(learner, params,maze())
            m.multi_q.learn()
            elapsed_time = time.time() - start
            print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
            avetime += elapsed_time
            writer.writerow(m.multi_q.stepsForGoal)
        avetime /= trial
        print("average time:{0}".format(avetime)+"[sec]")
        print(header)

    drg = DrawAveStepsGraph([filename],100,params['maxEpisodes'])