import csv

from DrawAveStepsGrapth import *
from old.MultiPursuit import *

if __name__ == '__main__':
    params = {'num_of_agents':3,
              'xsize':7,
              'ysize':7,
              'erate':0.3,
              'walk':-1,
              'wall':-3,
              #'wall':-1,
              'collision':-5,
              #'collision': -1,
              'touch':0,
              'capture':10,
              'targetMove':True,
              'targetPos':[],
              'eps':0.3,
              'gamma':0.8,
              'alpha':0.1,
              'maxEpisodes':200000,
              'maxSteps':3000,
              'window_size':20,
              'p_threshold':0.01,
              #'init_qvalue1':'result/MakeQandER.py20180304_200607_qvalue0',
              #'init_qvalue2':'result/MakeQandER.py20180304_200607_qvalue0',
              'ER1':'result/MakeQandER.py20180304_200607_ER0',
              'ER2': 'result/MakeQandER.py20180304_200607_ER0'}

    trial = 1

    learner = Agent
    #learner = JSQLearner
    #learner = CQLearner
    #learner = PCQLearner

    filename = 'log/'+__file__.split('/')[-1]+datetime.now().strftime("%Y%m%d_%H%M%S")+'.csv'
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        header = [learner.__name__]
        header += [params]
        row = header
        writer.writerow(row)
        avetime = 0
        for i in range(trial):
            start = time.time()
            # __init__(self, num_of_agents, learner, params, ysize, xsize, erate, touch, capture, targetMove=True, targetPos=[]):
            m = MultiPursuit(learner, params)
            m.multi_q.learn()
            elapsed_time = time.time() - start
            print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
            avetime += elapsed_time
            writer.writerow(m.multi_q.stepsForGoal)
        avetime /= trial
        print("average time:{0}".format(avetime)+"[sec]")

    drg = DrawAveStepsGraph([filename],100,params['maxEpisodes'])
