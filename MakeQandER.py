import csv

from DrawAveStepsGrapth import *
from ERLearner import *
from old.MultiPursuit import *

# シングルで学習した場合のQ値と、各状態・行動に対する平均即時報酬のデータを作成する
if __name__ == '__main__':
    params = {'num_of_agents':1,
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
              'maxEpisodes':100000,
              'maxSteps':3000,
              'window_size':20}
    learner = ERLearner

    filename = 'log/'+__file__.split('/')[-1]+datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(filename+'.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        header = [learner.__name__]
        header += [params]
        row = header
        writer.writerow(row)

        start = time.time()

        m = MultiPursuit(learner, params)
        m.multi_q.learn()
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        writer.writerow(m.multi_q.stepsForGoal)

        # エージェントごとにQテーブルと、即時報酬の平均値を記録
        for i in range(params['num_of_agents']):
            m.multi_q.agents[i].q.saveQvalue(filename+'_qvalue'+str(i))
            m.multi_q.agents[i].saveER(filename+'_ER'+str(i))

    drg = DrawAveStepsGraph([filename+'.csv'],100,params['maxEpisodes'])