from ERLearner import *
from MultiPursuit import *

# シングルで学習した場合のQ値と、各状態・行動に対する平均即時報酬のデータを作成する
if __name__ == '__main__':
    params = {
        'num_of_agents': 1,
        'size': 7,
        'erate': 0.3,
        'walk': -1,
        'wall': -1,
        'collision': -10,
        'touch': 0,
        'goal': 0,
        'targetMove': False,
        'startPos': [],
        'eps': 0.8,
        'gamma': 1.0,
        'alpha': 0.1,
        'maxEpisodes': 10000,
        'maxSteps': 10000,
        'window_size': 20,
        'p_threshold': 0.01}

    learner = ERLearner

    # 課題を１エージェント版に書き換えて実行

    filename = 'prelearning/goal'+str(params['goal'])+'/Pursuit'+str(params['size'])+'_e'+str(params['eps'])+'g'+str(params['gamma'])
    with open(filename+'.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        header = [learner.__name__]
        header += [params]
        row = header
        writer.writerow(row)

        start = time.time()

        m = MultiPursuit(learner, params)
        m.multi_q.agents[0].setFilename(filename)
        m.multi_q.learn()
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        writer.writerow(m.multi_q.stepsForGoal)

        drg = DrawAveStepsGraph([filename+'.csv'],['ER'],100,params['maxEpisodes'])