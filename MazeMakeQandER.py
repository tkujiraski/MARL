from ERLearner import *
from MultiMaze import *
from Tunnel2Goal import *
from ISR import *
from CIT import *
from CMU import *
from TunnelToGoal3 import *
from TunnelToGoal4 import *

# シングルで学習した場合のQ値と、各状態・行動に対する平均即時報酬のデータを作成する
if __name__ == '__main__':
    params = {
        'walk': -1,
        'wall': -1,
        'collision': -10,
        'goal': 10,
        'eps': 0.3,
        'gamma': 1.0,
        'alpha': 0.1,
        'maxEpisodes': 200000,
        'maxSteps': 30000,
        'window_size':20}

    learner = ERLearner
    #maze = Tunnel2Goal()
    #maze = ISR()
    #maze = CIT()
    #maze = CMU()
    #maze = TunnelToGoal3()
    maze = TunnelToGoal4()

    # 課題を１エージェント版に書き換えて実行
    num_of_agents = maze.num_of_agents
    maze.num_of_agents = 1
    starts = maze.start.copy()
    goals = maze.goal.copy()

    for i in range(num_of_agents):
        maze.start = [[starts[i][0], starts[i][1]]]
        maze.goal = [[goals[i][0], goals[i][1]]]

        filename = 'log/'+__file__.split('/')[-1]+datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(filename+'.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            header = [learner.__name__, maze.__class__.__name__]
            header += [params]
            row = header
            writer.writerow(row)

            start = time.time()

            m = MultiMaze(learner, params, maze)
            m.multi_q.learn()
            elapsed_time = time.time() - start
            print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

            writer.writerow(m.multi_q.stepsForGoal)

            # エージェントごとにQテーブルと、即時報酬の平均値を記録
            m.multi_q.agents[0].q.saveQvalue(filename+'_qvalue'+str(i))
            m.multi_q.agents[0].saveER(filename+'_ER'+str(i))

        drg = DrawAveStepsGraph([filename+'.csv'],['ER'],100,params['maxEpisodes'])