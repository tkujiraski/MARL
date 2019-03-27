import csv
import numpy as np

class CountAugmentedStates:
    def __init__(self, filenames):
        for filename in filenames:
            col = sum(1 for line in open(filename)) - 1
            num = np.zeros(col)
            with open(filename,'r') as f:
                reader = csv.reader(f)
                next(reader)
                i = 0
                for data in reader:
                    num[i] = len(data)
                    i=i+1
                print(filename+": num of states = {}, std dev = {}".format(num.mean(axis=0),num.std(axis=0, ddof=1)))

if __name__ == '__main__':
    agents = ['CQLearner','GCQLearner','PCQLearner2','GPCQLearner','RGPCQLearner',"GPCQwULearner","CQwULearner"]
    mazes = ['Tunnel2Goal','ISR','CIT','CMU','TunnelToGoal3']
    for agent in agents:
        for maze in mazes:
            filename = "log/e0.8/maze/{}{}e0.1g1.0wall-1ep10000trial50_aug.csv".format(agent,maze)
            a = CountAugmentedStates([filename])