import csv
import matplotlib.pyplot as plt
import numpy as np

class DrawPrelearningEffect:
    def __init__(self, mazename, title ='', color=True):
        c = ['blue','red','green','black','grey','pink','purple','orange']
        ls = ['-','--',':','-.']
        methods = ['CQ','GPCQ']
        maxEpisodes = 10000
        avestep = 100

        pre_episode = [(i + 1) * 1000 for i in range(9)] + [(i + 1) * 10000 for i in range(20)]
        drawdata = np.zeros((2,len(pre_episode)))

        for method in range(2):
            i = 0
            for ep in pre_episode:
                filename = 'log/{}Learner{}e0.1g1.0wall-1step100000pre{}goal0trial10.csv'.format(methods[method], mazename, ep)
                steps = np.loadtxt(filename, delimiter=',', skiprows=1)
                last100 = steps[:,-100]
                drawdata[method, i] = last100.mean()
                i=i+1

        # 描画
        fig = plt.figure()
        plt.xlabel('episodes')
        plt.ylabel('average number of steps to goal')
        plt.title(title)

        ax = fig.add_subplot(1, 1, 1)
        for method in range(2):
            ax.plot(pre_episode, drawdata[method], linestyle=ls[method], label=methods[method])

        plt.legend()
        plt.show()

if __name__ == '__main__':
    dr = DrawPrelearningEffect('Tunnel2Goal','TunnelToGoal')
    dr = DrawPrelearningEffect('CIT','CIT')
    dr = DrawPrelearningEffect('ISR','ISR')
    dr = DrawPrelearningEffect('CMU','CMU')
    dr = DrawPrelearningEffect('TunnelToGoal3','TunnelToGoal3')