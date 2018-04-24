import csv
import matplotlib.pyplot as plt
import numpy as np

class DrawAveStepsGraph:
    def __init__(self,filenames,legends,avestep,maxEpisodes, start=0):
        c = ['blue','red','green','black','grey','pink','purple','orange']
        num = len(filenames)
        drawdata = []
        for filename in filenames:
            col = sum(1 for line in open(filename)) - 1
            ave = np.zeros([col,maxEpisodes//avestep])
            last100 = np.zeros([col,avestep])
            trial = 0
            with open(filename,'r') as f:
                reader = csv.reader(f)
                next(reader)
                # next(reader) Q値の行を消したため。マルチだとQ値はエージェントごと
                for data in reader:
                    rowave = []
                    for i in range(maxEpisodes // avestep):
                        s = 0
                        for j in range(avestep):
                            s += float(data[i * avestep + j])
                            if i == maxEpisodes // avestep - 1:
                                last100[trial,j] = data[i * avestep + j]
                        ave[trial,i] = s/avestep
                    trial += 1
            avetrial = ave.mean(axis=0)
            last100ave = avetrial[-1]
            last100std = last100.std(ddof=1)
            drawdata.append(avetrial)
            print(filename+": ave={0}, std={1}".format(last100ave,last100std))

        fig = plt.figure()
        plt.xlabel('Episodes[x100]')
        plt.ylabel('Steps to goals')

        ax = fig.add_subplot(1, 1, 1)
        for i in range(len(filenames)):
            ax.plot(range(len(drawdata[i][start//avestep:])), drawdata[i][start//avestep:], c=c[i], label=legends[i])
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # agent=2 Q/JSQ/JSQAの比較
    #drg = DrawAveStepsGraph(['result/Evaluation.py20180225_164023.csv','result/Evaluation.py20180225_165325.csv','result/JSAQLearner.py20180225_170107.csv'],100, 100000)
    # agent=3 Q/JSQ/JSQAの比較
    #drg = DrawAveStepsGraph(['result/Evaluation.py20180225_171923.csv','result/Evaluation.py20180225_221706.csv','result/JSAQLearner.py20180225_175222.csv'], 100, 100000)
    # Q agent=2/3の比較
    #drg = DrawAveStepsGraph(['result/Evaluation.py20180225_164023.csv','result/Evaluation.py20180225_171923.csv'],100,100000)
    # agent=2 Q/JSQ/JSQAの比較 episode=30万
    #drg = DrawAveStepsGraph(['result/Evaluation.py20180226_052428.csv','result/Evaluation.py20180226_051202.csv','result/JSAQLearner.py20180226_050730.csv'],100,300000)

    # agent=2 Q/JSQ/JSQAの比較 episode=1万 20試行平均
    #drg = DrawAveStepsGraph(['result/ペナルティ無し/Evaluation.py20180226_073827.csv','result/ペナルティ無し/Evaluation.py20180303_203049.csv','result/ペナルティ無し/JSAQLearner.py20180226_081452.csv'],100,10000)
    # agent=3 Q/JSQ/JSQAの比較 episode=1万 20試行平均
    #drg = DrawAveStepsGraph(['result/ペナルティ無し/Evaluation.py20180227_130658.csv','result/ペナルティ無し/Evaluation.py20180303_203715.csv','result/ペナルティ無し/JSAQLearner.py20180227_130804.csv'],100,10000)
    # ペナルティを外から与えられるようにする前と後と、同じ条件で比較して確認　→　ほぼ一緒
    #drg = DrawAveStepsGraph(['result/ペナルティ無し/Evaluation.py20180227_130658.csv', 'result/Agent3_col1_wall1.csv'], 100,10000)

    # 衝突-5, 壁-3のペナルティを与えた場合の結果 -> CQLearningとQLearningの差が無い。協調不要ということ。
    #drg = DrawAveStepsGraph(['result/CQ2_col5_wall3.csv','result/Agent2_col5_wall3.csv','result/JSAQ2_col5_wall3.csv'], 100, 20000,18000)

    # Maze AgentとJSQLearner
    #drg = DrawAveStepsGraph(['result/maze/AgentTunnel2Goale0.1.csv','result/maze/JSTunnel2Goale0.1.csv','result/maze/JSATunnel2Goale0.1.csv'],['Indep.','JS','JSA'],100,20000)
    #drg = DrawAveStepsGraph(['result/maze/AgentTunnel2Goale0.3.csv', 'result/maze/JSTunnel2Goale0.3.csv','result/maze/JSATunnel2Goale0.3.csv'],['Indep.','JS','JSA'], 100, 20000)
    #drg = DrawAveStepsGraph(['result/maze/AgentISRe0.1.csv', 'result/maze/JSISRe0.1.csv', 'result/maze/JSAISRe0.1.csv'],['Indep.','JS','JSA'], 100, 20000)
    #drg = DrawAveStepsGraph(['result/maze/AgentISRe0.3.csv', 'result/maze/JSISRe0.3.csv', 'result/maze/JSAISRe0.3.csv'],['Indep.','JS','JSA'], 100, 20000)
    #drg = DrawAveStepsGraph(['result/maze/AgentCITe0.1.csv', 'result/maze/JSCITe0.1.csv', 'result/maze/JSACITe0.1.csv'],['Indep.','JS','JSA'], 100, 20000)
    #drg = DrawAveStepsGraph(['result/maze/AgentCITe0.3.csv', 'result/maze/JSCITe0.3.csv', 'result/maze/JSACITe0.3.csv'],['Indep.','JS','JSA'], 100, 20000)
    #drg = DrawAveStepsGraph(['result/maze/AgentTunnel2Goale0.1.csv', 'result/maze/AgentTunnel2Goale0.1g1.0.csv'],['gammma:0.8','gamma1.0'], 100, 20000)
    #drg = DrawAveStepsGraph(['result/maze/AgentISRe0.1.csv', 'result/maze/AgentISRe0.1g1.0.csv'], 100, 20000)
    #drg = DrawAveStepsGraph(['result/maze/AgentCITe0.1.csv', 'result/maze/AgentCITe0.1g1.0.csv'], 100, 20000)
    #drg = DrawAveStepsGraph(['result/maze/AgentTunnel2Goale0.1.csv', 'result/maze/JSTunnel2Goale0.1.csv','result/maze/JSATunnel2Goale0.1.csv','result/maze/CQTunnel2Goale0.1.csv'], 100, 20000)
    #drg = DrawAveStepsGraph(['result/maze/AgentISRe0.1.csv', 'result/maze/AgentISRe0.1g1.0.csv'],100, 20000)
    #drg = DrawAveStepsGraph(['result/maze/CQISRe0.1.csv'],100,20000)
    #drg = DrawAveStepsGraph(['result/maze/AgentCITe0.1g1.0.csv'], 100, 20000)
    #drg = DrawAveStepsGraph(['result/maze/CQCITe0.1.csv'], 100, 20000)
    drg = DrawAveStepsGraph(['result/maze/ep200000/AgentTunnel2Goale0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/JSQLearnerTunnel2Goale0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/JSAQLearnerTunnel2Goale0.1g1.0wall-1step300000trial50.csv']
                            ,['Indep.','JS','JSA'], 100, 20000)
    drg = DrawAveStepsGraph(['result/maze/ep200000/AgentISRe0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/JSQLearnerISRe0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/JSAQLearnerISRe0.1g1.0wall-1step300000trial50.csv'],['Indep.','JS','JSA'], 100,
                            20000)
    drg = DrawAveStepsGraph(['result/maze/ep200000/AgentCITe0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/JSQLearnerCITe0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/JSAQLearnerCITe0.1g1.0wall-1step300000trial50.csv'],['Indep.','JS','JSA'], 100,
                            20000)
    drg = DrawAveStepsGraph(['result/maze/ep200000/CQLearnerTunnel2Goale0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/GCQLearnerTunnel2Goale0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/PCQLearner2Tunnel2Goale0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/GPCQLearnerTunnel2Goale0.1g1.0wall-1step300000trial50.csv']
                            ,['CQ','GCQ','PCQ','GPCQ'], 100,
                            20000)
    drg = DrawAveStepsGraph(['result/maze/ep200000/CQLearnerISRe0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/GCQLearnerISRe0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/PCQLearner2ISRe0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/GPCQLearnerISRe0.1g1.0wall-1step300000trial50.csv'],['CQ','GCQ','PCQ','GPCQ'], 100,
                            20000)
    drg = DrawAveStepsGraph(['result/maze/ep200000/CQLearnerCITe0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/GCQLearnerCITe0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/PCQLearner2CITe0.1g1.0wall-1step300000trial50.csv',
                             'result/maze/ep200000/GPCQLearnerCITe0.1g1.0wall-1step300000trial50.csv'],['CQ','GCQ','PCQ','GPCQ'], 100,
                            20000)