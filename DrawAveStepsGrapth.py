import csv
import matplotlib.pyplot as plt
import numpy as np

class DrawAveStepsGraph:
    def __init__(self,filenames,legends,avestep,maxEpisodes, start=0, title='', color=True):
        c = ['blue','red','green','black','grey','pink','purple','orange']
        ls = ['-','--',':','-.']
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
                            s += float(data[i * avestep + j]) + 1.0
                            if i == maxEpisodes // avestep - 1:
                                last100[trial,j] = float(data[i * avestep + j]) + 1.0
                        ave[trial,i] = s/avestep
                    trial += 1
            avetrial = ave.mean(axis=0)
            stdtrial = ave.std(axis=0, ddof=1)
            last100ave = avetrial[-1]
            last100std = stdtrial[-1]
            last100std_each = last100.std(ddof=1)
            drawdata.append(avetrial)
            print(filename+": ave={0}, std={1}, std_each={2}".format(last100ave,last100std,last100std_each))

        fig = plt.figure()
        plt.xlabel('episodes [x100]')
        plt.ylabel('average number of steps to goal')
        plt.title(title)

        ax = fig.add_subplot(1, 1, 1)
        for i in range(len(filenames)):
            if color == True:
                ax.plot(range(start//avestep,len(drawdata[i][start//avestep:])+start//avestep), drawdata[i][start//avestep:], c=c[i], label=legends[i])
            else:
                ax.plot(range(start // avestep, len(drawdata[i][start // avestep:]) + start // avestep),
                    drawdata[i][start // avestep:], linestyle=ls[i], color='black', label=legends[i])
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

    """drg = DrawAveStepsGraph(['result/maze/ep200000/AgentTunnel2Goale0.1g1.0wall-1step300000trial50.csv',
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
                            20000)"""

    #drg = DrawAveStepsGraph(['log/memoした/GPCQLearnerISRe0.1g1.0wall-1step300000trial50.csv'], ['GPCQ'], 100,10000)

    # 論文用の評価結果
    files = [
        "log/交叉チェック付き/AgentTunnel2Goale0.1g1.0wall-1step300000trial50",
        "log/交叉チェック付き/JSQLearnerTunnel2Goale0.1g1.0wall-1step300000trial50",
        "log/JS200000/JSAQLearnerTunnel2Goale0.1g1.0wall-1step10000trial50",  # 1万回は省略して集計だけ1万回と20万回で行った
        "log/交叉チェック付き/CQLearnerTunnel2Goale0.1g1.0wall-1step300000trial50",
        "log/GPCQ修正/GCQLearnerTunnel2Goale0.1g1.0wall-1step10000trial50",
        "log/GPCQ修正/PCQLearner2Tunnel2Goale0.1g1.0wall-1step10000trial50",
        "log/GPCQ修正/GPCQLearnerTunnel2Goale0.1g1.0wall-1step10000trial50",

        "log/交叉チェック付き/AgentISRe0.1g1.0wall-1step300000trial50",
        "log/交叉チェック付き/JSQLearnerISRe0.1g1.0wall-1step300000trial50",
        "log/JS200000/JSAQLearnerISRe0.1g1.0wall-1step10000trial50",  # 1万回は省略して集計だけ1万回と20万回で行った
        "log/交叉チェック付き/CQLearnerISRe0.1g1.0wall-1step300000trial50",
        "log/GPCQ修正/GCQLearnerISRe0.1g1.0wall-1step10000trial50",
        "log/GPCQ修正/PCQLearner2ISRe0.1g1.0wall-1step10000trial50",
        "log/GPCQ修正/GPCQLearnerISRe0.1g1.0wall-1step10000trial50",

        "log/交叉チェック付き/AgentCITe0.1g1.0wall-1step300000trial50",
        "log/交叉チェック付き/JSQLearnerCITe0.1g1.0wall-1step300000trial50",
        "log/JS200000/JSAQLearnerCITe0.1g1.0wall-1step10000trial50",  # 1万回は省略して集計だけ1万回と20万回で行った
        "log/交叉チェック付き/CQLearnerCITe0.1g1.0wall-1step300000trial50",
        "log/GPCQ修正/GCQLearnerCITe0.1g1.0wall-1step10000trial50",
        "log/GPCQ修正/PCQLearner2CITe0.1g1.0wall-1step10000trial50",
        "log/GPCQ修正/GPCQLearnerCITe0.1g1.0wall-1step10000trial50",

        "log/交叉チェック付き/AgentCMUe0.1g1.0wall-1step300000trial50",
        "log/交叉チェック付き/JSQLearnerCMUe0.1g1.0wall-1step300000trial50",
        "log/JS200000/JSAQLearnerCMUe0.1g1.0wall-1step10000trial50",  # 1万回は省略して集計だけ1万回と20万回で行った
        "log/交叉チェック付き/CQLearnerCMUe0.1g1.0wall-1step300000trial50",
        "log/交叉チェック付き/PCQLearner2CMUe0.1g1.0wall-1step300000trial50",
        "log/GPCQ修正/GCQLearnerCMUe0.1g1.0wall-1step10000trial50",
        "log/GPCQ修正/PCQLearner2CMUe0.1g1.0wall-1step10000trial50",
        "log/GPCQ修正/GPCQLearnerCMUe0.1g1.0wall-1step10000trial50",

        "log/交叉チェック付き/AgentTunnelToGoal3e0.1g1.0wall-1step300000trial50",
        "log/交叉チェック付き/JSQLearnerTunnelToGoal3e0.1g1.0wall-1step300000trial50",
        "log/JS200000/JSAQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial50",  # 1万回は省略して集計だけ1万回と20万回で行った
        "log/交叉チェック付き/CQLearnerTunnelToGoal3e0.1g1.0wall-1step300000trial50",
        "log/GPCQ修正/GCQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial50",
        "log/GPCQ修正/PCQLearner2TunnelToGoal3e0.1g1.0wall-1step10000trial50",
        "log/GPCQ修正/GPCQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial50",

        "log/JS200000/JSQLearnerTunnel2Goale0.1g1.0wall-1step10000trial50",
        "log/JS200000/JSQLearnerISRe0.1g1.0wall-1step10000trial50",
        "log/JS200000/JSQLearnerCITe0.1g1.0wall-1step10000trial50",
        "log/JS200000/JSQLearnerCMUe0.1g1.0wall-1step10000trial50",
        "log/JS200000/JSQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial50"]

    # 論文用のグラフ
    # Q-Learningの拡張手法の学習曲線。
    """drg = DrawAveStepsGraph(['log/交叉チェック付き/AgentTunnel2Goale0.1g1.0wall-1step300000trial50.csv','log/交叉チェック付き/JSQLearnerTunnel2Goale0.1g1.0wall-1step300000trial50.csv','log/JS200000/JSAQLearnerTunnel2Goale0.1g1.0wall-1step10000trial50.csv'],['independent-learning','JSQ-learning','JSAQ-learning'],100,10000, title = 'Tunnel2Goal')
    drg = DrawAveStepsGraph(['log/交叉チェック付き/AgentISRe0.1g1.0wall-1step300000trial50.csv','log/交叉チェック付き/JSQLearnerISRe0.1g1.0wall-1step300000trial50.csv','log/JS200000/JSAQLearnerISRe0.1g1.0wall-1step10000trial50.csv'],['independent-learning', 'JSQ-learning', 'JSAQ-learning'], 100, 10000, title = 'ISR')
    drg = DrawAveStepsGraph(['log/交叉チェック付き/AgentCITe0.1g1.0wall-1step300000trial50.csv','log/交叉チェック付き/JSQLearnerCITe0.1g1.0wall-1step300000trial50.csv','log/JS200000/JSAQLearnerCITe0.1g1.0wall-1step10000trial50.csv'],['independent-learning', 'JSQ-learning', 'JSAQ-learning'], 100, 10000, title = 'CIT')
    drg = DrawAveStepsGraph(['log/交叉チェック付き/AgentCMUe0.1g1.0wall-1step300000trial50.csv',
                             'log/交叉チェック付き/JSQLearnerCMUe0.1g1.0wall-1step300000trial50.csv',
                             'log/JS200000/JSAQLearnerCMUe0.1g1.0wall-1step10000trial50.csv'],
                            ['independent-learning', 'JSQ-learning', 'JSAQ-learning'], 100, 10000, title = 'CMU')"""
    """drg = DrawAveStepsGraph(['log/交叉チェック付き/AgentTunnelToGoal3e0.1g1.0wall-1step300000trial50.csv',
                             'log/交叉チェック付き/JSQLearnerTunnelToGoal3e0.1g1.0wall-1step300000trial50.csv',
                             'log/JS200000/JSAQLearnerTunnelToGoal3e0.1g1.0wall-1step1000000trial50.csv'],
                            ['independent-learning', 'JSQ-learning', 'JSAQ-learning'], 100, 10000, title = 'TunnelToGoal3')"""

    # 1万回での学習曲線を載せるか？カラーになるし。
    """drg = DrawAveStepsGraph(["log/交叉チェック付き/JSQLearnerTunnel2Goale0.1g1.0wall-1step300000trial50.csv",
                             "log/JS200000/JSAQLearnerTunnel2Goale0.1g1.0wall-1step10000trial50.csv",
                             # 1万回は省略して集計だけ1万回と20万回で行った
                             "log/交叉チェック付き/CQLearnerTunnel2Goale0.1g1.0wall-1step300000trial50.csv",
                             "log/GPCQ修正/GCQLearnerTunnel2Goale0.1g1.0wall-1step10000trial50.csv",
                             "log/GPCQ修正/PCQLearner2Tunnel2Goale0.1g1.0wall-1step10000trial50.csv",
                             "log/GPCQ修正/GPCQLearnerTunnel2Goale0.1g1.0wall-1step10000trial50.csv"],
                            ['JSQ', 'JSAQ', 'CQ', 'GCQ', 'PCQ', 'GPCQ'], 100, 10000)
    drg = DrawAveStepsGraph(["log/JS200000/JSQLearnerTunnel2Goale0.1g1.0wall-1step10000trial50.csv","log/JS200000/JSAQLearnerTunnel2Goale0.1g1.0wall-1step10000trial50.csv"],['JSQ200000','JSAQ200000'],100,200000)
    """

    """drg = DrawAveStepsGraph(["log/交叉チェック付き/JSQLearnerISRe0.1g1.0wall-1step300000trial50.csv",
                             "log/JS200000/JSAQLearnerISRe0.1g1.0wall-1step10000trial50.csv",
                             # 1万回は省略して集計だけ1万回と20万回で行った
                             "log/交叉チェック付き/CQLearnerISRe0.1g1.0wall-1step300000trial50.csv",
                             "log/GPCQ修正/GCQLearnerISRe0.1g1.0wall-1step10000trial50.csv",
                             "log/GPCQ修正/PCQLearner2ISRe0.1g1.0wall-1step10000trial50.csv",
                             "log/GPCQ修正/GPCQLearnerISRe0.1g1.0wall-1step10000trial50.csv"],
                            ['JSQ', 'JSAQ', 'CQ', 'GCQ', 'PCQ', 'GPCQ'], 100, 10000)
    drg = DrawAveStepsGraph(["log/JS200000/JSQLearnerISRe0.1g1.0wall-1step10000trial50.csv",
                             "log/JS200000/JSAQLearnerISRe0.1g1.0wall-1step10000trial50.csv"],['JSQ200000','JSAQ200000'],100,200000)"""

    """drg = DrawAveStepsGraph(["log/交叉チェック付き/JSQLearnerCITe0.1g1.0wall-1step300000trial50.csv",
                             "log/JS200000/JSAQLearnerCITe0.1g1.0wall-1step10000trial50.csv",
                             # 1万回は省略して集計だけ1万回と20万回で行った
                             "log/交叉チェック付き/CQLearnerCITe0.1g1.0wall-1step300000trial50.csv",
                             "log/GPCQ修正/GCQLearnerCITe0.1g1.0wall-1step10000trial50.csv",
                             "log/GPCQ修正/PCQLearner2CITe0.1g1.0wall-1step10000trial50.csv",
                             "log/GPCQ修正/GPCQLearnerCITe0.1g1.0wall-1step10000trial50.csv"],
                            ['JSQ', 'JSAQ', 'CQ', 'GCQ', 'PCQ', 'GPCQ'], 100, 10000,9000)
    drg = DrawAveStepsGraph(["log/JS200000/JSQLearnerCITe0.1g1.0wall-1step10000trial50.csv",
                             "log/JS200000/JSAQLearnerCITe0.1g1.0wall-1step10000trial50.csv"],
                            ['JSQ200000', 'JSAQ200000'], 100, 200000)"""

    """drg = DrawAveStepsGraph(["log/交叉チェック付き/JSQLearnerCMUe0.1g1.0wall-1step300000trial50.csv",
                             "log/JS200000/JSAQLearnerCMUe0.1g1.0wall-1step10000trial50.csv",
                             # 1万回は省略して集計だけ1万回と20万回で行った
                             "log/交叉チェック付き/CQLearnerCMUe0.1g1.0wall-1step300000trial50.csv",
                             "log/GPCQ修正/GCQLearnerCMUe0.1g1.0wall-1step10000trial50.csv",
                             "log/GPCQ修正/PCQLearner2CMUe0.1g1.0wall-1step10000trial50.csv",
                             "log/GPCQ修正/GPCQLearnerCMUe0.1g1.0wall-1step10000trial50.csv"],
                            ['JSQ', 'JSAQ', 'CQ', 'GCQ', 'PCQ', 'GPCQ'], 100, 10000)
    drg = DrawAveStepsGraph(["log/JS200000/JSQLearnerCMUe0.1g1.0wall-1step10000trial50.csv",
                             "log/JS200000/JSAQLearnerCMUe0.1g1.0wall-1step10000trial50.csv"],
                            ['JSQ200000', 'JSAQ200000'], 100, 200000)"""

    """drg = DrawAveStepsGraph(["log/交叉チェック付き/JSQLearnerTunnelToGoal3e0.1g1.0wall-1step300000trial50.csv",
                             "log/JS200000/JSAQLearnerTunnelToGoal3e0.1g1.0wall-1step1000000trial50.csv",
                             # 1万回は省略して集計だけ1万回と20万回で行った
                             "log/交叉チェック付き/CQLearnerTunnelToGoal3e0.1g1.0wall-1step300000trial50.csv",
                             "log/GPCQ修正/GCQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial50.csv",
                             "log/GPCQ修正/PCQLearner2TunnelToGoal3e0.1g1.0wall-1step10000trial50.csv",
                             "log/GPCQ修正/GPCQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial50.csv"],
                            ['JSQ', 'JSAQ', 'CQ', 'GCQ', 'PCQ', 'GPCQ'], 100, 10000)
    drg = DrawAveStepsGraph(["log/JS200000/JSQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial50.csv",
                             "log/JS200000/JSAQLearnerTunnelToGoal3e0.1g1.0wall-1step1000000trial50.csv"],
                            ['JSQ200000', 'JSAQ200000'], 100, 200000)"""

    # TunnelToGoal3でも回数が増えると勝つという話をするか？
    #drg = DrawAveStepsGraph(['log/30万回/JSQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial2.csv','log/30万回/CQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial2.csv','log/30万回/GCQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial2.csv','log/30万回/PCQLearner2TunnelToGoal3e0.1g1.0wall-1step10000trial2.csv','log/30万回/GPCQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial2.csv'],['JSQ-Learner','CQ-Learner','GCQ-Learner','PCQ-Learner','GPCQ-Learner'],100,200000)
    """drg = DrawAveStepsGraph(['log/30万回/JSQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial2.csv',
                             'log/30万回/CQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial2.csv',
                             'log/30万回/GPCQLearnerTunnelToGoal3e0.1g1.0wall-1step10000trial2.csv'],
                            ['JSQ-Learner','CQ-Learner', 'GPCQ-Learner'], 100, 300000,200000)"""

    #drg = DrawAveStepsGraph(['log/GPCQLearnerTunnel2Goale0.1g1.0wall-1step100000pre10000goal0trial10.csv','log/GPCQLearnerTunnel2Goale0.1g1.0wall-1step100000pre50000goal0trial10.csv','log/GPCQLearnerTunnel2Goale0.1g1.0wall-1step100000pre100000goal0trial10.csv','log/GPCQLearnerTunnel2Goale0.1g1.0wall-1step100000pre200000goal0trial10.csv'],['10000','50000','100000','200000'],100,10000)
    """drg = DrawAveStepsGraph(['log/GPCQLearnerTunnel2Goale0.1g1.0wall-1step100000pre200000goal0trial10.csv',
                             'log/GPCQLearnerISRe0.1g1.0wall-1step100000pre200000goal0trial10.csv',
                             'log/GPCQLearnerCITe0.1g1.0wall-1step100000pre200000goal0trial10.csv',
                             'log/GPCQLearnerCMUe0.1g1.0wall-1step100000pre200000goal0trial10.csv',
                             'log/GPCQLearnerTunnelToGoal3e0.1g1.0wall-1step100000pre200000goal0trial10.csv'],
                            ['TunnelToGoal', 'ISR', 'CIT', 'CMU', 'TunnelToGoal3'], 100, 10000)"""

    # Pursuit
    #drg = DrawAveStepsGraph(['log/AgentPursuit7start0e0.3g1.0wall-1step10000trial50.csv','log/JSQLearnerPursuit7start0e0.3g1.0wall-1step10000trial50.csv','log/JSAQLearnerPursuit7start0e0.3g1.0wall-1step10000trial50.csv'],['Agent20000','JSQ20000','JSAQ20000'],100, 20000)
    """drg = DrawAveStepsGraph(['log/RGPCQLearnerPursuit7start5e0.1g1.0wall-1step10000trial50.csv',
                             'log/RGPCQLearnerPursuit7start6e0.1g1.0wall-1step10000trial50.csv'],
                            ['6', '7'], 100, 10000)"""

    # e=0.8
    """drg = DrawAveStepsGraph(['log/e0.8/CQLearnerTunnelToGoal3e0.1g1.0wall-1step100000trial50.csv',
                             'log/e0.8/GPCQLearnerTunnelToGoal3e0.1g1.0wall-1step100000trial50.csv',
                             'log/e0.8/RGPCQLearnerTunnelToGoal3e0.1g1.0wall-1step100000trial50.csv'],
                            ['CQ-Learner', 'GPCQ-Learner', 'RGPCQ-Learner'], 100, 10000)"""
    drg = DrawAveStepsGraph(['log/JSAQLearnerISRe0.1g1.0wall-1ep20000trial50.csv'], ['JSAQ-learning'], 100, 20000)
