import csv
import numpy as np

class StateCount:
    def __init__(self, filenames):
        for filename in filenames:
            if "Agent" in filename or "JS" in filename:
                continue
            with open(filename+"_aug.csv",'r') as f:
                reader = csv.reader(f)
                next(reader)
                count = np.zeros(50)
                i = 0
                for data in reader: # 1行ずつ取り出す
                    count[i] = len(data)
                    i = i + 1
                ave = count.mean()
                std = count.std(ddof=1)
                print("{}, ave={}, std={}".format(filename,ave,std))



if __name__ == '__main__':

    files = [
        "log/交叉チェック付き/AgentTunnel2Goale0.1g1.0wall-1step300000trial50",
        "log/交叉チェック付き/JSQLearnerTunnel2Goale0.1g1.0wall-1step300000trial50",
        "log/JS200000/JSAQLearnerTunnel2Goale0.1g1.0wall-1step10000trial50", # 1万回は省略して集計だけ1万回と20万回で行った
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

    StateCount(files)
