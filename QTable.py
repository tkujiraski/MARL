import numpy as np

class QTable:
    """
    状態とアクションの組に対するQ値を保持するクラス

    """
    def __init__(self, nstate, naction, init_qvalue=[]):
        """ QTableの初期化
            nstate: 状態変数ごとの状態数の配列
            naction: アクションの数
        """
        self.nstate = nstate
        self.naction = naction
        if init_qvalue == []:
            self.qvalue = np.zeros(nstate+[naction])
        elif type(init_qvalue) == str:
            self.loadQvalue(init_qvalue)
        else:
            self.qvalue = init_qvalue.copy()
        return

    def initializeZero(self):
        self.qvalue = np.zeros(self.nstate+[self.naction])

    def getQ(self, state, action):
        """
        状態state, 行動actionのQ値を返す
        :param state:
        :param action:
        :return:
        """
        return self.qvalue[tuple(state + [action])]

    def getMaxQ(self, state):
        """
        状態stateで最大のQ値を返す
        :param state:
        :return: 最大のQ値
        """
        return self.qvalue[tuple(state)].max()

    def getMaxAction(self, state):
        """
        状態stateで最大のQ値となる行動aを返す
        :param state:
        :return: 最大のQ値となる行動a
        """
        return self.qvalue[tuple(state)].argmax()

    def getQvector(self,state):
        return self.qvalue[tuple(state)]

    # nstate, nactionも保存する必要があるかどうか未検討
    # QTableを保存するだけではQ値がどれだけ安定しているかは分からない。その情報は必要か？
    # マルチ側で平均値に対する検定をするのであれば必要ない？
    def saveQvalue(self, filename):
        np.save(filename+".npy", self.qvalue)
        return

    def loadQvalue(self, filename):
        self.qvalue = np.load(filename+".npy")
        return