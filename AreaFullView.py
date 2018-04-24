import numpy as np

class AreaFullView:
    # ターゲットの座標を返す
    def __init__(self, maxy, maxx):
        self.maxy = maxy
        self.maxx = maxx
        self.state = self.__init_state()

    def __init_state(self):
        return np.zeros([self.maxy, self.maxx]).astype('uint8')

    def reset_state(self):
        self.state = self.__init_state()

    def getId(self,position):
        if position[0] < 0 or position[0] >= self.maxy or position[1] < 0 or position[1] >= self.maxx:
            return 0
        else:
            return self.state[tuple(position)]

    def setId(self,position, id):
        # 強制的にその場所にidのオブジェクトを置く。Area側では妥当性チェックはしない
        self.state[tuple(position)] = id

    def search_target(self):
        # positionの値に関わらず、ターゲットの位置を返す
        for y in range(self.maxy):
            for x in range(self.maxx):
                if self.state[y,x] == 128:
                    return y, x
        return self.maxy, self.maxx # ターゲットが存在しない場合。想定外

    def random_position(self):
        # 他のオブジェクトがない場所をランダムに返す
        while True:
            y = np.random.randint(self.maxy)
            x = np.random.randint(self.maxx)
            if self.state[y, x] == 0:
                break
        return [y, x]

    def find_object(self,id):
        ret = []
        for y in range(self.maxy):
            for x in range(self.maxx):
                if self.state[y,x] == id:
                    ret.append([y,x])
        return ret