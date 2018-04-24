from Agent import *

class JSQLearner(Agent):
    def __init__(self, id, nstate, naction, params, env):
        super().__init__(id, nstate, naction, params, env)

    # デフォルトのQLearnerとはMixed Policyなところだけが違う
    # 環境に対して、すべての状態がobserveできるように依頼することに相当
    @classmethod
    def mixed(cls):
        return True
