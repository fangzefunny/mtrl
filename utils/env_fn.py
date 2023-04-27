import numpy as np 
import pandas as pd 

class two_stage:
    nS = 13
    nA = 3
    nC = 3

    def __init__(self, config, p=1, seed=1234):
        '''A MDP is a 5-element-tuple

        S: state space
        A: action space
        T: transition function
        R: reward function
        γ: discount factor
        '''
        self.config = config
        self.rng    = np.random.RandomState(seed)
        self.p = p
        self.s_termination = list(range(4, 13))
        self._init_state()
        self._init_action()
        self._init_trans_fn()
        self._init_rew_fn()

    def _init_state(self):
        self.S  = np.arange(two_stage.nS)

    def _init_action(self):
        self.A  = np.arange(two_stage.nA)

    def _init_trans_fn(self):
        '''T(s'|s,a)
        
        p = T['state]['act']['state_next']
        '''
        def get_prob(s_tar, s_all):
            eps = (1-self.p) / (len(s_all)-1)
            prob = {s: eps for s in s_all}
            prob[s_tar] = self.p
            return prob

        def get_prob_absorb(s):
            return {a: {s: 1} for a in self.A}

        self.T ={
            0: {0: get_prob(1, [1, 2, 3]), 
                1: get_prob(2, [1, 2, 3]),
                2: get_prob(3, [1, 2, 3])},

            1: {0: get_prob(4, [4, 5, 6]), 
                1: get_prob(5, [4, 5, 6]),
                2: get_prob(6, [4, 5, 6])}, 

            2: {0: get_prob(7, [7, 8, 9]), 
                1: get_prob(8, [7, 8, 9]),
                2: get_prob(9, [7, 8, 9])}, 

            3: {0: get_prob(10, [10, 11, 12]), 
                1: get_prob(11, [10, 11, 12]),
                2: get_prob(12, [10, 11, 12])},
        }

        # for last state
        for s in list(range(4, 13)):
            self.T[s] = get_prob_absorb(s)

    def _init_rew_fn(self):
        '''R(s) = w * phi
        '''
        # get weight 
        weights = {
            0: [1, -2, 0],
            1: [-2, 1, 0],
            2: [1, -1, 0],
            3: [-1, 1, 0],
            4: [1, 1, 1]
        }

        # object
        object ={
            'exp1': {
                4:  [  0,  10,   0],
                5:  [100,   0,   0],
                6:  [ 70,  70,  70],
                7:  [ 90,   0,   0],
                8:  [100, 100,   0],
                9:  [  0,  90,   0],
                10: [  0,   0,  10],
                11: [  0, 100,  60],
                12: [ 10,   0,   0],
            },
            'exp2': {
                4:  [ 10,  20,  90],
                5:  [100,   0,  20],
                6:  [ 70,  70, 170],
                7:  [ 90,   0,  40],
                8:  [150, 150,   0],
                9:  [  0,  90,  30],
                10: [ 30,  10,  50],
                11: [  0, 100, 160],
                12: [ 20,  10, 100],
            },
            'exp3': {
                4:  [  0,  10,   0],
                5:  [100,   0,   0],
                6:  [ 40,  40, 130],
                7:  [ 90,   0,   0],
                8:  [100, 100,   0],
                9:  [  0,  90,   0],
                10: [  0,   0,  10],
                11: [  0, 100,  60],
                12: [ 10,   0,   0],
            },
        }
        
        self.w   = weights
        self.phi = {k: np.array(phi)/10 for k, phi in object[self.config].items()}
        for i in range(4):
            self.phi[i] = np.array([0, 0, 0])

    def reset(self):
        '''always start with state=0
        '''
        self.s = 0
        self.done = False 
        return self.s 
    
    def step(self, a):
        
        # get next state 
        s_next_space = list(self.T[self.s][a].keys())
        probs = [p for _, p in self.T[self.s][a].items()]
        s_next = self.rng.choice(s_next_space, p=probs)
        # get the reward
        obj = self.phi[s_next]
        # check the termination
        if s_next > 3: self.done = True 
        # move on 
        self.s = s_next 

        return self.s, obj, self.done 

    def instan(self, n_rep=25, seed=3072):
        
        rng = np.random.RandomState(seed)
        data = {}

        # training block 
        block_w = np.hstack([rng.choice(4, size=4, replace=False)       
                             for _ in range(n_rep)])
        data['w_idx']  = block_w.tolist()
        data['stim']   = [0]*n_rep*4
        data['config'] = ['exp1']*n_rep*4
        data['tps']    = np.hstack([[i]*n_rep for i in range(4)]).tolist()
        data['stage']  = ['train']*n_rep*4
        
        # testing block 
        data['w_idx']  += [5]
        data['stim']   += [0]
        data['config'] += ['exp1']
        data['tps']    += [0]
        data['stage']  += ['test']

        data['trial' ] = list(range(4*n_rep+1))

        return pd.DataFrame.from_dict(data)
    
