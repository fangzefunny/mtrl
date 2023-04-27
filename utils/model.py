import torch
import torch.nn as nn 
from torch.optim import Adam
import numpy as np
from scipy.special import softmax

class baseAgent:

    def __init__(self, env, params, rng):
        self.env = env 
        self.rng = rng 
        self._load_params(params)
        
    def _load_params(self, params):
        return NotImplementedError

    def train(self):
        return NotImplementedError
    
    def test(self):
        return NotImplementedError
    
class model_free(baseAgent):
    params0 = [.3, 10, .1]
    name = 'MF'

    def __init__(self, env, params, rng):
        super().__init__(env, params, rng)
        self._init_Q()

    def _load_params(self, params):
        self.alpha = params[0] # learning rate 
        self.beta  = params[1] # inverse temperature 
        self.eps   = params[2] # epsilon

    def _init_Q(self):
        self.Q = self.rng.rand(self.env.nS, self.env.nA)*.00001

    def e_greedy(self, q):
        a_max = np.argwhere(q==np.max(q)).flatten()
        policy = np.sum([np.eye(self.env.nA)[i] for i in a_max], axis=0) / len(a_max)
        if self.rng.rand() < 1-self.eps:
            a = self.rng.choice(self.env.nA, p=policy)
        else:
            a = self.rng.choice(self.env.nA)
        return a 

    def train(self, n_unroll=200):

        # get the train weight 0-3
        train_w = [self.env.w[i] for i in range(4)] 
        all_train_ws = np.tile(train_w, (n_unroll, 1))
        N = n_unroll*len(train_w)
        ridx = self.rng.choice(N, size=N, replace=False)
        all_train_ws = all_train_ws[ridx]
        
        for i in range(n_unroll):

            s = self.env.reset()
            done = self.env.done

            while not done:
                
                # forward, predict the Q value, and get action using e-greedy
                q = self.Q[s, :]
                a = self.e_greedy(q)
                # get the next state 
                s_next, phi, done = self.env.step(a)
                # compute the reward 
                r = (phi * all_train_ws[i]).sum()
                # update: Q(s,a) = Q(s,a) + alpha[r + max_a'Q(s',a') - Q(s,a)]
                rpe = r +  (self.Q[s_next, :]).max() - q[a]
                self.Q[s,a] += self.alpha*rpe

                s = s_next

    def test(self, n_unroll=100):

        # get the train weight 0-3
        w_test = self.env.w[4]
        s_lst = [] 
        r_lst = []
    
        for _ in range(n_unroll):

            s = self.env.reset()
            done = self.env.done

            while not done:
                
                # forward, predict the Q value, and get action using e-greedy
                q = self.Q[s, :]
                a = self.e_greedy(q)
                # get the next state 
                s_next, phi, done = self.env.step(a)
                # compute the reward 
                r = (phi * w_test).sum()
                # move on 
                s = s_next 
            
            # collect the final state and reward
            s_lst.append(s)
            r_lst.append(r)

        s_final = {s2: s_lst.count(s2) / n_unroll for s2 in range(4, 13)}

        return s_final, np.mean(r_lst)

class model_base(baseAgent):
    params0 = [.99, 2]
    name = 'MB'

    def _load_params(self, params):
        self.gamma = params[0]
        self.beta  = params[1]

    def value_iter(self, w, tol=1e-1):

        # init value fn
        V = np.array([(self.env.phi[s]*w).sum() for s in self.env.S])

        while True:

            delta = 0 
            policy = {}
            for s in self.env.S:
                v = V[s].copy()
                q = np.zeros([self.env.nA])
                r = (self.env.phi[s]*w).sum()
                # v = r + max_a \sum_s' r(s') + γv(s') 
                for a in self.env.A:
                    for s_next, p_next in self.env.T[s][a].items():
                        mask = (s in self.env.s_termination)
                        q[a] += p_next * (r + (1-mask)*self.gamma*V[s_next])
                V[s] = np.max(q)
                policy[s] = softmax(self.beta*q)
                delta = np.max([delta, np.abs(V[s] - v)])
                
            if delta < tol: break

        return V, policy 
    
    def train(self,):
        pass 

    def test(self, n_unroll=100):

        # get the test weight
        w_test = self.env.w[4]
        # get the policy and the value function
        _, policy = self.value_iter(w_test)
        s_lst = [] 
        r_lst = []
    
        for _ in range(n_unroll):

            s = self.env.reset()
            done = self.env.done

            while not done:
                
                # forward, predict the Q value, and get action using e-greedy
                a = self.rng.choice(self.env.nA, p=policy[s])
                # get the next state 
                s_next, phi, done = self.env.step(a)
                # compute the reward 
                r = (phi * w_test).sum()
                # move on 
                s = s_next 
            
            # collect the final state and reward
            s_lst.append(s)
            r_lst.append(r)

        s_final = {s2: s_lst.count(s2) / n_unroll for s2 in range(4, 13)}

        return s_final, np.mean(r_lst)
    
class uv_net(nn.Module):

    def __init__(self, in_dim, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(in_dim, n_hidden),
                nn.Sigmoid(),
                nn.Linear(n_hidden, 1)
            )
        self.optim = Adam(self.net.parameters(), lr=.05)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        return self.net(x)

    def train(self, x, y, tol=1e-8, max_epoch=5000):

        x_nn = torch.FloatTensor(x).view(x.shape[0], -1)
        y_nn = torch.FloatTensor(y).view(x.shape[0], -1)
        ind  = list(range(x.shape[0]))
        loss_prev = 0
        epoch = 0 

        while True:
            
            np.random.shuffle(ind)
            x_nn = x_nn[ind, :]
            y_nn = y_nn[ind]

            # forward
            y_nn_hat = self.net.forward(x_nn)
            # calculate loss 
            loss = self.loss_fn(y_nn_hat, y_nn)
            # backward
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            #check convergence
            delta = torch.abs(loss_prev - loss)
            if (delta < tol) or (epoch > max_epoch) : 
                break
            
            loss_prev = loss
            epoch += 1

        return loss.data.numpy()

class UVFA(model_base):
    name = 'UVFA'

    def train(self, n_iter=100):

        # preprare the training input and label 
        uv, ux = [], []
        for c in range(4):
            ws_train = self.env.w[c]
            # the state and weight as input
            x = np.hstack([np.eye(self.env.nS), 
                       np.tile(ws_train, [self.env.nS, 1])])
            # get the value function as label
            v, _ = self.value_iter(ws_train)
            ux.append(x)
            uv.append(v)
        ux = np.vstack(ux)    # [nSxnC, nS+nC]
        uv = np.hstack(uv).T  # [nSxnC, 1]

        ux_train = np.tile(ux, [n_iter, 1])
        uv_train = np.tile(uv, [n_iter, 1])
        
        self.uv_approx = uv_net(in_dim=ux.shape[1], n_hidden=10)
        self.uv_approx.train(ux_train, uv_train)

    def test(self, n_unroll=100):

        # get the test weight
        w_test = self.env.w[4]
        # the state and weight as input
        x = np.hstack([np.eye(self.env.nS), 
                np.tile(w_test, [self.env.nS, 1])])
        V = self.uv_approx(torch.FloatTensor(x)).data.numpy()
        policy = {}
        for s in self.env.S:
            q = np.zeros([self.env.nA])
            r = (self.env.phi[s]*w_test).sum()
            # v = r + max_a \sum_s' r(s') + γv(s') 
            for a in self.env.A:
                for s_next, p_next in self.env.T[s][a].items():
                    mask = (s in self.env.s_termination)
                    q[a] += p_next * (r + (1-mask)*self.gamma*V[s_next])
            policy[s] = softmax(q)
    
        s_lst = [] 
        r_lst = []
    
        for _ in range(n_unroll):

            s = self.env.reset()
            done = self.env.done

            while not done:
                
                # forward, predict the Q value, and get action using e-greedy
                a = self.rng.choice(self.env.nA, p=policy[s])
                # get the next state 
                s_next, phi, done = self.env.step(a)
                # compute the reward 
                r = (phi * w_test).sum()
                # move on 
                s = s_next 
            
            # collect the final state and reward
            s_lst.append(s)
            r_lst.append(r)

        s_final = {s2: s_lst.count(s2) / n_unroll for s2 in range(4, 13)}

        return s_final, np.mean(r_lst)
        
class SFGPI(model_base):
    name = 'SFGPI'

    def train(self, tol=.1):
        
        # get value 
        pis = {}
        for c in range(4):
            _, pi = self.value_iter(self.env.w[c])
            pis[c] = pi

        # get succesor feature 
        psi = {c: {} for c in range(4)}
        for c in range(4):

            # init 
            for s in self.env.S:
                psi[c][s] = self.env.phi[s]
            
            while True:
                delta = 0
                for s in self.env.S:
                    psi_old = psi[c][s].copy()
                    psi_new = self.env.phi[s].astype(float)
                    for a in self.env.A:
                        assert(np.abs(pis[c][s].sum() - 1) < 1e-12)
                        for s_next in self.env.S:
                            s_next_ava = list(self.env.T[s][a].keys())
                            p_s_next = self.env.T[s][a][s_next] if s_next in s_next_ava else 0 
                            psi_new += pis[c][s][a]*p_s_next*self.gamma*psi[c][s_next]
                    psi[c][s] = psi_new 
                    delta = np.max([delta, np.linalg.norm(psi_new - psi_old)])

                if delta < tol: break 

        self.psi = psi 

    def test(self, n_unroll=100):

        # get the value function absed on policy
        w_test = self.env.w[4]
        v = np.zeros(self.env.nS)
        for s in self.env.S:
            vals = [0]+[(psi_c[s]*w_test).sum() for _, psi_c in self.psi.items()]
            v[s] = np.max(vals)
        policy = {}
        for s in self.env.S:
            q = np.zeros([self.env.nA])
            r = (self.env.phi[s]*w_test).sum()
            # v = r + max_a \sum_s' r(s') + γv(s') 
            for a in self.env.A:
                for s_next, p_next in self.env.T[s][a].items():
                    mask = (s in self.env.s_termination)
                    q[a] += p_next * (r + (1-mask)*self.gamma*v[s_next])
            policy[s] = softmax(q)

        s_lst = [] 
        r_lst = []
    
        for _ in range(n_unroll):

            s = self.env.reset()
            done = self.env.done

            while not done:
                
                # forward, predict the Q value, and get action using e-greedy
                a = self.rng.choice(self.env.nA, p=policy[s])
                # get the next state 
                s_next, phi, done = self.env.step(a)
                # compute the reward 
                r = (phi * w_test).sum()
                # move on 
                s = s_next 
            
            # collect the final state and reward
            s_lst.append(s)
            r_lst.append(r)

        s_final = {s2: s_lst.count(s2) / n_unroll for s2 in range(4, 13)}

        return s_final, np.mean(r_lst)


        



