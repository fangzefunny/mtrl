import os
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns 

from utils.env_fn import two_stage
from utils.model import *
from utils.viz import viz 
viz.get_style()

pth = os.path.dirname(os.path.abspath(__file__))
# create the folders if not existed
folders = ['figures', 'data']
for f in folders:
    if not os.path.exists(f'{pth}/{f}'):
        os.mkdir(f'{pth}/{f}')

def get_model_behaviors(exp):

    agents = [model_free, model_base, UVFA, SFGPI]
    sim_data = []
    for agent in agents:

        env   = two_stage(config=exp)
        rng   = np.random.RandomState(1234)
        model = agent(env=env, params=agent.params0, rng=rng)
        model.train()
        s_final, _ = model.test()
        s_final2 = {k+1: [i] for k, i in s_final.items()}
        sim_datum = pd.DataFrame.from_dict(s_final2, orient='columns')
        sim_datum['agent'] = agent.name
        sim_data.append(sim_datum)
    
    pd.concat(sim_data, axis=0).to_csv(f'data/sim_{exp}.csv')

def show_sim(exp):

    sim_data = pd.read_csv(f'data/sim_{exp}.csv', index_col=0)
    agents = ['MB', 'MF', 'SFGPI', 'UVFA']
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 7.5))
    for i, agent in enumerate(agents):
        ax = axs[i//2, i%2]
        sel_data = sim_data.query(f'agent=="{agent}"')
        sel_data.drop(columns='agent')
        sns.barplot(sel_data, color=viz.Palette[i], ax=ax)
        ax.set_title(agent)
        ax.set_box_aspect(1)
    fig.tight_layout()
    plt.savefig(f'figures/sim_{exp}.png', dpi=300)



if __name__ == '__main__':

    exp = 'exp1'
    get_model_behaviors('exp1')
    show_sim(exp)
   
