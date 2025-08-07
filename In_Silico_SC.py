# The dependencies for this file are challenging. Suggest using a separate environment from all other code in this project.
import numpy as np
import pandas as pd
import ryp
ryp.r('library(miaSim)')
from generate_glv_simulations import glv_simulator, initialize_glv, random_training_samples, generate_klemm_net
from reconstruct_from_ss import reconstruct_from_ss,self_consistency_score
import matplotlib.pyplot as plt
import seaborn as sns
from SparCC import SparCC
import networkx as nx
from scipy.stats import spearmanr
#%%
seed=12345
n=50
A = generate_klemm_net(n=n,seed=seed) # generate klemm distributed interactions matrix
ryp.to_r(A,'A') # convert to R so that it can be used in miaSim
soi_string = 'simulateSOI(n_species,x0 = x0,names_species = NULL,carrying_capacity = 10000,A = A,k_events = 5,t_end = 1000,metacommunity_probability=runif(n_species, min = 0.0001, max = 0.001))'
#%%
rng = np.random.default_rng(seed=seed)
train_samples = np.where(rng.random((100,n)) < 0.2,0.,1.)
ts = train_samples.copy()
ryp.to_r(n,'n_species')
for i,idx in enumerate(train_samples):
    idx = np.array(idx)*10
    ryp.to_r(idx,'x0')
    sample = ryp.to_py(soi_string,format='numpy')
    train_samples[i,:] = sample['assays']['data']['listData']['counts'][:,-1]

perturbed = (train_samples.T/train_samples.sum(axis=1)).T
compositional=True

# %%
scores = []
n_rand_subsamples = 10
sigma_list = np.arange(0,2,0.1)
for sigma in sigma_list:
    temp_scores = []
    for i in range(n_rand_subsamples):
        samples = perturbed + rng.normal(loc=0,scale=sigma/n,size=perturbed.shape)
        samples[samples<0] = 0
        samples = (samples.T/samples.sum(axis=1)).T
        A_pred,r_pred = reconstruct_from_ss(samples, compositional=compositional,max_iter=5000)
        scores.append(self_consistency_score(samples,A_pred,r_pred))
soi_scores = scores.copy()
# %%
rng = np.random.default_rng(seed=seed)

A,r,baseline = initialize_glv(n,seed=12345)
sim = glv_simulator(A=A,r=r)

perturbed = random_training_samples(A=A,r=r,n_train_samples=100,seed=seed)
perturbed = (perturbed.T/perturbed.sum(axis=1)).T
scores = []

for sigma in sigma_list:
    temp_scores = []
    for i in range(n_rand_subsamples):
        samples = perturbed + rng.normal(loc=0,scale=sigma/n,size=perturbed.shape)
        samples[samples<0] = 0
        samples = (samples.T/samples.sum(axis=1)).T
        A_pred,r_pred = reconstruct_from_ss(samples, compositional=compositional,max_iter=5000)
        scores.append(self_consistency_score(samples,A_pred,r_pred))
glv_scores = scores.copy()

#%%
rng = np.random.default_rng(seed=seed)

perturbed = rng.random(size=(100,n))
perturbed = (perturbed.T/perturbed.sum(axis=1)).T
scores = []

for sigma in sigma_list:
    temp_scores = []
    for i in range(n_rand_subsamples):
        samples = perturbed + rng.normal(loc=0,scale=sigma/n,size=perturbed.shape)
        samples[samples<0] = 0
        samples = (samples.T/samples.sum(axis=1)).T
        A_pred,r_pred = reconstruct_from_ss(samples, compositional=compositional,max_iter=5000)
        scores.append(self_consistency_score(samples,A_pred,r_pred))
noise_scores = scores.copy()
# %%
soi_scores = pd.DataFrame(data=soi_scores,columns=['Score'])
soi_scores['Model'] = 'Noisy SOI'
noise_scores = pd.DataFrame(data=noise_scores,columns=['Score'])
noise_scores['Model'] = 'Noise'
glv_scores = pd.DataFrame(data=glv_scores,columns=['Score'])
glv_scores['Model'] = 'Noisy GLV'
scores = pd.concat([glv_scores,soi_scores,noise_scores],axis=0)
sns.histplot(scores,x='Score',hue='Model',kde=True)
plt.xlabel('S$_{sc}$')
plt.ylabel('Number of Modeled Systems')
plt.savefig('sc_eval.eps',format='eps',bbox_inches='tight')

plt.figure()
sns.boxplot(scores,y='Score',hue='Model',x='Model')
plt.ylabel('S$_{sc}$')
plt.savefig('sc_boxplot.eps',format='eps',bbox_inches='tight')

from scipy.stats import kruskal
s,p = kruskal(glv_scores.Score,soi_scores.Score,noise_scores.Score)
print(f'Kruskal Test P Value: {p}')
# %%

import numpy as np
import pandas as pd
from generate_glv_simulations import glv_simulator, initialize_glv, random_training_samples, generate_klemm_net
from reconstruct_from_ss import reconstruct_from_ss,self_consistency_score
import matplotlib.pyplot as plt
import seaborn as sns
from SparCC import SparCC
import networkx as nx
from scipy.stats import spearmanr

def scatter_plot(x,y,xlabel='x',ylabel='y',title='title',ax=None):
    if ax is None:
        plt.figure(dpi=250)
        ax = plt.axes()
    ax.plot(x,y,'.',markersize=5)
    statistic = spearmanr(x,y).statistic
    pvalue = spearmanr(x,y).pvalue
    if pvalue < 0.01:
        pstring = f'p={pvalue:.1e}'
    else:
        pstring = f'p={pvalue:.2f}'
    ax.annotate(f'ρ= {statistic:.2f}, {pstring}',xy=(0.025,0.9),xycoords='axes fraction',fontsize=10)
    ax.set_xlabel(xlabel,fontsize=10)
    ax.set_ylabel(ylabel,fontsize=10)
    ax.set_title(title,fontsize=12)

# %%
n=50
seed=12345
A,r,baseline = initialize_glv(n,seed=seed)
rng = np.random.default_rng(seed=seed)
perturbed = random_training_samples(A=A,r=r,n_train_samples=100,seed=seed)
perturbed = (perturbed.T/perturbed.sum(axis=1)).T
sim = glv_simulator(A,r)
keystoneness = sim.bcd_keystones()
scores = []
sc_scores = []
n_rand_subsamples = 5
sigma_list = np.arange(0,2,0.1)
for sigma in sigma_list:
    for i in range(n_rand_subsamples):
        samples = perturbed + rng.normal(loc=0,scale=sigma/n,size=perturbed.shape)
        samples[samples<0] = 0
        samples = (samples.T/samples.sum(axis=1)).T
        A_pred,r_pred = reconstruct_from_ss(samples,max_iter=5000)
        reconstructed_sim = glv_simulator(A=A_pred,r=r_pred)
        keystoneness_ss =  reconstructed_sim.bcd_keystones()
        rho = spearmanr(keystoneness,keystoneness_ss).statistic
        scores.append(rho)
        sc_scores.append(self_consistency_score(samples,A_pred,r_pred))

scatter_plot(scores,sc_scores,'K$_{DK}$ Predictive Correlation','S$_{sc}$','Prediction Correlation vs. Self Consistency')

# %%
