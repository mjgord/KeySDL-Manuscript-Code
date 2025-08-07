import numpy as np
import pandas as pd
from generate_glv_simulations import glv_simulator, initialize_glv, random_training_samples
from reconstruct_from_ss import reconstruct_from_ss
import matplotlib.pyplot as plt
import seaborn as sns
from SparCC import SparCC
import networkx as nx
from scipy.stats import spearmanr
from plot_helpers import scatter_plot

#%%

n = 50

A,r,baseline = initialize_glv(n,seed=12345)
sim = glv_simulator(A=A,r=r)

perturbed = random_training_samples(A=A,r=r,n_train_samples=500)
perturbed = (perturbed.T/perturbed.sum(axis=1)).T

plt.figure()
bwn_df = pd.DataFrame([r,perturbed.mean(axis=0)],index=['GLV Growth Rates','Mean Relative Abundance']).T
scatter_plot(bwn_df,'GLV Growth Rates vs. Mean Relative Abundance')

#%%

compositional = True
A_pred,r_pred = reconstruct_from_ss(perturbed, compositional=compositional,verbose=False)

#%% Figure 2

keystoneness =  sim.bcd_keystones()

reconstructed_sim = glv_simulator(A=A_pred,r=r_pred)
keystoneness_ss =  reconstructed_sim.bcd_keystones()

kbc_df = pd.DataFrame([keystoneness,keystoneness_ss],index=['True K$_{BC}$','Reconstructed K$_{BC}$']).T
scatter_plot(kbc_df,'True vs. Compositionally Reconstructed K$_{BC}$')
plt.savefig('replicator_keystones.eps',format='eps',bbox_inches='tight')

corr,cov = SparCC(perturbed)
corr_net = (corr > 0.02)
corr_net = nx.from_numpy_array(corr_net)

plt.figure()
deg_df = pd.DataFrame([keystoneness,nx.degree_centrality(corr_net).values()],index=['True K$_{BC}$','Co-Occurrence Degree']).T
scatter_plot(deg_df,'True K$_{BC}$ vs. Co-Occurrence Degree')
plt.savefig('replicator_degree.eps',format='eps',bbox_inches='tight')

plt.figure()
bwn_df = pd.DataFrame([keystoneness,nx.betweenness_centrality(corr_net).values()],index=['True K$_{BC}$','Co-Occurrence Betweenness']).T
scatter_plot(bwn_df,'True K$_{BC}$ vs. Co-Occurrence Betweenness')
plt.savefig('replicator_betweenness.eps',format='eps',bbox_inches='tight')

plt.figure()
perturbed = random_training_samples(A=A_pred,r=r_pred,n_train_samples=500)
perturbed = (perturbed.T/perturbed.sum(axis=1)).T
kbc_df = pd.DataFrame([keystoneness_ss,perturbed.mean(axis=0)],index=['Reconstructed K$_{BC}$','Mean Relative Abundance']).T
scatter_plot(kbc_df,'Reconstructed K$_{BC}$ vs. Simulated MRA')
#plt.savefig('replicator_mra.eps',format='eps',bbox_inches='tight')
#%% Figure 3
from scipy.stats import spearmanr
spearman_scores = []
n_rand_subsamples = 5
n_samples_list = np.geomspace(5,500,10).astype(int)
for n_train_samples in n_samples_list:
    temp_spearman = []
    for i in range(n_rand_subsamples):
        samples = random_training_samples(A,r,n_train_samples)
        A_pred,r_pred = reconstruct_from_ss(samples, compositional=compositional,max_iter=5000)
        reconstructed_sim = glv_simulator(A=A_pred,r=r_pred)
        keystoneness_ss =  reconstructed_sim.bcd_keystones()
        temp_spearman.append(spearmanr(keystoneness,keystoneness_ss).statistic)
    spearman_scores.append(np.mean(temp_spearman))


plt.plot(n_samples_list,spearman_scores)
plt.xticks([5,100,200,300,400,500])
plt.title('Correlation Between True and Compositionally Reconstructed K$_{BC}$')
plt.xlabel('Number of Samples Used In Reconstruction')
plt.ylabel('$K_{BC}$ Spearman Correlation')
plt.savefig('replicator_nsamples.eps',format='eps',bbox_inches='tight')

# %%'''
