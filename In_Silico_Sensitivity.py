import numpy as np
import pandas as pd
from generate_glv_simulations import glv_simulator, initialize_glv, random_training_samples
from reconstruct_from_ss import reconstruct_from_ss
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from scipy.stats import spearmanr
import networkx as nx
from plot_helpers import scatter_plot

#%%

n = 50
seed=12345

A,r,baseline = initialize_glv(n,seed=seed)
sim = glv_simulator(A=A,r=r)
perturbed = random_training_samples(A=A,r=r,n_train_samples=500,seed=seed)

#%%

compositional = False
A_pred,r_pred = reconstruct_from_ss(perturbed, compositional=compositional)

# %% Sensitivity Analysis
keystoneness =  sim.bcd_keystones()

alphas = [1e-20,1e-15,1e-10,1e-5,1]
diag_alphas = [1,1e2,1e4,1e6,1e8,1e10]
diag_epsilons = [-1e1,-1e0,-1e-1,-1e-2,-1e-3]
lrs = [1e-9,1e-6,1e-3,1]
samples = random_training_samples(A,r,500)
n_rand_subsamples = 5

rng = np.random.default_rng(seed=seed)

spearman_scores = []
for alpha in alphas:
    temp_scores = []
    for i in range(n_rand_subsamples):
        samples = perturbed + rng.normal(loc=0,scale=1/n,size=perturbed.shape)
        samples[samples<0] = 0
        samples = (samples.T/samples.sum(axis=1)).T
        A_pred,r_pred = reconstruct_from_ss(samples, compositional=True,max_iter=5000,alpha=alpha)
        reconstructed_sim = glv_simulator(A=A_pred,r=r_pred)
        keystoneness_ss =  reconstructed_sim.bcd_keystones()
        temp_scores.append(spearmanr(keystoneness,keystoneness_ss).statistic)
    spearman_scores.append(np.mean(temp_scores))
plt.figure()
plt.plot(alphas,spearman_scores)
plt.ylim(0,1)
plt.xscale('log')
plt.xticks(alphas)
plt.title('Correlation Between True and Reconstructed K$_{BC}$',fontsize=18)
plt.xlabel(r'$\alpha_{L1}$',fontsize=16)
plt.ylabel('$K_{BC}$ Spearman Correlation',fontsize=16)
plt.savefig('sensitivity_a_l1.eps',format='eps',bbox_inches='tight')

spearman_scores = []
for diag_alpha in diag_alphas:
    temp_scores = []
    for i in range(n_rand_subsamples):
        samples = perturbed + rng.normal(loc=0,scale=1/n,size=perturbed.shape)
        samples[samples<0] = 0
        samples = (samples.T/samples.sum(axis=1)).T
        A_pred,r_pred = reconstruct_from_ss(samples, compositional=True,max_iter=5000,diag_alpha=diag_alpha)
        reconstructed_sim = glv_simulator(A=A_pred,r=r_pred)
        keystoneness_ss =  reconstructed_sim.bcd_keystones()
        temp_scores.append(spearmanr(keystoneness,keystoneness_ss).statistic)
    spearman_scores.append(np.mean(temp_scores))
plt.figure()
plt.plot(diag_alphas,spearman_scores)
plt.ylim(0,1)
plt.xscale('log')
plt.xticks(diag_alphas)
plt.title('Correlation Between True and Reconstructed K$_{BC}$',fontsize=18)
plt.xlabel(r'$\alpha_{diag}$',fontsize=16)
plt.ylabel('$K_{BC}$ Spearman Correlation',fontsize=16)
plt.savefig('sensitivity_a_diag.eps',format='eps',bbox_inches='tight')

spearman_scores = []
for diag_epsilon in diag_epsilons:
    temp_scores = []
    for i in range(n_rand_subsamples):
        samples = perturbed + rng.normal(loc=0,scale=1/n,size=perturbed.shape)
        samples[samples<0] = 0
        samples = (samples.T/samples.sum(axis=1)).T
        A_pred,r_pred = reconstruct_from_ss(samples, compositional=True,max_iter=5000,diag_alpha=diag_alpha)
        reconstructed_sim = glv_simulator(A=A_pred,r=r_pred)
        keystoneness_ss =  reconstructed_sim.bcd_keystones()
        temp_scores.append(spearmanr(keystoneness,keystoneness_ss).statistic)
    spearman_scores.append(np.mean(temp_scores))
plt.figure()
plt.plot(np.abs(diag_epsilons),spearman_scores)
plt.ylim(0,1)
plt.xscale('log')
plt.xticks(np.abs(diag_epsilons),diag_epsilons)
plt.title('Correlation Between True and Reconstructed K$_{BC}$',fontsize=18)
plt.xlabel(r'$\epsilon_{diag}$',fontsize=16)
plt.ylabel('$K_{BC}$ Spearman Correlation',fontsize=16)
plt.savefig('sensitivity_e_diag.eps',format='eps',bbox_inches='tight')

spearman_scores = []
for lr in lrs:
    temp_scores = []
    for i in range(n_rand_subsamples):
        samples = perturbed + rng.normal(loc=0,scale=1/n,size=perturbed.shape)
        samples[samples<0] = 0
        samples = (samples.T/samples.sum(axis=1)).T
        A_pred,r_pred = reconstruct_from_ss(samples, compositional=True,max_iter=5000,lr=lr)
        reconstructed_sim = glv_simulator(A=A_pred,r=r_pred)
        keystoneness_ss =  reconstructed_sim.bcd_keystones()
        temp_scores.append(spearmanr(keystoneness,keystoneness_ss).statistic)
    spearman_scores.append(np.mean(np.nan_to_num(temp_scores)))
plt.figure()
plt.plot(lrs,spearman_scores)
plt.ylim(0,1)
plt.xscale('log')
plt.xticks(lrs)
plt.title('Correlation Between True and Reconstructed K$_{BC}$',fontsize=18)
plt.xlabel('LR',fontsize=16)
plt.ylabel('$K_{BC}$ Spearman Correlation',fontsize=16)
plt.savefig('sensitivity_lr.eps',format='eps',bbox_inches='tight')

# %%
