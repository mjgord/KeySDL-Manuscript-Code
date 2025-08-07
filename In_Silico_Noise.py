import numpy as np
import pandas as pd
from generate_glv_simulations import glv_simulator, initialize_glv, random_training_samples
from reconstruct_from_ss import reconstruct_from_ss
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

#%%

n = 50

seed = 12345

rng = np.random.default_rng(seed=seed)

A,r,baseline = initialize_glv(n,seed=seed)
sim = glv_simulator(A=A,r=r)

perturbed = random_training_samples(A=A,r=r,n_train_samples=100,seed=seed)
perturbed = (perturbed.T/perturbed.sum(axis=1)).T
#%%

compositional = True
A_pred,r_pred = reconstruct_from_ss(perturbed, compositional=compositional)

keystoneness =  sim.bcd_keystones()
#%% Figure 1
mean_scores = []
n_rand_subsamples = 5
sigma_list = np.arange(0,2,0.1)
for sigma in sigma_list:
    temp_scores = []
    for i in range(n_rand_subsamples):
        samples = perturbed + rng.normal(loc=0,scale=sigma/n,size=perturbed.shape)
        samples[samples<0] = 0
        samples = (samples.T/samples.sum(axis=1)).T
        A_pred,r_pred = reconstruct_from_ss(samples, compositional=compositional,max_iter=5000)
        reconstructed_sim = glv_simulator(A=A_pred,r=r_pred)
        keystoneness_ss =  reconstructed_sim.bcd_keystones()
        rho = spearmanr(keystoneness,keystoneness_ss).statistic
        temp_scores.append(rho)
    mean_scores.append(np.mean(temp_scores))


plt.plot(sigma_list*100,mean_scores)
plt.title('Impact of Additive Noise on K$_{BC}$ Prediction')
plt.xlabel('Gaussian Noise Standard Deviation (% of Mean Abundance)')
plt.ylabel('K$_{BC}$ Spearman Correlation')
plt.savefig('noise_eval.eps',format='eps',bbox_inches='tight')

#%% Figure 2
scores = []
n_rand_subsamples = 5
total_count_list = np.geomspace(2000,20000,10).astype(int)

for count in total_count_list:
    temp_scores = []
    for i in range(n_rand_subsamples):
        samples = perturbed
        samples[samples<0] = 0

        samples_p = (samples.T/samples.sum(axis=1)).T
        simulated_counts = np.zeros_like(samples_p)
        for row in range(samples_p.shape[0]):
            sim_reads = rng.choice(samples_p.shape[1],size=count,p=samples_p[row,:])
            idx,sim_count = np.unique(sim_reads,return_counts=True)
            simulated_counts[row,idx] = sim_count
        samples = (simulated_counts.T/np.sum(simulated_counts,axis=1)).T

        A_pred,r_pred = reconstruct_from_ss(samples, compositional=compositional,verbose=False,max_iter=5000)
        reconstructed_sim = glv_simulator(A=A_pred,r=r_pred)
        keystoneness_ss =  reconstructed_sim.bcd_keystones()
        temp_scores.append(spearmanr(keystoneness,keystoneness_ss).statistic)

    scores.append(np.mean(temp_scores))


plt.plot(total_count_list,scores)
plt.title('Impact of Library Size on K$_{BC}$ Prediction')
plt.xlabel('Library Size')
plt.ylabel('K$_{BC}$ Spearman Correlation')
plt.savefig('quant_eval.eps',format='eps',bbox_inches='tight')

#%%