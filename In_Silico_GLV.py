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


#%% Figure 1
print(f'A MSE: {np.mean((A-A_pred)**2):.2e}')
print(f'r MSE: {np.mean((r-r_pred)**2):.2e}')

fig = plt.figure()
cbar_ax = fig.add_axes([0.91,0.3,0.03,0.4])
vmin = np.min([np.min(A_pred),np.min(A)])
vmax = np.max([np.max(A_pred),np.max(A)])

plt.subplot(1,2,1,aspect='equal')
ax1 = sns.heatmap(A,vmin=vmin,vmax=vmax,cbar=False)
ax1.tick_params(left=False, bottom=False, labelbottom=False,labelleft=False)
plt.ylabel('Influencing Microbe')
plt.xlabel('Influenced Microbe')
plt.title('True A Matrix')

plt.subplot(1,2,2,aspect='equal')
ax2 = sns.heatmap(A_pred,vmin=vmin,vmax=vmax,cbar_ax=cbar_ax)
ax2.tick_params(left=False, bottom=False, labelbottom=False,labelleft=False)
plt.ylabel('Influencing Microbe')
plt.xlabel('Influenced Microbe')
plt.title('Reconstructed A Matrix')
plt.savefig('glv_reconstructed.eps',format='eps',bbox_inches='tight')

#%% Figure 2

keystoneness =  sim.bcd_keystones()

reconstructed_sim = glv_simulator(A=A_pred,r=r_pred)
keystoneness_ss =  reconstructed_sim.bcd_keystones()


kbc_df = pd.DataFrame([keystoneness,keystoneness_ss],index=['True K$_{BC}$','Reconstructed K$_{BC}$']).T
scatter_plot(kbc_df,'True vs. Reconstructed K$_{BC}$')
plt.savefig('glv_keystones.eps',format='eps',bbox_inches='tight')

corr = np.corrcoef(perturbed.T)
corr_net = (corr > 0.05)
corr_net = nx.from_numpy_array(corr_net)

plt.figure()
deg_df = pd.DataFrame([keystoneness,nx.degree_centrality(corr_net).values()],index=['True K$_{BC}$','Co-Occurrence Degree']).T
scatter_plot(deg_df,'True K$_{BC}$ vs. Co-Occurrence Degree')
plt.savefig('glv_degree.eps',format='eps',bbox_inches='tight')

plt.figure()
bwn_df = pd.DataFrame([keystoneness,nx.betweenness_centrality(corr_net).values()],index=['True K$_{BC}$','Co-Occurrence Betweenness']).T
scatter_plot(bwn_df,'True K$_{BC}$ vs. Co-Occurrence Betweenness')
plt.savefig('glv_betweenness.eps',format='eps',bbox_inches='tight')

plt.figure()
bwn_df = pd.DataFrame([keystoneness,perturbed.mean(axis=0)],index=['True K$_{BC}$','Mean Relative Abundance']).T
scatter_plot(bwn_df,'True K$_{BC}$ vs. Mean Relative Abundance')



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
plt.title('Correlation Between True and Reconstructed K$_{BC}$')
plt.xlabel('Number of Samples Used In Reconstruction')
plt.ylabel('$K_{BC}$ Spearman Correlation')
plt.savefig('glv_nsamples.eps',format='eps',bbox_inches='tight')

# %%'''
