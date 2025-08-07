#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from reconstruct_from_ss import reconstruct_from_ss, glv_keystone_estimator,glv_composition_estimator,self_consistency_score
from scipy.stats import spearmanr
from SparCC import SparCC
import networkx as nx
from plot_helpers import scatter_plot
from matplotlib.colors import to_hex
#%%
# colors for plotting
colormap = plt.get_cmap('jet')
colors = [to_hex(colormap(k)) for k in np.linspace(0, 1, 15)]

path = '../../data/Gutierrez2019'

dropout_data = pd.read_excel(f'{path}/gutierrez_2019_qpcr.xlsx',header=1)
dropout_data['Unnamed: 0'] = dropout_data['Unnamed: 0'].fillna(method='ffill')
dropout_data = dropout_data.set_index('Unnamed: 0')
dropout_data.replace(r'[^0-9.]',0.0,regex=True,inplace=True)

dropout_data = dropout_data[dropout_data['Time (h)']==30]
dropout_data.reset_index(inplace=True)
dropout_data = dropout_data.groupby('Unnamed: 0').mean()
dropout_data.drop(['Time (h)'],axis='columns',inplace=True)
dropout_data.index = dropout_data.index.str.replace('Δ','')
dropout_data.index = dropout_data.index.str.capitalize()

#%%

# separate baseline sample
dropout_species = dropout_data.index.drop(['All1','All2'])
dropout_data = dropout_data[dropout_species]


#TSS
dropout_data.loc[:,:] = (dropout_data.values.T/dropout_data.sum(axis=1).values).T

baseline = (dropout_data.loc['All1',:]+dropout_data.loc['All2',:])/2
baseline_comp = baseline/baseline.sum()
dropout_data.drop(['All1','All2'],axis='index',inplace=True)


#%%


baseline = np.repeat(baseline.values.reshape(1,-1),dropout_data.shape[1],axis=0)
baseline = baseline*(dropout_data!=0)
keystones = (np.abs(dropout_data-baseline)).sum(axis=1)/(np.abs(dropout_data+baseline)).sum(axis=1)
keystones = (np.abs(dropout_data-baseline)).sum(axis=1)/(np.abs(dropout_data+baseline)).sum(axis=1)

keystones = pd.Series(keystones,index=dropout_data.index)
# %%

train_samples = dropout_data.values

gksp = glv_keystone_estimator()

pred_keystones = cross_val_predict(gksp,X=dropout_data.values,cv=3)
pred_keystones = pd.Series(pred_keystones,index=dropout_species)

kbc_df = pd.DataFrame([keystones.values,pred_keystones[keystones.index].values],index=['True K$_{BC}$','Reconstructed K$_{BC}$'],columns=keystones.index).T
scatter_plot(kbc_df,'True vs. Reconstructed K$_{BC}$',color=True)
plt.savefig('gutierrez_keystones.eps',format='eps',bbox_inches='tight')

corr,cov = SparCC(train_samples)
corr_net = (corr > 0.02)
corr_net = nx.from_numpy_array(corr_net)

plt.figure()
deg_df = pd.DataFrame([keystones.values,nx.degree_centrality(corr_net).values()],index=['True K$_{BC}$','Co-Occurrence Degree'],columns=keystones.index).T
scatter_plot(deg_df,'True K$_{BC}$ vs. Co-Occurrence Degree',color=True)
plt.savefig('gutierrez_degree.eps',format='eps',bbox_inches='tight')

plt.figure()
bwn_df = pd.DataFrame([keystones.values,nx.betweenness_centrality(corr_net).values()],index=['True K$_{BC}$','Co-Occurrence Betweenness'],columns=keystones.index).T
scatter_plot(bwn_df,'True K$_{BC}$ vs. Co-Occurrence Betweenness',color=True)
plt.savefig('gutierrez_betweenness.eps',format='eps',bbox_inches='tight')

A_pred,r_pred = reconstruct_from_ss(train_samples)
sc_score = self_consistency_score(train_samples,A_pred,r_pred)
print(f'Self Consistency Score: {sc_score}')
# %%
from generate_glv_simulations import random_training_samples, glv_simulator
perturbed = random_training_samples(A=A_pred,r=r_pred,n_train_samples=500)
perturbed = (perturbed.T/perturbed.sum(axis=1)).T
kbc_df = pd.DataFrame([pred_keystones[keystones.index].values,perturbed.mean(axis=0)],index=['Reconstructed K$_{BC}$','Mean Relative Abundance'],columns=keystones.index).T
scatter_plot(kbc_df,'Reconstructed K$_{BC}$ vs. Simulated MRA',color=True)
plt.savefig('gutierrez_mra.eps',format='eps',bbox_inches='tight')
# %%

sim = glv_simulator(A_pred,r_pred)
pred_samples = np.zeros_like(train_samples)
for steady_state in range(train_samples.shape[0]):
    pred_samples[steady_state,:] = sim.ss_from_assemblage(train_samples[steady_state,:],compositional=True)

#%%
pred_errors = []
null_errors = []
fig = plt.figure(dpi=500,figsize=(8,4))
for row in range(train_samples.shape[0]):
    pred = pred_samples[row,:]
    true = train_samples[row,:]
    null = baseline_comp.values.copy()
    null[row] = 0
    null = null/np.sum(null)

    pred_errors.append(np.mean(np.abs(pred-true))/np.mean(np.abs(pred+true)))
    null_errors.append(np.mean(np.abs(null-true))/np.mean(np.abs(pred+true)))

    microbe_names = dropout_data.columns
    microbe_counts = {name:np.array([true[i],pred[i],null[i]]) for i,name in enumerate(microbe_names)}
    bottom = np.zeros(3)
    width=1
    hatch = ['','///','xxx']
    plt.subplot(2,8,row+1)
    i = 0
    for microbe, weight_count in microbe_counts.items():
        plt.bar(['True','Predicted','Null'], weight_count, width, label=microbe, bottom=bottom,hatch=hatch,color=colors[i])
        plt.yticks([])
        plt.xticks(fontsize=4)
        bottom += weight_count
        i += 1
    plt.title(microbe_names[row])

microbe_names = dropout_data.columns
microbe_counts = {name:np.array([baseline_comp[i]]) for i,name in enumerate(microbe_names)}
bottom = np.zeros(3)
width=1
plt.subplot(2,8,row+2)
i = 0
for microbe, weight_count in microbe_counts.items():
    plt.bar(['True'], weight_count, label=microbe, bottom=bottom,color=colors[i])
    plt.yticks([])
    plt.xticks(fontsize=4)
    bottom += weight_count
    i += 1
plt.title('Baseline')

handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('gutierrez_compositions.eps',format='eps',bbox_inches='tight')

pred_errors = pd.DataFrame(pred_errors)
pred_errors['Model Type'] = 'KeySDL'
null_errors = pd.DataFrame(null_errors)
null_errors['Model Type'] = 'Null Model'
errors = pd.concat([pred_errors,null_errors],axis=0)
plt.figure()
sns.boxplot(errors,y=0,hue='Model Type',x='Model Type')
plt.ylabel('Error (Bray-Curtis Distance)')
plt.savefig('gutierrez_errors.eps',format='eps',bbox_inches='tight')
# %%
