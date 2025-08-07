#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from reconstruct_from_ss import reconstruct_from_ss, glv_keystone_estimator, glv_composition_estimator, self_consistency_score
import networkx as nx
from SparCC import SparCC
from plot_helpers import scatter_plot
from matplotlib.colors import to_hex

#%%
# colors for plotting
colormap = plt.get_cmap('jet')
colors = [to_hex(colormap(k)) for k in np.linspace(0, 1, 25)]

path = '../../data/Carlstrom2019_raw/'
# These files are used as provided by the authors of DOI: 10.1038/s41559-019-0994-z without modification
dropout_data_r1 = pd.read_csv('../../data/Carlstrom2019_raw/expt61.txt',delimiter='\t',index_col=0,decimal=',')
dropout_data_r2 = pd.read_csv('../../data/Carlstrom2019_raw/expt62.txt',delimiter='\t',index_col=0,decimal=',')
dropout_meta_r1 = pd.read_csv('../../data/Carlstrom2019_raw/metadata61.csv',delimiter=';',index_col=0,decimal=',')
dropout_meta_r2 = pd.read_csv('../../data/Carlstrom2019_raw/metadata62.csv',delimiter=';',index_col=0,decimal=',')
dropout_data_r1 = dropout_data_r1.loc[dropout_data_r1.index.str.contains('Leaf')]
dropout_data_r2 = dropout_data_r2.loc[dropout_data_r2.index.str.contains('Leaf')]

dropout_data_r1 = dropout_data_r1.T
dropout_data_r2 = dropout_data_r2.T

dropout_data_r1.index = dropout_data_r1.index.astype(int)
dropout_data_r2.index = dropout_data_r2.index.astype(int)

dropout_data_r1 = pd.merge(dropout_data_r1,dropout_meta_r1,left_index=True,right_index=True)
dropout_data_r2 = pd.merge(dropout_data_r2,dropout_meta_r2,left_index=True,right_index=True)
dropout_data_r1['Treatment'] = dropout_data_r1['Treatment'].str.replace('Axenic','Ax')

dropout_data = pd.concat([dropout_data_r1,dropout_data_r2],axis=0)

dropout_data = dropout_data[dropout_data['Treatment']!='Ax']
dropout_data = dropout_data[dropout_data['Time'] == 't2']

dropout_data = dropout_data.drop(labels=['Replicate','Time','Name'],axis='columns')
dropout_data = dropout_data.groupby(by='Treatment').mean() # average replicates

dropout_data.index = dropout_data.index.str.replace('-','L')
dropout_data.columns = dropout_data.columns.str.replace('Leaf','L')

# separate baseline sample
dropout_species = dropout_data.index.drop('ALL')
dropout_data = dropout_data[dropout_species]

# dropped-out microbes must be absent
dropout_data.iloc[:-1,:] = dropout_data.iloc[:-1,:]*(np.eye(dropout_data.shape[1])==0)

#TSS
dropout_data.loc[:,:] = (dropout_data.values.T/dropout_data.sum(axis=1).values).T

baseline = dropout_data.loc['ALL',:]
baseline_comp = baseline/baseline.sum()
dropout_data = dropout_data.loc[dropout_data.index != 'ALL',:]


baseline = np.repeat(baseline.values.reshape(1,-1),dropout_data.shape[1],axis=0)
baseline = baseline*(dropout_data!=0)
keystones = (np.abs(dropout_data-baseline)).sum(axis=1)/(np.abs(dropout_data+baseline)).sum(axis=1)

keystones = pd.Series(keystones,index=dropout_data.index)
# %%


train_samples = dropout_data.values

gksp = glv_keystone_estimator()

pred_keystones = cross_val_predict(gksp,X=train_samples,cv=3)
pred_keystones = pd.Series(pred_keystones,index=dropout_species)
plt.figure()

kbc_df = pd.DataFrame([keystones.values,pred_keystones[keystones.index].values],index=['True K$_{BC}$','Reconstructed K$_{BC}$'],columns=keystones.index).T
scatter_plot(kbc_df,'True vs. Reconstructed K$_{BC}$',point_label_size=8,color=colors)
plt.savefig('carlstrom_keystones.eps',format='eps',bbox_inches='tight')

corr,cov = SparCC(train_samples)
corr_net = (corr > 0.02)
corr_net = nx.from_numpy_array(corr_net)

plt.figure()
deg_df = pd.DataFrame([keystones.values,nx.degree_centrality(corr_net).values()],index=['True K$_{BC}$','Co-Occurrence Degree'],columns=keystones.index).T
scatter_plot(deg_df,'True K$_{BC}$ vs. Co-Occurrence Degree',point_label_size=8,color=colors)
plt.savefig('carlstrom_degree.eps',format='eps',bbox_inches='tight')

plt.figure()
bwn_df = pd.DataFrame([keystones.values,nx.betweenness_centrality(corr_net).values()],index=['True K$_{BC}$','Co-Occurrence Betweenness'],columns=keystones.index).T
scatter_plot(bwn_df,'True K$_{BC}$ vs. Co-Occurrence Betweenness',point_label_size=8,color=colors)
plt.savefig('carlstrom_betweenness.eps',format='eps',bbox_inches='tight')

A_pred,r_pred = reconstruct_from_ss(train_samples)
sc_score = self_consistency_score(train_samples,A_pred,r_pred)
print(f'Self Consistency Score: {sc_score}')
# %%
from generate_glv_simulations import random_training_samples, glv_simulator
perturbed = random_training_samples(A=A_pred,r=r_pred,n_train_samples=500)
perturbed = (perturbed.T/perturbed.sum(axis=1)).T
kbc_df = pd.DataFrame([pred_keystones[keystones.index].values,perturbed.mean(axis=0)],index=['Reconstructed K$_{BC}$','Mean Relative Abundance'],columns=keystones.index).T
scatter_plot(kbc_df,'Reconstructed K$_{BC}$ vs. Simulated MRA',point_label_size=8,color=colors)
plt.savefig('carlstrom_mra.eps',format='eps',bbox_inches='tight')
#%%
sim = glv_simulator(A_pred,r_pred)
pred_samples = np.zeros_like(train_samples)
for steady_state in range(train_samples.shape[0]):
    pred_samples[steady_state,:] = sim.ss_from_assemblage(train_samples[steady_state,:],compositional=True)
#%%

pred_errors = []
null_errors = []
fig = plt.figure(dpi=500,figsize=(9,6))
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
    plt.subplot(3,9,row+1)
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
plt.subplot(3,9,row+2)
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
plt.savefig('carlstrom_compositions.eps',format='eps',bbox_inches='tight')

pred_errors = pd.DataFrame(pred_errors)
pred_errors['Model Type'] = 'KeySDL'
null_errors = pd.DataFrame(null_errors)
null_errors['Model Type'] = 'Null Model'
errors = pd.concat([pred_errors,null_errors],axis=0)
plt.figure()
sns.boxplot(errors,y=0,hue='Model Type',x='Model Type')
plt.ylabel('Error (Bray-Curtis Distance)')
plt.title('Steady State Error')
plt.savefig('carlstrom_errors.eps',format='eps',bbox_inches='tight')
# %%
