import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme()
from scipy.stats import spearmanr

def scatter_plot(data,title=None,spearman=True,point_label_size=False,color=None):
    if color:
        n = int(np.ceil(data.shape[0]/4))

        for i in range(n):
            bottom = 4*i
            top = np.min([4*(i+1),data.shape[0]])
            shape_data = data.iloc[bottom:top,:]
            ax=sns.scatterplot(data=shape_data,x=shape_data.columns[0],y=shape_data.columns[1],s=30,style=shape_data.index)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',ncol = data.shape[0]//20+1)
    else:
        sns.scatterplot(data=data,x=data.columns[0],y=data.columns[1],s=15)
    if spearman:
        x = data.iloc[:,0]
        y = data.iloc[:,1]
        statistic = spearmanr(x,y).statistic
        pvalue = spearmanr(x,y).pvalue
        if pvalue < 0.01:
            pstring = f'p={pvalue:.1e}'
        else:
            pstring = f'p={pvalue:.2f}'
        plt.annotate(f'ρ= {statistic:.2f}, {pstring}',xy=(0.025,0.9),xycoords='axes fraction',fontsize=12)
    if title:
        plt.title(title,fontsize=12)
    if point_label_size and not color:
        for i,txt in enumerate(data.index):
            x = data.iloc[:,0]
            y = data.iloc[:,1]
            plt.annotate(txt,(x.iloc[i],y.iloc[i]),fontsize=point_label_size)
