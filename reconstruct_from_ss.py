import numpy as np
from generate_glv_simulations import glv_simulator, initialize_glv
import torch.nn as nn
import torch

class ss_optim(nn.Module):
    def __init__(self,n,A=None,r=None,compositional=False):
        super().__init__()
        if A is None:
            # A must have negative diagonal to represent finite carrying capacity
            A_init=-torch.eye(n)
            # A will be mostly negative so start it on that side of 0
            A_init = torch.where(A_init == 0,-1e-6,A_init)
            self.A = nn.Parameter(A_init)
        else: # allow passing of starting value of A
            self.A = nn.Parameter(A)
        if r is None:
            self.r = nn.Parameter(torch.ones(n))
        else: # allow passing of starting value of r
            self.r = nn.Parameter(r)
        self.compositional=compositional
        
    def forward(self,x,x_mask):
        if not self.compositional:
            z = self.fun(x)
        else:
            x = x/torch.sum(x)
            f = self.fun(x)
            theta = torch.sum(x*f)
            z = f - theta
        # masking to prevent gradient updates for extinct species
        return (z*x_mask, self.A)
    
    def fun(self,x):
        if not self.compositional:
            return self.r+torch.matmul(self.A,x.T).T
        else:
            return torch.matmul(self.A,x.T).T

class ss_optim_loss(nn.Module):
    def __init__(self,alpha=1e-15):
            super().__init__()
            self.alpha = alpha
    def forward(self,ss_residual,A):
        det = torch.abs(torch.det(A)) # determinant term to enforce invertibility
        det_loss = torch.where(det < 0.1,1/det,0)
        diag = torch.diag(A) # diagonal to prevent self-influence from becoming positive
        diag_loss = 1e6*torch.norm(torch.where(diag > -0.1,diag+0.1,0),p=1)
        l1 = self.alpha * torch.norm(A,p=1) # l1 norm to enforce sparsity
        return (torch.norm(ss_residual,p=2) + diag_loss + det_loss + l1)

# %%
def reconstruct_from_ss(X,
                        compositional=True,
                        max_iter=10000, 
                        lr=1e-3, 
                        alpha=1e-15, 
                        batch_size=32, 
                        A_init=None, 
                        r_init=None, 
                        verbose=False,):
    '''
    The core reconstruction function of KeySDL. Finds the GLV or replicator model that best explains the observed steady states.

    Parameters
    ----------
    X : array-like of shape (steady states, features)
        Feature values (e.g.microbial abundances) of observed steady states.
    compositional: bool, default = True
        Whether to model as GLV (False) or replicator (True). Default value is True because most experimental datasets are inherently compositional.
    max_iter : int, default = 10000
        Number of gradient descent iterations used. This should rarely require adjustment.
    lr : float, default = 1e-3
        Gradient descent learning rate. This should rarely require adjustment.
    alpha : float, default = 1e-15
        L1 penalty. This should rarely require adjustment.
    batch_size : int, default = 32
        Gradient descent batch size. This should rarely require adjustment.
    A_init: array-like of shape (features,features), default = None
        Initial value for interactions matrix A, default value of None will initialize with the identity matrix.
    r_init: array-like of shape (features,), default = None
        Initial value for growth rates r, default value of None will initialize with ones
    
    '''
    X = torch.from_numpy(X).float()
    X_mask= torch.where(X==0,0,1).float() # mask to prevent gradient updates to extinct species
        
    model = ss_optim(n=X.shape[1],compositional=compositional,A=A_init,r=r_init)
    loss_fn = ss_optim_loss(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    for i in range(max_iter):
        optimizer.zero_grad()
        batch_idx = torch.randperm(X.shape[0])[:batch_size]
        pred,coef = model(X[batch_idx,:],X_mask[batch_idx,:])
        loss = loss_fn(pred,coef)
        loss.backward()
        optimizer.step()
        if verbose:
            print(f'Train Loss: {loss.item()}')

    A_pred = model.A.cpu().detach().numpy()
    r_pred = model.r.cpu().detach().numpy()
    return A_pred, r_pred
    
def self_consistency_score(data,A,r,compositional=True):
    sim = glv_simulator(A=A,r=r)
    pred_ss = np.zeros_like(data)
    for steady_state in range(data.shape[0]):
        pred_ss[steady_state,:] = sim.ss_from_assemblage(data[steady_state,:],compositional=compositional)
    bcd = np.sum(np.abs(pred_ss-data),axis=1)/np.sum(np.abs(pred_ss+data),axis=1)
    return 1-np.mean(bcd)
#%%

# dummy estimator to wrap reconstruction for cross validation
# X needs to be indices of data rather than data itself
# This is to keep track of indexing for which keystones to return
from sklearn.base import BaseEstimator
class glv_composition_estimator(BaseEstimator):
    def __init__(self,compositional=True,alpha=1e-15):
        self.A_pred = 0
        self.r_pred = 0
        self.pred_sim = []
        self.compositional=compositional
        self.alpha=alpha

    def fit(self,X,y=0):
        A_pred,r_pred = reconstruct_from_ss(X, compositional=self.compositional,max_iter=10000,alpha=self.alpha,verbose=False)
        self.A_pred = A_pred
        self.r_pred = r_pred
        self.pred_sim = glv_simulator(A=self.A_pred,r=self.r_pred)

    def predict(self,X):
        pred_comps = []
        for row in range(X.shape[0]):
            pred_comps.append(self.pred_sim.ss_from_assemblage(X[row,:],compositional=self.compositional))
        return pred_comps


class glv_keystone_estimator(glv_composition_estimator):
    def predict(self,X):
        baseline = self.pred_sim.ss_from_assemblage(np.ones(X.shape[1]),compositional=self.compositional)
        keystones = []
        for row in range(X.shape[0]):
            pred_comp = self.pred_sim.ss_from_assemblage(X[row,:],compositional=self.compositional)
            keystones.append(np.sum(np.abs(pred_comp-baseline))/np.sum(np.abs(pred_comp+baseline)))
        return np.array(keystones)
