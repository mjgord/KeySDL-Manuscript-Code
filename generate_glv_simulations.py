import numpy as np

class glv_simulator:
    def __init__(self,A,r):
        self.A = A
        self.r = r
        return

    def ss_from_assemblage(self,assemblage,compositional=True):
        # performs perturbation calculations
        # assemblage > 0 where a microbe is present
        n = self.A.shape[0]
        result = np.zeros(n)
        num_idx = np.argwhere(assemblage > 0)
        while True:
            A_p = self.A[num_idx,:].reshape(len(num_idx),n)
            A_p = A_p[:,num_idx].reshape(len(num_idx),len(num_idx))
            r_p = self.r[num_idx]
            A_p_i = np.linalg.inv(A_p)
            result[:] = 0
            result[num_idx] = -np.matmul(A_p_i,r_p)
            if not (result < 0).any():
                break
            else:
                num_idx = num_idx[np.where(result[num_idx] > 0)[0]]
        if compositional:
            return result/np.sum(result)
        else:
            return result
    
    def compute_dropouts(self,compositional=True):
        n = self.A.shape[0]
        # compute baseline
        baseline = self.ss_from_assemblage(np.ones(n))
        dropout = np.zeros((n,n))
        for i in range(n):
            dropout[i,:] = self.ss_from_assemblage(np.arange(n)!=i)
        if compositional:
            return baseline/np.sum(baseline),(dropout.T/np.sum(dropout,axis=1)).T
        else:
            return baseline,dropout
        
    def bcd_keystones(self):
            baseline,dropout = self.compute_dropouts(compositional=True) # bcd operates on compositions
            baseline = np.repeat(baseline.reshape(1,-1),dropout.shape[1],axis=0)
            baseline = baseline*(dropout!=0)
            return np.sum(np.abs(dropout-baseline),axis=1)/np.sum(np.abs(dropout+baseline),axis=1)

def random_training_samples(A,r,n_train_samples,p_zero_train=0.1, seed=None):
    n = A.shape[0]
    rng = np.random.default_rng(seed=seed)
    train_samples = np.where(rng.random((n_train_samples,n)) < p_zero_train,0.,1.)

    for i,idx in enumerate(train_samples):
        idx = np.array(idx)
        if sum(idx) != 0:
            num_idx = np.argwhere(idx).reshape(-1)
            sub_A = A[num_idx,:]
            sub_A = sub_A[:,num_idx]
            sub_A_i = np.linalg.inv(sub_A)
            sub_r = r[num_idx]
            ss = -np.matmul(sub_A_i,sub_r)
            train_samples[i,:] = 0
            train_samples[i,num_idx] = ss
    return train_samples


#%%

def klemm_degree(graph):
    return(np.sum((graph!=0)|(graph.T!=0),axis=0))

def generate_klemm_net(n,mu=0.01,m=5,conn_sigma=0.15,conn_mu=0,seed=None):
    rng = np.random.default_rng(seed=seed)
    if n <= m:
        print('too few nodes. increase number of nodes or decrease clique size.')
        return
    # shuffled order
    node_idx = np.arange(n)
    rng.shuffle(node_idx)
    # empty graph matrix
    graph = np.zeros((n,n))
    # accounting vector to keep track of active nodes
    active = np.zeros(n)
    # generate fully-connected clique
    m_idx = node_idx[:m]
    for i in m_idx:
        for j in m_idx:
            if i != j:
                graph[i,j] = 1
                graph[j,i] = 1
    # mark nodes in clique as active
    active[m_idx]=1
    for node in node_idx[m:]:
        # add node by sending connection to each active/selected node
        # mu chance to connect to a random node instead
        sel_mat = np.random.rand(m)
        selected = active.copy()
        selected[selected !=0] = sel_mat > mu
        n_rand = np.sum(sel_mat < mu)
        # random node selection
        if n_rand != 0:
            degree = klemm_degree(graph)
            degree = degree*(active==0) # disallow selection of an active node
            if np.sum(degree)==0: # if no degree to weight by, choose uniformly
                degree[:] = 1
            if np.sum(degree != 0) < n_rand: # if not enough nonzero, give some weight to all non-active
                degree = degree + 0.01*(active == 0)
            addl_sel = rng.choice(n,size=n_rand,p=degree/np.sum(degree),replace=False)
            selected[addl_sel] = 1
        # apply selection to network
        graph[node,selected!=0] = 1
        # deactivate existing node, inversely proportional to degree
        degree = klemm_degree(graph)
        degree[active==0] = 1 # avoid division by zero
        deactivation_p = (1/degree)
        deactivation_p[active==0] = 0 # discard non-active nodes
        deactivation_p = deactivation_p/np.sum(deactivation_p)
        active[rng.choice(n,size=1,p=deactivation_p)] = 0
        # activate current node
        active[node] = 1
    # applying interaction strengths
    graph = graph * rng.normal(loc=conn_mu,scale=conn_sigma,size=(n,n))
    # stabilizing by shifting diag to -1 (equivalent to logistic population limit)
    d = -1
    return graph+d*np.eye(n)

def initialize_glv(n,seed=None,r=None,**kwargs):
    state_center = np.array(-1)
    rng = np.random.default_rng(seed=seed)
    while (state_center < 0).any():
        if r is None:
            r = rng.random(n)
        A = generate_klemm_net(n=n,seed=seed, **kwargs)
        if seed:
            seed += 1 # ensure that repeats get fresh but repeatable random numbers
        sim = glv_simulator(A=A,r=r)
        A_i = np.linalg.inv(A)
        state_center = -np.matmul(A_i,r)
    return A,r,state_center
