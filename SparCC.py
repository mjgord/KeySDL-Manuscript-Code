#!/usr/bin/env python

'''
@author: Jonathan Friedman 

Converted for use in Python3 and refactored into a single file by mjgord (mjgordo3@nscu.edu) 
Module for estimating the correlations in the basis when only compositional data is available.

Original repository: https://bitbucket.org/yonatanf/sparcc/src/default/

The MIT License (MIT)

Copyright (c) 2018-2020 Jonathan Friedman and Eric Alm

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import warnings
import numpy as np
from numpy import (unravel_index, argmax, ones, corrcoef, cov, r_, 
                   diag, sqrt, where, nan, asarray,tile,array,zeros,log,var) 
from pandas import DataFrame as DF
try:
    from scipy.stats import nanmedian
except ImportError:
    from numpy import nanmedian

def _get_axis(given_axis):
    names0 = set([0,'0', 'rows','index','r'])
    names1 = set([1,'1','cols','columns','c'])
    ## to lower case
    if hasattr(given_axis, 'lower'):
        given_axis = given_axis.lower()
    ## get axis
    if given_axis is None:
        return None            
    elif given_axis in names0:
        axis = 0
    elif given_axis in names1:
        axis = 1
    else:
        raise ValueError('Unsupported axis "%s"' %given_axis)  
    return axis

def normalize(frame, axis=0):
    '''
    Normalize counts by sample total.
    
    Parameters
    ----------
    axis : {0, 1}
        0 : normalize each row
        1 : normalize each column

    Returns new instance of same class as input frame.
    '''
    axis = _get_axis(axis)
    tmp = np.apply_along_axis(lambda x:1.*x/x.sum(), 1-axis, frame)
    return DF(tmp)

def to_fractions(frame, method='dirichlet', p_counts=1, axis=0):
    '''
    Covert counts to fraction using given method.
    
    Parameters
    ----------
    method : string {'dirichlet' (default) | 'normalize' | 'pseudo'}
        dirichlet - randomly draw from the corresponding posterior 
                    Dirichlet distribution with a uniform prior.
                    That is, for a vector of counts C, 
                    draw the fractions from Dirichlet(C+1). 
        normalize - simply divide each row by its sum.
        pseudo    - add given pseudo count (defualt 1) to each count and
                    do simple normalization.
    p_counts : int/float (default 1)
        The value of the pseudo counts to add to all counts.
        Used only if method is dirichlet
    axis : {0 | 1}
        0 : normalize each row.
        1 : normalize each column.
    
    Returns
    -------
    fracs: frame/array
        Estimated component fractions.
        Returns new instance of same class as input frame.
    '''
    axis = _get_axis(axis)
    if method == 'normalize': 
        fracs = normalize(frame, axis)
        return fracs
    
    ## if method is not normalize, get the pseudo counts (dirichlet prior)
    from numbers import Number
    if not isinstance(p_counts, Number):
        p_counts = np.asarray(p_counts)
        
    if method == 'pseudo': 
        fracs = normalize(frame+p_counts, axis)
    elif method == 'dirichlet':
        from numpy.random.mtrand import dirichlet
        def dir_fun(x):
            a = x+p_counts
            f = dirichlet(a)
            return f
        fracs = np.apply_along_axis(dir_fun, 1-axis, frame)
        fracs = DF(fracs)
    else:
        raise ValueError('Unsupported method "%s"' %method)
    return fracs

def clr(frame, centrality='mean', axis=0):
    '''
    Do the central log-ratio (clr) transformation of frame.
    'centraility' is the metric of central tendency to divide by 
    after taking the logarithm.
    
    Parameters
    ----------
    centrality : 'mean' (default) | 'median'    
    axis : {0, 1}
        0 : transform each row (default)
        1 : transform each colum
    '''
    temp = log(frame)
    if   centrality == 'mean':   f = lambda x: x - x.mean()
    elif centrality == 'median': f = lambda x: x - x.median()
    if isinstance(frame, DF):
        z = temp.apply(f, axis=1-axis)
    else:
        z = np.apply_along_axis(f, 1-axis, temp)
    return z

def variation_mat(frame):
    '''
    Return the variation matrix of frame.
    Element i,j is the variance of the log ratio of components i and j.
    '''
    x    = 1.*asarray(frame)
    n,m  = x.shape
    if m > 1000:
        return variation_mat_slow(frame)
    else:
        xx   = tile(x.reshape((n,m,1)) ,(1,1,m))
        xx_t = xx.transpose(0,2,1)
        try:
            l    = log(1.*xx/xx_t)
            V    = l.var(axis=0, ddof=1)
            return V
        except MemoryError:
            return variation_mat_slow(frame) 
        
def variation_mat_slow(frame, shrink=False):
    '''
    Return the variation matrix of frame.
    Element i,j is the variance of the log ratio of components i and j.
    Slower version to be used in case the fast version runs out of memeory.
    '''
    print('in slow')
    frame_a = 1.*asarray(frame)
    k    = frame_a.shape[1]
    V      = zeros((k,k))
    for i in range(k-1):
        for j in range(i+1,k):
            y     = array(log(frame_a[:,i]/frame_a[:,j]))
            v = var(y, ddof=1) # set ddof to divide by (n-1), rather than n, thus getting an unbiased estimator (rather than the ML one). 
            V[i,j] = v
            V[j,i] = v
    return V
    

def append_indices(excluded,exclude):
    '''
    Append the indx of current excluded value to tuple of previously excluded values.
    '''
    if excluded is None: inds = exclude
    else:                inds = (r_[excluded[0],exclude[0]], r_[excluded[1],exclude[1]])
    return inds
    
def new_excluded_pair(C, th=0.1, previously_excluded=[]):
    '''
    Find component pair with highest correlation among pairs that 
    weren't previously excluded.
    Return the i,j of pair if it's correlaiton >= than th.
    Otherwise return None.
    '''
#    C_temp = abs(C - diag(diag(C)) )
    C_temp = np.triu(abs(C),1) # work only on upper triangle, excluding diagonal
    C_temp[tuple(zip(*previously_excluded))] = 0 
    i,j = unravel_index(argmax(C_temp), C_temp.shape) 
    cmax = C_temp[i,j]
    if cmax > th:
        return i,j
    else:  
        return None

def basis_var(f, Var_mat, M, **kwargs):
    '''
    Estimate the variances of the basis of the compositional data x.
    Assumes that the correlations are sparse (mean correlation is small).
    The element of V_mat are refered to as t_ij in the SparCC paper.
    '''
    ## compute basis variances
    try:    M_inv = np.linalg.inv(M)
    except: M_inv = np.linalg.pinv(M)
    V_vec  = Var_mat.sum(axis=1) # elements are t_i's of SparCC paper
    V_base = np.dot(M_inv, V_vec)  # basis variances. 
    ## if any variances are <0 set them to V_min
    V_min  = kwargs.get('V_min', 1e-10)
    V_base[V_base <= 0] = V_min 
    return V_base

def C_from_V(Var_mat, V_base):
    '''
    Given the estimated basis variances and observed fractions variation matrix, 
    compute the basis correlation & covaraince matrices.
    '''
    Vi, Vj = np.meshgrid(V_base, V_base)
    Cov_base = 0.5*(Vi + Vj - Var_mat)
    C_base = Cov_base/ sqrt(Vi) / sqrt(Vj)
    return C_base, Cov_base

def run_sparcc(f, **kwargs):
    '''
    Estimate the correlations of the basis of the compositional data f.
    Assumes that the correlations are sparse (mean correlation is small).
    '''
    th    = kwargs.get('th', 0.1)
    xiter = kwargs.get('xiter', 10)
    ## observed log-ratio variances
    Var_mat = variation_mat(f)
    Var_mat_temp = Var_mat.copy()
    ## Make matrix from eqs. 13 of SparCC paper such that: t_i = M * Basis_Varainces
    D = Var_mat.shape[0] # number of components
    M = ones((D,D)) + diag([D-2]*D) 
    ## get approx. basis variances and from them basis covariances/correlations 
    V_base = basis_var(f, Var_mat_temp, M)
    C_base, Cov_base = C_from_V(Var_mat, V_base)
    ## Refine by excluding strongly correlated pairs
    excluded_pairs = []
    excluded_comp  = np.array([])
    for xi in range(xiter):
        # search for new pair to exclude
        to_exclude = new_excluded_pair(C_base, th, excluded_pairs) #i,j pair, or None
        if to_exclude is None: #terminate if no new pairs to exclude
            break
        # exclude pair
        excluded_pairs.append(to_exclude)
        i,j = to_exclude
        M[i,j] -= 1
        M[j,i] -= 1
        M[i,i] -= 1
        M[j,j] -= 1
        inds = tuple(zip(*excluded_pairs))
        Var_mat_temp[inds]   = 0
        Var_mat_temp.T[inds] = 0
        # search for new components to exclude
        nexcluded = np.bincount(np.ravel(excluded_pairs)) #number of excluded pairs for each component
        excluded_comp_prev = set(excluded_comp.copy())
        excluded_comp      = where(nexcluded>=D-3)[0]
        excluded_comp_new  = set(excluded_comp) - excluded_comp_prev
        if len(excluded_comp_new)>0:
            print(excluded_comp)
            # check if enough components left 
            if len(excluded_comp) > D-4:
                warnings.warn('Too many component excluded. Returning clr result.')
                #return run_clr(f)
            for xcomp in excluded_comp_new:
                Var_mat_temp[xcomp,:] = 0
                Var_mat_temp[:,xcomp] = 0
                M[xcomp,:] = 0
                M[:,xcomp] = 0
                M[xcomp,xcomp] = 1
        # run another sparcc iteration
        V_base = basis_var(f, Var_mat_temp, M)
        C_base, Cov_base = C_from_V(Var_mat, V_base)
        # set excluded components infered values to nans
        for xcomp in excluded_comp:
            V_base[xcomp] = nan
            C_base[xcomp,:] = nan
            C_base[:,xcomp] = nan
            Cov_base[xcomp,:] = nan
            Cov_base[:,xcomp] = nan
    return V_base, C_base, Cov_base

def run_clr(f):
    '''
    Estimate the correlations of the compositional data f.
    Data is transformed using the central log ratio (clr) transform.
    '''
    z        = clr(f)
    Cov_base = cov(z, rowvar=0)
    C_base   = corrcoef(z, rowvar=0)
    V_base   = diag(Cov_base)
    return V_base, C_base, Cov_base
        
def basis_corr(f, method='sparcc', **kwargs):
    '''
    Compute the basis correlations between all components of 
    the compositional data f. 
    
    Parameters
    ----------
    f : array_like
        2D array of relative abundances. 
        Columns are counts, rows are samples. 
    method : str, optional (default 'SparCC')
        The algorithm to use for computing correlation.
        Supported values: SparCC, clr, pearson, spearman, kendall
        Note that the pearson, spearman, kendall methods are not
        altered to account for the fact that the data is compositional,
        and are provided to facilitate comparisons to 
        the clr and sparcc methods.

    Returns
    -------
    V_base: array
        Estimated basis variances.
    C_base: array
        Estimated basis correlation matrix.
    Cov_base: array
        Estimated basis covariance matrix.

    =======   ============ =======   ================================================
    kwarg     Accepts      Default   Desctiption
    =======   ============ =======   ================================================
    th        0<th<1       0.1       exclusion threshold for SparCC.
    xiter     int          10        number of exclusion iterations for SparCC.
    =======   ============ ========= ================================================
    '''    
    method = method.lower()
    k = f.shape[1]
    ## compute basis variances & correlations
    if k<4: 
        raise ValueError('Can not detect correlations between compositions of <4 components (%d given)' %k)     
    if method == 'clr':
        V_base, C_base, Cov_base = run_clr(f)
    elif method == 'sparcc':
        V_base, C_base, Cov_base = run_sparcc(f, **kwargs)
        tol = 1e-3 # tolerance for correlation range
        if np.max(np.abs(C_base)) > 1 + tol:
            warnings.warn('Sparcity assumption violated. Returning clr result.')
            V_base, C_base, Cov_base = run_clr(f)    
    else:
        raise ValueError('Unsupported basis correlation method: "%s"' %method)
    return V_base, C_base, Cov_base 

def SparCC(counts, method='SparCC', **kwargs):
    '''
    Compute correlations between all components of counts matrix.
    Run several iterations, in each the fractions are re-estimated, 
    and return the median of all iterations.
    Running several iterations is only helpful with 'dirichlet' 
    normalization method, as with other methods all iterations 
    will give identical results. Thus, if using other normalizations
    set 'iter' parameter to 1.
     
    Parameters
    ----------
    counts : DataFrame
        2D array of counts. Columns are components, rows are samples.
        If using 'dirichlet' or 'pseudo' normalization, 
        counts (positive integers) are required to produce meaningful results, 
        though this is not explicitly checked by the code.  
    method : str, optional (default 'SparCC')
        The algorithm to use for computing correlation.
        Supported values: SparCC, clr, pearson, spearman, kendall
        Note that the pearson, spearman, kendall methods are not
        altered to account for the fact that the data is compositional,
        and are provided to facilitate comparisons to 
        the clr and sparcc methods.

    Returns
    -------
    cor_med: array
        Estimated correlation values.
    cov_med: array
        Estimated covariance matrix if method in {SparCC, clr},
        None otherwise.
              
    =======   ============ =======   ================================================
    kwarg     Accepts      Default   Desctiption
    =======   ============ =======   ================================================
    iter      int          20        number of estimation iteration to average over.
    oprint    bool         True      print iteration progress?
    th        0<th<1       0.1       exclusion threshold for SparCC.
    xiter     int          10        number of exclusion iterations for sparcc.
    norm      str          dirichlet method used to normalize the counts to fractions.
    log       bool         True      log-transform fraction? used if method ~= SparCC/CLR
    =======   ============ ========= ================================================
    '''
    cor_list = []  # list of cor matrices from different random fractions
    var_list = []  # list of cov matrices from different random fractions
    oprint   = kwargs.pop('oprint',True)
    n_iter     = kwargs.pop('iter',20)  # number of iterations 
    norm     = kwargs.pop('norm','dirichlet')
    log      = kwargs.pop('log','True')
    th       = kwargs.setdefault('th',0.1)   # exclusion threshold for iterative sparse algo
    for i in range(n_iter):
        if oprint: print('\tRunning iteration' + str(i))
        fracs = to_fractions(counts, method=norm)
        v_sparse, cor_sparse, cov_sparse = basis_corr(fracs, method=method, **kwargs)
        var_list.append(np.diag(cov_sparse))
        cor_list.append(cor_sparse)
    cor_array = np.array(cor_list)
    var_med = nanmedian(var_list,axis=0) #median variances
    cor_med = nanmedian(cor_array,axis=0) #median correlations
    x,y     = np.meshgrid(var_med,var_med)
    cov_med = cor_med * x**0.5 * y**0.5
    return cor_med,cov_med
