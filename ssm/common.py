# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.linalg import block_diag as blkdiag

class ssmat(dict):
    """ Represents a state space matrix.
    Attributes
    ----------
    transform : 
    linear : 
    gaussian : 
    dynamic : 
    constant : 
    func : 
    nparam : 
    mat : 
    """
    def __init__(self, **kwargs):
        self['transform']  = kwargs['transform']
        if self['transform']:
            self['linear']  = True
        else:
            self['gaussian']  = True
        self['constant']  = kwargs['constant']
        self['dynamic']   = kwargs['dynamic']
        if self['constant']:
            if self['dynamic']:
                self['mat']   = [np.mat(x) for x in kwargs['mat']]
                self['shape'] = self['mat'][0].shape + (len(self['mat']),)
            else:
                self['mat']   = np.mat(kwargs['mat'])
                self['shape'] = self['mat'].shape
        else:
            self['shape']  = kwargs['shape']
            self['func']   = kwargs['func']
            self['nparam'] = kwargs['nparam']

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        props  = ('transform','linear' if self.transform else 'gaussian','dynamic','constant','shape') + (('mat',) if self.constant else ('func','nparam') + (('mat',) if 'mat' in self else ()))
        m  = max(map(len, list(props))) + 1
        return '\n'.join([x.rjust(m) + ': ' + repr(self[x]).replace('\n','\n'+' '*(m+2)) for x in props])

    def __dir__(self):
        return list(self.keys())

    def __nonzero__(self):
        if 'transform' not in self: return False
        if self.transform:
            if 'linear' not in self: return False
        elif 'gaussian' not in self: return False
        if 'dynamic' not in self: return False
        if 'constant' not in self: return False
        if 'shape' not in self: return False
        if self.constant:
            if 'mat' not in self: return False
            m  = self.mat
        else:
            if 'func' not in self: return False
            if self.func is None:
                if 'mat' not in self: return False
                m  = self.mat
            elif not callable(self.func): return False
            else:
                if 'nparam' not in self: return False
                m  = self.func([0.0]*self.nparam)
        if self.dynamic:
            if len(self.shape) < 3: return False
            if type(m) != list: return False
            if self.shape[2] != len(m): return False
            for i in range(len(m)):
                if type(m[i]) != np.matrix: return False
                if m[i].shape[0] != self.shape[0] or m[i].shape[1] != self.shape[1]: return False
        else:
            if len(self.shape) > 2 and self.shape[2] != 1: return False
            if type(m) != np.matrix: return False
            if m.shape[0] != self.shape[0] or m.shape[1] != self.shape[1]: return False
        return True

#---------------------------------------------------#
#-- Functions for State Space Matrix Construction --#
#---------------------------------------------------#
def mat_const(mat,dynamic=False,trans=True):
    """Construct a constant state space matrix.

    dynamic --
    trans -- True if this is a "transform" matrix (Z,T,R,c,a1), False if this is a "distribution" matrix (H,Q,P1)
    """
    return ssmat(transform=trans, constant=True, dynamic=dynamic, mat=mat)

def f_psi_to_cov(p):
    """Returns a function that generates a (full) covariance matrix from a standard parametrization vector psi
    The covariance matrix is (p, p)
    The expected parameter vector psi is (p*(p+1)/2,)
    """
    mask  = np.nonzero(np.tril(np.ones((p,p),dtype=bool) & ~np.eye(p,dtype=bool)))
    def psi_to_cov(x):
        # bound variables: p,mask
        x1  = np.asarray(x[:p])
        x2  = np.asarray(x[p:])
        Y   = np.exp(x1)[:,None].T
        Y   = Y.T * Y
        C   = np.zeros((p,p))
        C[mask] = Y[mask] * (x2/np.sqrt(1 + x2**2))
        C   = C + C.T + np.diag(np.diag(Y))
        return np.matrix(C)
    return psi_to_cov

def mat_var(p=1,cov=True):
    """Create a parametrized normal covariance matrix for use as state space matrix
    p is the number of variables.
    cov specifies complete covariance if true, or complete independence if false.
    """
    #-- Construct function to generate model --#
    if p == 1:
        return ssmat(transform=False, dynamic=False, constant=False,
            shape=(1,1), func=lambda x: np.mat(np.exp(2*x[0])), nparam=1)
    elif not cov:
        return ssmat(transform=False, dynamic=False, constant=False,
            shape=(p,p), func=lambda x: np.mat(np.diag(np.exp(2*np.asarray(x)))),
            nparam=p)
    else:
        return ssmat(transform=False, dynamic=False, constant=False,
            shape=(p,p), func=f_psi_to_cov(p), nparam=p*(p+1)//2)

def mat_dupvar(p, d, cov=True):
    """
    Each of the p variables have a single variance duplicated d times
    cov = True indicates that there are covariances between the p variables, each of which are also duplicated d times
    """
    if d == 1: return mat_var(p, cov)

    if cov:
        mask   = np.nonzero(np.tril(np.ones((p,p),dtype=bool) & ~np.eye(p,dtype=bool))) # mask for a single full covariance matrix
        W      = np.matrix(np.eye(d))
        def psi_to_dup_cov(x):
            # bound variables: p, mask, W
            x1  = np.asarray(x[:p])
            x2  = np.asarray(x[p:])
            Y   = np.exp(x1)[:,None].T
            Y   = Y.T * Y
            C   = np.zeros((p,p))
            C[mask] = Y[mask] * (x2/np.sqrt(1 + x2**2))
            C   = C + C.T + np.diag(np.diag(Y))
            return np.kron(C, W)

        return ssmat(transform=False, dynamic=False, constant=False,
            shape=(p*d,)*2, func=psi_to_dup_cov, nparam=p*(p+1)//2)
    else:
        return ssmat(transform=False, dynamic=False, constant=False,
            shape=(p*d,)*2, nparam=p,
            func=lambda x: np.mat(np.diag(np.repeat(np.exp(2*np.asarray(x)),d))))

def mat_interlvar(p, q, cov):
    """Create state space matrix for q-interleaved variance noise.
    p is the number of variables.
    q is the number of variances affecting each variable.
    cov is a logical vector that specifies whether each q variances covary
       across variables. shape = (q,)
    The variances affecting any single given variable is always assumed to be
       independent.
    """

    if p == 1: return mat_var(q, False)

    mask   = np.nonzero(np.tril(np.ones((p,p),dtype=bool) & ~np.eye(p,dtype=bool))) # mask for a single full covariance matrix
    Vmask  = [None]*q # individual masks into the whole interleaved variance matrix for each of the q variances
    nparam = 0
    for j in range(q):
        emask       = np.zeros((q,q),dtype=bool)
        emask[j,j]  = True
        Vmask[j]    = np.kron(np.ones((p,p),dtype=bool) if cov[j] else np.eye(p,dtype=bool), emask)
        nparam     += p*(p+1)//2 if cov[j] else p

    def psi_to_interlvar(x):
        # bound variables: p, q, cov, mask, Vmask
        i  = 0 # pointer into x
        V  = np.zeros((p*q,)*2)
        for j in range(q):
            if cov[j]:
                xj  = x[i : i + (p*(p+1)/2)]
                i  += p*(p+1)/2
                # Generate the covariance matrix for the qth variance across all p variables
                x1  = np.asarray(xj[:p])
                x2  = np.asarray(xj[p:])
                Y   = np.exp(x1)[:,None].T
                Y   = Y.T * Y
                C   = np.zeros((p,p))
                C[mask] = Y[mask] * (x2/np.sqrt(1 + x2**2))
                Vj  = C + C.T + np.diag(np.diag(Y))
            else:
                xj  = x[i : i + p]
                i  += p
                Vj  = np.diag(np.exp(2*np.asarray(xj)))
            V[Vmask[j]]  = Vj

        return V

    return ssmat(transform=False, dynamic=False, constant=False,
        shape=(p*q,)*2, func=psi_to_interlvar, nparam=nparam)

def func_stat_to_dyn(func,n):
    """helper function for mat_cat()"""
    # Nested functions inside a function can be defined multiple times, but any outside variables "bound" into the function will take the last value at the outer function exit, making multiple function definitions equivalent ...
    return lambda x: [func(x)]*n

def mat_cat(mode,mats):
    """Concatenate state space matrices
    mode : 'h' for horizontal concatenation, 'v' for vertical concatenation, and 'd' for diagonal concatenation
    mats : list of state space matrices
    """
    N        = len(mats)
    trans    = mats[0].transform
    dynamic  = any([mats[i].dynamic for i in range(N)])
    n        = max([mats[i].shape[2] if mats[i].dynamic else 1 for i in range(N)])
    constant_l  = [mats[i].constant for i in range(N)]
    constant    = all(constant_l)
    if not constant:
        nparam_l  = [mats[i].nparam if not mats[i].constant else 0 for i in range(N)]
        nparam    = sum(nparam_l)
        nparam_l  = np.cumsum([0] + nparam_l)
    if mode == 'h':
        mstack  = lambda x: np.asmatrix(np.hstack(x))
        shape   = mats[0].shape[0], sum([mats[i].shape[1] for i in range(N)])
    elif mode == 'v':
        mstack  = lambda x: np.asmatrix(np.vstack(x))
        shape   = sum([mats[i].shape[0] for i in range(N)]), mats[0].shape[1]
    else: # mode == 'd'
        mstack  = lambda x: np.asmatrix(blkdiag(*x))
        shape   = sum([mats[i].shape[0] for i in range(N)]), sum([mats[i].shape[1] for i in range(N)])
    if dynamic: shape += (n,)

    # Make all models dynamic if one is dynamic, and collapse entries into either matrix or function
    for i in range(N):
        if mats[i].constant:
            if dynamic and not mats[i].dynamic:
                mats[i]  = [mats[i].mat]*n
            else:
                mats[i]  = mats[i].mat
        else: # not mats[i].constant
            if dynamic and not mats[i].dynamic:
                mats[i]  = func_stat_to_dyn(mats[i].func,n) # a helper function must be used here to prevent i being "bound" to the last value in the current function scope, which would result in a list of identical functions
            else:
                mats[i]  = mats[i].func

    if constant:
        if dynamic: mats  = [mstack([mats[i][t] for i in range(N)]) for t in range(n)]
        else:       mats  = mstack([mats[i] for i in range(N)])
    else: # not constant
        func_mask  = np.nonzero(~np.asarray(constant_l))[0]
        mats1      = list(mats) # make a shallow copy to store realizations (w.r.t. some model parameter values)
        if dynamic:
            def mcat_func(x):
                # bound variables: func_mask, mats1, mats, nparam_l, mstack, N, n
                for i in func_mask:
                    mats1[i]  = mats[i](x[nparam_l[i]:nparam_l[i+1]]) # mats stores the function permanently, while the corresponding entry in mats1 stores the current realization
                return [mstack([mats1[i][t] for i in range(N)]) for t in range(n)]
        else:
            def mcat_func(x):
                # bound variables: func_mask, mats1, mats, nparam_l, mstack, N
                for i in func_mask:
                    mats1[i]  = mats[i](x[nparam_l[i]:nparam_l[i+1]]) # mats stores the function permanently, while the corresponding entry in mats1 stores the current realization
                return mstack([mats1[i] for i in range(N)])

    M  = {'transform': trans, 'dynamic': dynamic, 'constant': constant,
        'shape': shape}
    if constant:
        M['mat']    = mats
    else:
        M['func']   = mcat_func
        M['nparam'] = nparam
    return ssmat(**M)

class ssmodel(dict):
    """ Represents a state space model.
    Attributes
    ----------
    H :
    Z :
    T :
    R :
    Q :
    c :
    a1 :
    P1 :
    """
    def __init__(self,**kwargs):
        """Construct a single component state space model from provided state space matrices.
        keyword arguments : 'H','Z','T','R','Q','c','a1','P1', and optionally 'A' pointing to individual state space matrices
        """
        self['H']   = kwargs['H']
        self['Z']   = kwargs['Z']
        self['T']   = kwargs['T']
        self['R']   = kwargs['R']
        self['Q']   = kwargs['Q']
        self['c']   = kwargs['c']
        self['a1']  = kwargs['a1']
        self['P1']  = kwargs['P1']
        if 'A' in kwargs:  self['A'] = kwargs['A']
        if not self: raise TypeError('invalid combination of model matrices for model construction')
        self['dynamic']  = np.any([self[M]['dynamic'] for M in ('H','Z','T','R','Q','c','a1','P1')])
        self['n']  = min([self[M].shape[2] for M in ('H','Z','T','R','Q','c','a1','P1') if self[M].dynamic]) if self['dynamic'] else np.inf
        self['p']  = self['Z'].shape[0]
        self['m']  = self['T'].shape[0]
        self['r']  = self['R'].shape[1]
        self['nparam'] = sum([self[M].nparam for M in ('H','Z','T','R','Q','c','a1','P1') if not self[M].constant]) + (self['A']['nparam'] if 'A' in self else 0)
        self['mcom']   = [self['m']] # models built from ssmat constructor are considered a single "component"

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
                
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        props  = ('dynamic','n','p','m','r','nparam','mcom') + (('A',) if 'A' in self else ()) + ('H','Z','T','R','Q','c','a1','P1')
        m  = max(map(len, list(props))) + 1
        return '\n'.join([x.rjust(m) + ((':\n' + ' '*(m+2)) if type(self[x])==ssmat else ': ') + repr(self[x]).replace('\n','\n'+' '*(m+2)) for x in props])

    def __dir__(self):
        return list(self.keys())

    def __nonzero__(self):
        if 'A' in self:
            if 'target' not in self.A: return False
            if 'func' not in self.A: return False
            if 'nparam' not in self.A: return False
            m  = self.A['func']([0.0]*self.A['nparam'])
            for j in range(len(m)):
                self[self.A['target'][j]].mat  = m[j]

        MM  = ('H', 'Z', 'T', 'R', 'Q', 'c', 'a1', 'P1')
        for M in MM:
            if M not in self: return False
            if not self[M]: return False
        return True

#--------------------------------------------------#
#-- Functions for State Space Model Construction --#
#--------------------------------------------------#
def model_cat(models):
    """Concatenate state space models
    models -- a list of state space models
    Each model is considered a separate set of components
    The final H matrix is taken from models[0]
    Concatenation of models with 'A' is not supported yet
    """
    N  = len(models)
    final_model       = {}
    final_model['H']  = models[0].H
    final_model['Z']  = mat_cat('h',[models[i].Z for i in range(N)])
    final_model['T']  = mat_cat('d',[models[i].T for i in range(N)])
    final_model['R']  = mat_cat('d',[models[i].R for i in range(N)])
    final_model['Q']  = mat_cat('d',[models[i].Q for i in range(N)])
    final_model['c']  = mat_cat('v',[models[i].c for i in range(N)])
    final_model['a1'] = mat_cat('v',[models[i].a1 for i in range(N)])
    final_model['P1'] = mat_cat('d',[models[i].P1 for i in range(N)])
    final_model = ssmodel(**final_model)
    final_model.mcom = [m for i in range(N) for m in models[i].mcom]
    return final_model

#--------------------------------#
#-- Function for Data Analysis --#
#--------------------------------#
def prepare_data(y):
    # y is a 2D matrix n*p, missing data is currently not supported for 3D (batch mode)
    p,n     = y.shape
    mis     = np.asarray(np.isnan(y))
    anymis  = np.any(mis,0)
    allmis  = np.all(mis,0)
    y       = np.asmatrix(y) # asmatrix may or may not make a copy
    return n, p, y, mis, anymis, allmis

def prepare_mat(M,n):
    return M.mat if M.dynamic else [M.mat]*n

def prepare_model(model,n):
    H   = prepare_mat(model.H,n)
    Z   = prepare_mat(model.Z,n)
    T   = prepare_mat(model.T,n)
    R   = prepare_mat(model.R,n)
    Q   = prepare_mat(model.Q,n)
    c   = prepare_mat(model.c,n)
    a1  = model.a1.mat
    P1  = model.P1.mat
    RQdyn       = model.R.dynamic or model.Q.dynamic
    stationary  = not (model.H.dynamic or model.Z.dynamic or model.T.dynamic or RQdyn) # c does not effect convergence of P
    return H, Z, T, R, Q, c, a1, P1, stationary, RQdyn

def set_param(model,x):
    # The model is modified inplace, but reference returned for convenience
    i  = 0
    if 'A' in model:
        target  = model.A['target']
        nparam  = model.A['nparam']
        m       = model.A['func'](x[:nparam])
        for j in range(len(m)):
            model[target[j]].mat  = m[j]
        i       = nparam
    for M in ('H','Z','T','R','Q','c'):
        if not model[M].constant and model[M].func is not None:
            nparam  = model[M].nparam
            model[M].mat = model[M].func(x[i:i+nparam])
            i  += nparam
    return model
