# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import block_diag as blkdiag

#---------------------------------------------------#
#-- Functions for State Space Matrix Construction --#
#---------------------------------------------------#
def mat_const(mat,dynamic=False,trans=True):
    """Construct a constant state space matrix.

    dynamic --
    trans -- True if this is a "transform" matrix (Z,T,R,c,a1), False if this is a "distribution" matrix (H,Q,P1)
    """
    if dynamic:
        mat  = [np.asmatrix(x) for x in mat]
    else:
        mat  = np.asmatrix(mat)
    if trans:
        prop_name  = 'linear'
    else:
        prop_name  = 'gaussian'
    return {
        trans:      True,
        'dynamic':  type(mat) == list,
        'constant': True,
        'shape':    mat[0].shape + (len(mat),) if type(mat) == list else mat.shape,
        'mat':      mat}

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
        return {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': (1,1),
            'func': lambda x: np.asmatrix(np.exp(2*x[0])),
            'nparam': 1}
    elif not cov:
        return {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': (p,p),
            'func': lambda x: np.matrix(np.diag(np.exp(2*np.asarray(x)))),
            'nparam': p}
    else:
        return {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': (p,p),
            'func': f_psi_to_cov(p),
            'nparam': p*(p+1)/2}

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

        return {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': (p*d,)*2,
            'func': psi_to_dup_cov,
            'nparam': p*(p+1)/2}
    else:
        return {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': (p*d,)*2,
            'func': lambda x: np.asmatrix(np.diag(np.repeat(np.exp(2*np.asarray(x)),d))),
            'nparam': p}

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
        nparam     += p*(p+1)/2 if cov[j] else p

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

    return {
        'gaussian': True,
        'dynamic':  False,
        'constant': False,
        'shape': (p*q,)*2,
        'func': psi_to_interlvar,
        'nparam': nparam}

def func_stat_to_dyn(func,n):
    """helper function for mat_cat()"""
    # Nested functions inside a function can be defined multiple times, but any outside variables "bound" into the function will take the last value at the outer function exit, making multiple function definitions equivalent ...
    return lambda x: [func(x)]*n

def mat_cat(M,mats):
    """Concatenate state space matrices
    M -- one of 'H','Z','T','R','Q','c','a1','P1'
    mats -- list of state space matrices
    """
    N        = len(mats)
    dynamic  = any([mats[i]['dynamic'] for i in range(N)])
    n        = max([mats[i]['shape'][2] if mats[i]['dynamic'] else 1 for i in range(N)])
    constant_l  = [mats[i]['constant'] for i in range(N)]
    constant    = all(constant_l)
    if not constant:
        nparam_l  = [mats[i]['nparam'] if not mats[i]['constant'] else 0 for i in range(N)]
        nparam    = sum(nparam_l)
        nparam_l  = np.cumsum([0] + nparam_l)
    if M == 'Z':
        mstack  = lambda x: np.asmatrix(np.hstack(x))
        shape   = mats[0]['shape'][0], sum([mats[i]['shape'][1] for i in range(N)])
    elif M in ('c','a1'):
        mstack  = lambda x: np.asmatrix(np.vstack(x))
        shape   = sum([mats[i]['shape'][0] for i in range(N)]), mats[0]['shape'][1]
    else:
        mstack  = lambda x: np.asmatrix(blkdiag(*x))
        shape   = sum([mats[i]['shape'][0] for i in range(N)]), sum([mats[i]['shape'][1] for i in range(N)])
    if dynamic: shape += (n,)

    # Make all models dynamic if one is dynamic, and collapse entries into either matrix or function
    for i in range(N):
        if mats[i]['constant']:
            if dynamic and not mats[i]['dynamic']:
                mats[i]  = [mats[i]['mat']]*n
            else:
                mats[i]  = mats[i]['mat']
        else: # not mats[i]['constant']
            if dynamic and not mats[i]['dynamic']:
                mats[i]  = func_stat_to_dyn(mats[i]['func'],n) # a helper function must be used here to prevent i being "bound" to the last value in the current function scope, which would result in a list of identical functions
            else:
                mats[i]  = mats[i]['func']

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

    if M in ('H','Q','a1','P1'): # "Distribution" matrices
        M  = {'gaussian': True}
    else: # M in ('Z','T','R','c'), the "transform" matrices
        M  = {'linear':   True}
    M['dynamic']    =  dynamic
    M['constant']   = constant
    M['shape']      =    shape
    if constant:
        M['mat']    = mats
    else:
        M['func']   = mcat_func
        M['nparam'] = nparam
    return M

#--------------------------------------------------#
#-- Functions for State Space Model Construction --#
#--------------------------------------------------#
def validate_model(model):
    if 'A' in model:
        if 'target' not in model['A']: return False
        if 'func' not in model['A']: return False
        if 'nparam' not in model['A']: return False
        m  = model['A']['func']([0.0]*model['A']['nparam'])
        for j in range(len(m)):
            model[model['A']['target'][j]]['mat']  = m[j]

    MM  = ('H', 'Z', 'T', 'R', 'Q', 'c', 'a1', 'P1')
    for M in MM:
        if M not in model: return False
        if 'dynamic' not in model[M]: return False
        if 'constant' not in model[M]: return False
        if 'shape' not in model[M]: return False
        if model[M]['constant']:
            if 'mat' not in model[M]: return False
            m  = model[M]['mat']
        else:
            if 'func' not in model[M]: return False
            if model[M]['func'] is None:
                if 'mat' not in model[M]: return False
                m  = model[M]['mat']
            elif not callable(model[M]['func']): return False
            else:
                if 'nparam' not in model[M]: return False
                m  = model[M]['func']([0.0]*model[M]['nparam'])
        if model[M]['dynamic']:
            if len(model[M]['shape']) < 3: return False
            if type(m) != list: return False
            if model[M]['shape'][2] != len(m): return False
            for i in range(len(m)):
                if type(m[i]) != np.matrix: return False
                if m[i].shape[0] != model[M]['shape'][0] or m[i].shape[1] != model[M]['shape'][1]: return False
        else:
            if len(model[M]['shape']) > 2 and model[M]['shape'][2] != 1: return False
            if type(m) != np.matrix: return False
            if m.shape[0] != model[M]['shape'][0] or m.shape[1] != model[M]['shape'][1]: return False
    return True

def model_from_mats(H_or_model,Z=None,T=None,R=None,Q=None,c=None,a1=None,P1=None,A=None):
    """Construct a single component state space model from provided state space matrices
    model_from_mats(model) or model_from_mats(H,Z,T,R,Q,c,a1,P1)
    model -- A dictionary with keys 'H','Z','T','R','Q','c','a1','P1' pointing to individual state space matrices
    """
    if Z is None:  model = H_or_model
    else:
        model = {'H': H_or_model, 'Z': Z, 'T': T, 'R': R, 'Q': Q, 'c': c, 'a1': a1, 'P1': P1}
        if A is not None:  model['A'] = A
    if not validate_model(model): raise TypeError('invalid combination of model matrices for model construction')
    model['dynamic']  = np.any([model[M]['dynamic'] for M in ('H','Z','T','R','Q','c','a1','P1')])
    model['n']  = min([model[M]['shape'][2] for M in ('H','Z','T','R','Q','c','a1','P1') if model[M]['dynamic']]) if model['dynamic'] else np.inf
    model['p']  = model['Z']['shape'][0]
    model['m']  = model['T']['shape'][0]
    model['r']  = model['R']['shape'][1]
    model['nparam'] = sum([model[M]['nparam'] for M in ('H','Z','T','R','Q','c','a1','P1') if not model[M]['constant']]) + (model['A']['nparam'] if 'A' in model else 0)
    model['mcom']   = [model['m']] # models built from model_from_mats are considered a single "component"
    return model

def model_cat(models):
    """Concatenate state space models
    models -- a list of state space models
    Each model is considered a separate set of components
    The final H matrix is taken from models[0]
    Concatenation of models with 'A' is not supported yet
    """
    N  = len(models)
    final_model = {}
    final_model['H'] = models[0]['H']
    for M in ('Z','T','R','Q','c','a1','P1'):
        final_model[M] = mat_cat(M,[models[i][M] for i in range(N)])
    final_model = model_from_mats(final_model)
    final_model['mcom'] = [m for i in range(N) for m in models[i]['mcom']]
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
    return M['mat'] if M['dynamic'] else [M['mat']]*n

def prepare_model(model,n):
    H   = prepare_mat(model['H'],n)
    Z   = prepare_mat(model['Z'],n)
    T   = prepare_mat(model['T'],n)
    R   = prepare_mat(model['R'],n)
    Q   = prepare_mat(model['Q'],n)
    c   = prepare_mat(model['c'],n)
    a1  = model['a1']['mat']
    P1  = model['P1']['mat']
    RQdyn       = model['R']['dynamic'] or model['Q']['dynamic']
    stationary  = not (model['H']['dynamic'] or model['Z']['dynamic'] or model['T']['dynamic'] or RQdyn) # c does not effect convergence of P
    return H, Z, T, R, Q, c, a1, P1, stationary, RQdyn

def set_param(model,x):
    # The model is modified inplace, but reference returned for convenience
    i  = 0
    if 'A' in model:
        target  = model['A']['target']
        nparam  = model['A']['nparam']
        m       = model['A']['func'](x[:nparam])
        for j in range(len(m)):
            model[target[j]]['mat']  = m[j]
        i       = nparam
    for M in ('H','Z','T','R','Q','c'):
        if not model[M]['constant'] and model[M]['func'] is not None:
            nparam  = model[M]['nparam']
            model[M]['mat'] = model[M]['func'](x[i:i+nparam])
            i  += nparam
    return model
