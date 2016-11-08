# -*- coding: utf-8 -*-

from ssm_alg_int import *
from scipy.linalg import block_diag
from scipy.optimize import minimize

def x_intv(n, intv_type, tau):
    # %X_INTV Create regression variables for intervention components.
    # %   x = X_INTV(n, type, tau)
    # %       n is the time series length.
    # %       type specifies intervention type: 'step', 'pulse', 'slope' or 'null'.
    # %       tau is the intervention onset time. (0-index)
    x   = zeros((1,n))
    if intv_type == 'step':
        x[:,tau:]  = 1
    elif intv_type == 'pulse':
        x[:,tau]   = 1
    elif intv_type == 'slope':
        x[:,tau:]  = range(1,n-tau+1)
    return x

def mat_const(mat):
    if type(mat) == list:
        mat  = [asmatrix(x) for x in mat]
    else:
        mat  = asmatrix(mat)
    return {
        'linear':   True,
        'dynamic':  type(mat) == list,
        'constant': True,
        'shape': mat[0].shape + (len(mat),) if type(mat) == list else mat.shape,
        'mat': mat}

def mat_var(p=1,cov=True):
    # Create a parametrized normal covariance matrix for use as state space matrix
    #   p is the number of variables.
    #   cov specifies complete covariance if true, or complete independence if false.

    #-- Construct function to generate model --#
    if p == 1:
        return {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': (1,1),
            'func': lambda x: asmatrix(exp(2*x[0])),
            'nparam': 1}
    elif not cov:
        return {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': (p,p),
            'func': lambda x: matrix(diag(exp(2*asarray(x)))),
            'nparam': p}
    else:
        mask  = nonzero(tril(ones((p,p),dtype=bool) & ~eye(p,dtype=bool)))
        def x_to_cov(x):
            # bound variables: p,mask
            x1  = asarray(x[:p])
            x2  = asarray(x[p:])
            Y   = exp(x1)[:,None].T
            Y   = Y.T * Y
            C   = zeros((p,p))
            C[mask] = Y[mask] * (x2/sqrt(1 + x2**2))
            C   = C + C.T + diag(diag(Y))
            return matrix(C)

        return {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': (p,p),
            'func': x_to_cov,
            'nparam': p*(p+1)/2}

def set_param(model,x):
    # The model is modified inplace, but reference returned for convenience
    i  = 0
    for M in ('H','Z','T','R','Q','c'):
        if not model[M]['constant']:
            nparam  = model[M]['nparam']
            model[M]['mat'] = model[M]['func'](x[i:i+nparam])
            i  += nparam
    return model

def estimate(y,model,x0,method='Nelder-Mead'):
    #-- Get information about data --#
    p,n     = y.shape
    mis     = asarray(isnan(y))
    anymis  = any(mis,0)
    allmis  = all(mis,0)
    nmis    = n - sum(allmis)
    w       = sum([model[M]['nparam'] for M in model.keys() if not model[M]['constant']])

    #-- Estimate model parameters --#
    nloglik = lambda x: kalman_int(4,n,y,mis,anymis,allmis,set_param(model,x))[0]
    res     = minimize(nloglik,x0,method='Nelder-Mead')
    logL    = -nmis * (p*log(2*pi) + res.fun) / 2
    AIC     = (-2*logL + 2*(w + sum(model['P1']['mat'] == inf)))/nmis
    BIC     = (-2*logL + log(nmis)*(w + sum(model['P1']['mat'] == inf)))/nmis

    return res.x,logL,AIC,BIC

def model_llm():
    # Local linear model
    return {
        'H':  mat_var(1),
        'Z':  mat_const(1),
        'T':  mat_const(1),
        'R':  mat_const(1),
        'Q':  mat_var(1),
        'c':  mat_const(0),
        'a1': mat_const(0),
        'P1': mat_const(inf)}

def model_lpt(d,stochastic=True):
    # Local polynomial trend model
    #   d is the order of the polynomial trend.
    return {
        'H':  mat_var(1),
        'Z':  mat_const(hstack([matrix(1),zeros((1,d))])),
        'T':  mat_const(triu(ones((d+1,d+1)))),
        'R':  mat_const(eye(d+1) if stochastic else vstack([zeros((d,1)),matrix(1)])),
        'Q':  mat_var(d,False),
        'c':  mat_const(zeros((d+1,1))),
        'a1': mat_const(zeros((d+1,1))),
        'P1': mat_const(diag([inf]*(d+1)))}

def model_seasonal(seasonal_type,s):
    # %SSM_SEASONAL Create SSMODEL object for seasonal components.
    # %   model = SSM_SEASONAL(seasonal_type, s)
    # %       seasonal_type can be 'dummy', 'dummy fixed', 'h&s', 'trig1', 'trig2' or 'trig
    # %           fixed'.
    # %       s is the seasonal period.
    H  = mat_var(1)
    if seasonal_type in ('dummy','dummy_fixed'):
        #-- The dummy seasonal component --#
        Z   = mat_const(hstack([matrix(1),zeros((1,s-2))]))
        T   = mat_const(vstack([-ones((1,s-1)),hstack([eye(s-2),zeros((s-2,1))])]))
        if seasonal_type == 'dummy':
            R  = mat_const(vstack([matrix(1),zeros((s-2,1))]))
            Q  = mat_var(1)
        else:
            R  = mat_const(zeros((s-1,0)))
            Q  = mat_const(zeros((0,0)))
        c   = mat_const(zeros((s-1,1)))
        a1  = mat_const(zeros((s-1,1)))
        P1  = mat_const(diag([inf]*(s-1)))
    elif seasonal_type == 'h&s':
        #-- The seasonal component suggested by Harrison and Stevens (1976) --#
        Z     = mat_const(hstack([matrix(1),zeros((1,s-1))]))
        T     = mat_const(vstack([hstack([zeros((s-1,1)),eye(s-1)]),hstack([matrix(1),zeros((1,s-1))])]))
        R     = mat_const(eye(s))
        W     = matrix(eye(s) - tile(1.0/s,(s,s)))
        Q     = {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': (s,s),
            'func': lambda x: exp(2*x[0])*W,
            'nparam': 1}
        c     = mat_const(zeros((s,1)))
        a1    = mat_const(zeros((s,1)))
        P1    = mat_const(diag([inf]*s))
    elif seasonal_type in ('trig1','trig2'):
        #-- Trigonometric seasonal component --#
        Z  = mat_const(hstack([tile([1,0],(1,floor((s-1)/2.))),ones((1,1 - s%2))]))
        T  = []
        if s%2 == 0:
            for i in range(1,s/2):
                Lambda  = 2*pi*i/s
                T.append([[cos(Lambda),sin(Lambda)],[-sin(Lambda),cos(Lambda)]])
            T  = mat_const(block_diag(*(T+[-1])))
        else: # s%2 == 1
            for i in range(1,(s+1)/2):
                Lambda  = 2*pi*i/s
                T.append([[cos(Lambda),sin(Lambda)],[-sin(Lambda),cos(Lambda)]])
            T  = mat_const(block_diag(*T))
        R  = mat_const(eye(s-1))
        if seasonal_type == 'trig1':
            #-- Trigonometric seasonal component with equal variance --#
            W  = matrix(eye(s-1))
            Q  = {
                'gaussian': True,
                'dynamic':  False,
                'constant': False,
                'shape': (s-1,s-1),
                'func': lambda x: exp(2*x[0])*W,
                'nparam': 1}
        else:
            Q  = mat_var(s-1,False)
        c  = mat_const(zeros((s-1,1)))
        a1    = mat_const(zeros((s-1,1)))
        P1    = mat_const(diag([inf]*(s-1)))

    return {'H': H, 'Z': Z, 'T': T, 'R': R, 'Q': Q, 'c': c, 'a1': a1, 'P1': P1}

def model_intv(n, intv_type, tau, dynamic=False):
    x  = x_intv(n, intv_type, tau)
    m  = x.shape[0]
    return {
        'H':  mat_var(1),
        'Z':  mat_const(dsplit(x[None,:,:],n)),
        'T':  mat_const(eye(m)),
        'R':  mat_const(eye(m)) if dynamic else mat_const(zeros((m,0))),
        'Q':  mat_var(1) if dynamic else mat_const(zeros((0,0))),
        'c':  mat_const(zeros((m,1))),
        'a1': mat_const(zeros((m,1))),
        'P1': mat_const(diag([inf]*m))}

def model_reg(x, dynamic=False):
    m,n  = asmatrix(x).shape
    return {
        'H':  mat_var(1),
        'Z':  mat_const(dsplit(x[None,:,:],n)),
        'T':  mat_const(eye(m)),
        'R':  mat_const(eye(m)) if dynamic else mat_const(zeros((m,0))),
        'Q':  mat_var(1) if dynamic else mat_const(zeros((0,0))),
        'c':  mat_const(zeros((m,1))),
        'a1': mat_const(zeros((m,1))),
        'P1': mat_const(diag([inf]*m))}

def model_stsm(lvl, seas, s, cycle=False, x=None):
    # %SSM_STSM Create SSMODEL object for structural time series models.
    # %   model = SSM_STSM(lvl, seas, s[, cycle, x, varname])
    # %       lvl is 'level' or 'trend'.
    # %       seas is the seasonal type (see ssm_seasonal).
    # %       s is the seasonal period.
    # %       Set cycle to true if there is a cycle component in the model.
    # %       x is explanatory variables.

    if lvl == 'level':
        model1 = model_llm()
    elif lvl == 'trend':
        model1 = model_lpt(1)
    else:
        model1 = {
            'H': mat_var(1),
            'Z': mat_const(zeros((1,0))),
            'T': mat_const(zeros((0,0))),
            'R': mat_const(zeros((0,0))),
            'Q': mat_const(zeros((0,0))),
            'c': mat_const(zeros((0,1))),
            'a1': mat_const(zeros((0,1))),
            'P1': mat_const(zeros((0,0)))}

    model2 = model_seasonal(seas,s)

    # if cycle, model = [model ssm_cycle]; end
    # if nargin >= 5, model = [model ssm_reg(varargin{:})]; end

    return {
        'H':  model1['H'],
        'Z':  mat_const(hstack([model1['Z']['mat'],model2['Z']['mat']])),
        'T':  mat_const(block_diag(model1['T']['mat'],model2['T']['mat'])),
        'R':  mat_const(block_diag(model1['R']['mat'],model2['R']['mat'])),
        'Q':  {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': asarray(model1['Q']['shape']) + asarray(model2['Q']['shape']),
            'func': lambda x: asmatrix(block_diag(model1['Q']['func'](x[:model1['Q']['nparam']]),model2['Q']['func'](x[model1['Q']['nparam']:]))),
            'nparam': model1['Q']['nparam'] + model2['Q']['nparam']},
        'c':  mat_const(vstack([model1['c']['mat'],model2['c']['mat']])),
        'a1': mat_const(vstack([model1['a1']['mat'],model2['a1']['mat']])),
        'P1': mat_const(block_diag(model1['P1']['mat'],model2['P1']['mat']))}

# Nested functions inside a function can be defined multiple times, but any outside variables "bound" into the function will take the last value at the outer function exit, making multiple function definitions equivalent ...
def model_cat(models):
    # Combine state space models
    N  = len(models)
    final_model = {}
    final_model['H'] = models[0]['H']
    for M in ('Z','T','R','Q','c','a1','P1'):
        final_model[M] = mat_cat(M,[models[i][M] for i in range(N)])
    return final_model

def func_stat_to_dyn(func,n):
    return lambda x: [func(x)]*n

def mat_cat(M,mats):
    # M is one of 'H','Z','T','R','Q','c','a1','P1'
    N        = len(mats)
    dynamic  = any([mats[i]['dynamic'] for i in range(N)])
    n        = max([mats[i]['shape'][2] if mats[i]['dynamic'] else 1 for i in range(N)])
    constant_l  = [mats[i]['constant'] for i in range(N)]
    constant    = all(constant_l)
    if not constant:
        nparam_l  = [mats[i]['nparam'] if not mats[i]['constant'] else 0 for i in range(N)]
        nparam    = sum(nparam_l)
        nparam_l  = cumsum([0] + nparam_l)
    if M == 'Z':
        mstack  = lambda x: asmatrix(hstack(x))
        shape   = mats[0]['shape'][0], sum([mats[i]['shape'][1] for i in range(N)])
    elif M in ('c','a1'):
        mstack  = lambda x: asmatrix(vstack(x))
        shape   = sum([mats[i]['shape'][0] for i in range(N)]), mats[0]['shape'][1]
    else:
        mstack  = lambda x: asmatrix(block_diag(*x))
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
        func_mask  = nonzero(~asarray(constant_l))[0]
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
