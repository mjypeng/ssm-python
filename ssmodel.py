# -*- coding: utf-8 -*-

from ssalg import *
from scipy.linalg import block_diag as blkdiag
from scipy.optimize import minimize

def x_intv(n, intv_type, tau):
    # %X_INTV Create regression variables for intervention components.
    # %   x = X_INTV(n, type, tau)
    # %       n is the time series length.
    # %       type specifies intervention type: 'step', 'pulse', 'slope' or 'null'.
    # %       tau is the intervention onset time. (0-index)
    x   = np.zeros((1,n))
    if intv_type == 'step':
        x[:,tau:]  = 1
    elif intv_type == 'pulse':
        x[:,tau]   = 1
    elif intv_type == 'slope':
        x[:,tau:]  = range(1,n-tau+1)
    return x

def mat_const(mat,dynamic=False):
    if dynamic:
        mat  = [np.asmatrix(x) for x in mat]
    else:
        mat  = np.asmatrix(mat)
    return {
        'linear':   True,
        'dynamic':  type(mat) == list,
        'constant': True,
        'shape': mat[0].shape + (len(mat),) if type(mat) == list else mat.shape,
        'mat': mat}

def f_psi_to_cov(p):
    # Returns a function that generates a (full) covariance matrix from a standard parametrization vector psi
    #   The covariance matrix is (p, p)
    #   The expected parameter vector psi is (p*(p+1)/2,)
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
    #   Each of the p variables have a single variance duplicated d times
    #   cov = True indicates that there are covariances between the p variables, each of which are also duplicated d times
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
    # %MAT_INTERLVAR Create base matrices for q-interleaved variance noise.
    # %   [m mmask] = MAT_INTERLVAR(p, q, cov)
    # %       p is the number of variables.
    # %       q is the number of variances affecting each variable.
    # %       cov is a logical vector that specifies whether each q variances covary
    # %           across variables. shape = (q,)
    # %       The variances affecting any single given variable is always assumed to be
    # %           independent.

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
    mis,anymis,allmis = get_missing(y)
    nmis    = n - sum(allmis)
    w       = sum([model[M]['nparam'] for M in model.keys() if not model[M]['constant']])

    #-- Estimate model parameters --#
    nloglik = lambda x: kalman_int(4,n,y,mis,anymis,allmis,set_param(model,x))[0]
    res     = minimize(nloglik,x0,method='Nelder-Mead')
    logL    = -nmis * (p*np.log(2*np.pi) + res.fun) / 2
    AIC     = (-2*logL + 2*(w + sum(model['P1']['mat'] == np.inf)))/nmis
    BIC     = (-2*logL + np.log(nmis)*(w + sum(model['P1']['mat'] == np.inf)))/nmis

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
        'P1': mat_const(np.inf)}

def model_lpt(d,stochastic=True):
    # Local polynomial trend model
    #   d is the order of the polynomial trend.
    return {
        'H':  mat_var(1),
        'Z':  mat_const([[1.0] + [0.0]*d]),
        'T':  mat_const(np.triu(np.ones((d+1,)*2))),
        'R':  mat_const(np.eye(d+1) if stochastic else [[0.0]*d + [1.0]]),
        'Q':  mat_var(d,False),
        'c':  mat_const(np.zeros((d+1,1))),
        'a1': mat_const(np.zeros((d+1,1))),
        'P1': mat_const(np.diag([np.inf]*(d+1)))}

def model_seasonal(seasonal_type,s):
    # %SSM_SEASONAL Create SSMODEL object for seasonal components.
    # %   model = SSM_SEASONAL(seasonal_type, s)
    # %       seasonal_type can be 'dummy', 'dummy fixed', 'h&s', 'trig1', 'trig2' or 'trig
    # %           fixed'.
    # %       s is the seasonal period.
    H  = mat_var(1)
    if seasonal_type in ('dummy','dummy_fixed'):
        #-- The dummy seasonal component --#
        Z   = mat_const([[1.0] + [0.0]*(s-2)])
        T   = mat_const(np.bmat([[-np.ones((1,s-1))],[np.eye(s-2),np.zeros((s-2,1))]]))
        if seasonal_type == 'dummy':
            R  = mat_const([[1.0]] + [[0.0]]*(s-2))
            Q  = mat_var(1)
        else:
            R  = mat_const(np.zeros((s-1,0)))
            Q  = mat_const(np.zeros((0,0)))
        c   = mat_const(np.zeros((s-1,1)))
        a1  = mat_const(np.zeros((s-1,1)))
        P1  = mat_const(np.diag([np.inf]*(s-1)))
    elif seasonal_type == 'h&s':
        #-- The seasonal component suggested by Harrison and Stevens (1976) --#
        Z     = mat_const([1.0] + [0.0]*(s-1))
        T     = mat_const(np.bmat([[np.zeros((s-1,1)),np.eye(s-1)],[np.asmatrix(1),np.zeros((1,s-1))]]))
        R     = mat_const(np.eye(s))
        W     = np.matrix(np.eye(s) - np.tile(1.0/s,(s,s)))
        Q     = {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': (s,s),
            'func': lambda x: np.exp(2*x[0])*W,
            'nparam': 1}
        c     = mat_const(np.zeros((s,1)))
        a1    = mat_const(np.zeros((s,1)))
        P1    = mat_const(np.diag([np.inf]*s))
    elif seasonal_type in ('trig1','trig2','trig fixed'):
        #-- Trigonometric seasonal component --#
        Z  = mat_const(np.bmat([np.tile([1.0,0.0],(1,np.floor((s-1)/2.))),np.ones((1,1 - s%2))]))
        T  = []
        if s%2 == 0:
            for i in range(1,s/2):
                Lambda  = 2*np.pi*i/s
                T.append([[np.cos(Lambda),np.sin(Lambda)],[-np.sin(Lambda),np.cos(Lambda)]])
            T  = mat_const(blkdiag(*(T+[-1])))
        else: # s%2 == 1
            for i in range(1,(s+1)/2):
                Lambda  = 2*np.pi*i/s
                T.append([[np.cos(Lambda),np.sin(Lambda)],[-np.sin(Lambda),np.cos(Lambda)]])
            T  = mat_const(blkdiag(*T))
        if seasonal_type == 'trig1':
            #-- Trigonometric seasonal component with equal variance --#
            R  = mat_const(np.eye(s-1))
            W  = np.matrix(np.eye(s-1))
            Q  = {
                'gaussian': True,
                'dynamic':  False,
                'constant': False,
                'shape': (s-1,s-1),
                'func': lambda x: np.exp(2*x[0])*W,
                'nparam': 1}
        elif seasonal_type == 'trig2':
            R  = mat_const(np.eye(s-1))
            Q  = mat_var(s-1,False)
        else: # seasonal_type == 'trig fixed'
            R  = mat_const(np.zeros((s-1,0)))
            Q  = mat_const(np.zeros((0,0)))
        c   = mat_const(np.zeros((s-1,1)))
        a1  = mat_const(np.zeros((s-1,1)))
        P1  = mat_const(np.diag([np.inf]*(s-1)))

    return {'H': H, 'Z': Z, 'T': T, 'R': R, 'Q': Q, 'c': c, 'a1': a1, 'P1': P1}

def model_intv(n, intv_type, tau, dynamic=False):
    x  = x_intv(n, intv_type, tau)
    m  = x.shape[0]
    return {
        'H':  mat_var(1),
        'Z':  mat_const(np.dsplit(x[None,:,:],n),dynamic=True),
        'T':  mat_const(np.eye(m)),
        'R':  mat_const(np.eye(m) if dynamic else np.zeros((m,0))),
        'Q':  mat_var(1) if dynamic else mat_const(np.zeros((0,0))),
        'c':  mat_const(np.zeros((m,1))),
        'a1': mat_const(np.zeros((m,1))),
        'P1': mat_const(np.diag([np.inf]*m))}

def model_reg(x, dynamic=False):
    m,n  = x.shape
    return {
        'H':  mat_var(1),
        'Z':  mat_const(np.dsplit(x[None,:,:],n),dynamic=True),
        'T':  mat_const(np.eye(m)),
        'R':  mat_const(np.eye(m) if dynamic else np.zeros((m,0))),
        'Q':  mat_var(1) if dynamic else mat_const(np.zeros((0,0))),
        'c':  mat_const(np.zeros((m,1))),
        'a1': mat_const(np.zeros((m,1))),
        'P1': mat_const(np.diag([np.inf]*m))}

def model_stsm(lvl, seasonal_type, s, cycle=False, x=None):
    # %SSM_STSM Create SSMODEL object for structural time series models.
    # %   model = SSM_STSM(lvl, seasonal_type, s[, cycle, x, varname])
    # %       lvl is 'level' or 'trend'.
    # %       seasonal_type is the seasonal type (see ssm_seasonal).
    # %       s is the seasonal period.
    # %       Set cycle to true if there is a cycle component in the model.
    # %       x is explanatory variables.

    if lvl == 'level':
        model1 = model_llm()
    elif lvl == 'trend':
        model1 = model_lpt(1)
    else:
        model1 = {
            'H':  mat_var(1),
            'Z':  mat_const(np.zeros((1,0))),
            'T':  mat_const(np.zeros((0,0))),
            'R':  mat_const(np.zeros((0,0))),
            'Q':  mat_const(np.zeros((0,0))),
            'c':  mat_const(np.zeros((0,1))),
            'a1': mat_const(np.zeros((0,1))),
            'P1': mat_const(np.zeros((0,0)))}

    model2 = model_seasonal(seasonal_type,s)

    # if cycle, model = [model ssm_cycle]; end
    # if nargin >= 5, model = [model ssm_reg(varargin{:})]; end

    return {
        'H':  model1['H'],
        'Z':  mat_const(np.hstack([model1['Z']['mat'],model2['Z']['mat']])),
        'T':  mat_const(blkdiag(model1['T']['mat'],model2['T']['mat'])),
        'R':  mat_const(blkdiag(model1['R']['mat'],model2['R']['mat'])),
        'Q':  {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': np.asarray(model1['Q']['shape']) + np.asarray(model2['Q']['shape']),
            'func': lambda x: np.asmatrix(blkdiag(model1['Q']['func'](x[:model1['Q']['nparam']]),model2['Q']['func'](x[model1['Q']['nparam']:]))),
            'nparam': model1['Q']['nparam'] + model2['Q']['nparam']},
        'c':  mat_const(np.vstack([model1['c']['mat'],model2['c']['mat']])),
        'a1': mat_const(np.vstack([model1['a1']['mat'],model2['a1']['mat']])),
        'P1': mat_const(blkdiag(model1['P1']['mat'],model2['P1']['mat']))}

def model_cat(models):
    # Combine state space models
    N  = len(models)
    final_model = {}
    final_model['H'] = models[0]['H']
    for M in ('Z','T','R','Q','c','a1','P1'):
        final_model[M] = mat_cat(M,[models[i][M] for i in range(N)])
    return final_model

def func_stat_to_dyn(func,n):
    #-- helper function for mat_cat() --#
    # Nested functions inside a function can be defined multiple times, but any outside variables "bound" into the function will take the last value at the outer function exit, making multiple function definitions equivalent ...
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

def model_mvllm(p, cov=(True,True)):
    # %SSM_MVLLM Create SSMODEL object for multivariate local level model.
    # %   model = SSM_MVLLM(p[, cov])
    # %       p is the number of variables.
    # %       cov specifies complete covariance if true, or complete independence if
    # %           false, cov[0] for observation disturbance, cov[1] for state transition disturbance
    #   Each of the p variables evolve independently, as indicated by identity matrices for Z,T,R, so the only source of dependence is in the disturbances. Hence it does not make much sense to have cov all False, since it is equivalent to running p local linear models separately.
    return {
        'H':  mat_var(p, cov[0]),
        'Z':  mat_const(np.eye(p)),
        'T':  mat_const(np.eye(p)),
        'R':  mat_const(np.eye(p)),
        'Q':  mat_var(p, cov[1]),
        'c':  mat_const(np.zeros((p,1))),
        'a1': mat_const(np.zeros((p,1))),
        'P1': mat_const(np.diag([np.inf]*p))}

def model_mvllt(p, cov=(True,True,True)):
    # %SSM_MVLLT Create SSMODEL object for multivariate local level trend model.
    # %   model = SSM_MVLLT(p[, cov])
    # %       p is the number of variables.
    # %       cov specifies complete covariance if true, or complete independence if
    # %           false, extended to a vector where needed.
    return {
        'H':  mat_var(p, cov[0]),
        'Z':  mat_const(np.kron(np.eye(p), [1,0])),
        'T':  mat_const(np.kron(np.eye(p), [[1,1],[0,1]])),
        'R':  mat_const(np.eye(p*2)),
        'Q':  mat_interlvar(p, 2, cov[1:]),
        'c':  mat_const(np.zeros((p*2,1))),
        'a1': mat_const(np.zeros((p*2,1))),
        'P1': mat_const(np.diag([np.inf]*(p*2)))}

def model_mvseasonal(p, cov, seasonal_type, s):
    # %SSM_MVSEASONAL Create SSMODEL object for multivariate seasonal component.
    # %   model = SSM_MVSEASONAL(p, cov, seasonal_type, s)
    # %       p is the number of variables.
    # %       cov specifies complete covariance if true, or complete independence if
    # %           false
    if seasonal_type in ('dummy','dummy fixed'):
        m   = p*(s-1)
        Z   = mat_const(np.kron(np.eye(p),np.hstack([np.matrix(1),np.zeros((1,s-2))])))
        T   = mat_const(np.kron(np.eye(p),bmat([[-np.ones((1,s-1))],[np.eye(s-2),np.zeros((s-2,1))]])))
        if seasonal_type == 'dummy':
            R   = mat_const(np.kron(np.eye(p),np.vstack([np.matrix(1),np.zeros((s-2,1))])))
            Q   = mat_var(p, cov)
        else:
            R   = mat_const(np.zeros((p*(s-1),0)))
            Q   = mat_const(np.zeros((0,0)))
    elif seasonal_type == 'h&s':
        # Multivariate H&S seasonal is always assumed independent
        m   = p*s
        Z   = np.kron(np.eye(p),np.hstack([np.matrix(1),np.zeros((1,s-1))]))
        T   = np.kron(np.eye(p),np.bmat([[np.zeros((s-1,1)),np.eye(s-1)],[np.matrix(1),np.zeros((1,s-1))]]))
        R   = np.eye(p*s)
        W   = np.matrix(np.eye(s) - np.tile(1.0/s,(s,s)))
        Q   = {
            'gaussian': True,
            'dynamic':  False,
            'constant': False,
            'shape': (s*p,)*2,
            'func': lambda x: np.kron(np.diag(np.exp(2*np.asarray(x))),W),
            'nparam': p}
    elif seasonal_type in ('trig1','trig2','trig fixed'):
        m   = p*(s-1)
        Z   = mat_const(np.kron(np.eye(p),np.bmat([np.tile([1.0,0.0],(1,np.floor((s-1)/2.))),np.ones((1,1 - s%2))])))
        T  = []
        if s%2 == 0:
            for i in range(1,s/2):
                Lambda  = 2*np.pi*i/s
                T.append([[np.cos(Lambda),np.sin(Lambda)],[-np.sin(Lambda),np.cos(Lambda)]])
            T.append(-1)
        else: # s%2 == 1
            for i in range(1,(s+1)/2):
                Lambda  = 2*np.pi*i/s
                T.append([[np.cos(Lambda),np.sin(Lambda)],[-np.sin(Lambda),np.cos(Lambda)]])
        T  = mat_const(np.kron(np.eye(p),blkdiag(*T)))
        if seasonal_type == 'trig1':
            R  = mat_const(np.eye(p*(s-1)))
            Q  = mat_dupvar(p, s-1, cov)
        elif seasonal_type == 'trig2':
            R  = mat_const(np.eye(p*(s-1)))
            Q  = mat_interlvar(p, s-1, cov)
        else: # seasonal_type == 'trig fixed'
            R  = mat_const(np.zeros((p*(s-1),0)))
            Q  = mat_const(np.zeros((0,0)))

    return {
        'H':  mat_var(p,True),
        'Z':  Z,
        'T':  T,
        'R':  R,
        'Q':  Q,
        'c':  mat_const(np.zeros((m,1))),
        'a1': mat_const(np.zeros((m,1))),
        'P1': mat_const(np.diag([np.inf]*m))}

def model_mvreg(p, x, dep=None):
    # %MAT_MVREG Create base matrices for multivariate regression component.
    # %   [Z Zdmmask Zdvec T R] = MAT_MVREG(p, x, dep)
    # %       p is the number of observation variables.
    # %       dep is a p*size(x, 1) logical matrix which specifies the dependence of
    # %           each observation with each regression variable.
    m0,n  = x.shape
    if dep is not None:
        m   = np.sum(dep)
        X   = [x[None,dep[i,:],:] for i in range(p)]
        Z   = mat_const([blkdiag(*[X[i][:,:,t] for i in range(p)]) for t in range(n)],dynamic=True)
    else:
        m   = p*m0
        Z   = mat_const(np.dsplit(np.kron(np.eye(p)[:,:,None],x[None,:,:]),n),dynamic=True)
    return {
        'H':  mat_var(p,True),
        'Z':  Z,
        'T':  mat_const(np.eye(m)),
        'R':  mat_const(np.zeros((m,0))),
        'Q':  mat_const(np.zeros((0,0))),
        'c':  mat_const(np.zeros((m,1))),
        'a1': mat_const(np.zeros((m,1))),
        'P1': mat_const(np.diag([np.inf]*m))}

def model_mvstsm(p, cov, lvl, seasonal_type, s, cycle=False, x=None):
    # %SSM_MVSTSM Create SSMODEL object for multivariate structural time series models.
    # %   model = SSM_MVSTSM(p, cov, lvl, seasonal_type, s[, cycle, x, dep (?)])
    # %       p is the number of variables.
    # %       cov specifies complete covariance if true, or complete independence if
    # %           false, extended to a vector where needed.

    if lvl == 'level':
        model1 = model_mvllm(p, cov[:2]); cov = cov[2:]
    elif lvl == 'trend':
        model1 = model_mvllt(p, cov[:3]); cov = cov[3:]
    else:
        model1 = {
            'H': mat_var(p,cov[0]),
            'Z': mat_const(np.zeros((p,0))),
            'T': mat_const(np.zeros((0,0))),
            'R': mat_const(np.zeros((0,0))),
            'Q': mat_const(np.zeros((0,0))),
            'c': mat_const(np.zeros((0,1))),
            'a1': mat_const(np.zeros((0,1))),
            'P1': mat_const(np.zeros((0,0)))}; cov = cov[1:]

    model2  = model_mvseasonal(p, cov[0], seasonal_type, s)

    models  = [model1,model2]

    # if cycle, model = [model ssm_mvcycle(p, cov(k+1))]; end

    if x is not None: models.append(model_mvreg(p, x))

    return model_cat(models)
