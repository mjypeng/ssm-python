# -*- coding: utf-8 -*-

from common import *

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

def model_llm():
    # Local linear model
    return ssmodel(H=mat_var(1), Z=mat_const(1), T=mat_const(1),
        R=mat_const(1), Q=mat_var(1), c=mat_const(0),
        a1=mat_const(0), P1=mat_const(np.inf))

def model_lpt(d,stochastic=True):
    # Local polynomial trend model
    #   d is the order of the polynomial trend.
    return ssmodel(H=mat_var(1), Z=mat_const([[1.0] + [0.0]*d]),
        T=mat_const(np.triu(np.ones((d+1,)*2))),
        R=mat_const(np.eye(d+1) if stochastic else [[0.0]*d + [1.0]]),
        Q=mat_var(d+1,False),
        c=mat_const(np.zeros((d+1,1))),
        a1=mat_const(np.zeros((d+1,1))),
        P1=mat_const(np.diag([np.inf]*(d+1))))

def model_seasonal(seasonal_type,s):
    """Create seasonal components.
    model = model_seasonal(seasonal_type, s)
    seasonal_type -- can be 'dummy', 'dummy fixed', 'h&s', 'trig1', 'trig2' or 'trig
       fixed'.
    s -- the seasonal period.
    """
    if seasonal_type in ('dummy','dummy_fixed'):
        #-- The dummy seasonal component --#
        m   = s-1
        Z   = mat_const([[1.0] + [0.0]*(s-2)])
        T   = mat_const(np.bmat([[-np.ones((1,s-1))],[np.eye(s-2),np.zeros((s-2,1))]]))
        if seasonal_type == 'dummy':
            R  = mat_const([[1.0]] + [[0.0]]*(s-2))
            Q  = mat_var(1)
        else:
            R  = mat_const(np.zeros((s-1,0)))
            Q  = mat_const(np.zeros((0,0)))
    elif seasonal_type == 'h&s':
        #-- The seasonal component suggested by Harrison and Stevens (1976) --#
        m     = s
        Z     = mat_const([1.0] + [0.0]*(s-1))
        T     = mat_const(np.bmat([[np.zeros((s-1,1)),np.eye(s-1)],[np.asmatrix(1),np.zeros((1,s-1))]]))
        R     = mat_const(np.eye(s))
        W     = np.matrix(np.eye(s) - np.tile(1.0/s,(s,s)))
        Q     = ssmat(transform=False, dynamic=False, constant=False,
                      shape=(s,s), func=lambda x: np.exp(2*x[0])*W,
                      nparam=1)
    elif seasonal_type in ('trig1','trig2','trig fixed'):
        #-- Trigonometric seasonal component --#
        m  = s-1
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
            Q  = ssmat(transform=False, dynamic=False, constant=False,
                       shape=(s-1,s-1), func=lambda x: np.exp(2*x[0])*W,
                       nparam=1)
        elif seasonal_type == 'trig2':
            R  = mat_const(np.eye(s-1))
            Q  = mat_var(s-1,False)
        else: # seasonal_type == 'trig fixed'
            R  = mat_const(np.zeros((s-1,0)))
            Q  = mat_const(np.zeros((0,0)))

    return ssmodel(H=mat_var(1), Z=Z, T=T, R=R, Q=Q,
                   c=mat_const(np.zeros((m,1))),
                   a1=mat_const(np.zeros((m,1))),
                   P1=mat_const(np.diag([np.inf]*m)))

def model_intv(n, intv_type, tau, dynamic=False):
    x  = x_intv(n, intv_type, tau)
    m  = x.shape[0]
    return ssmodel(H=mat_var(1),
        Z=mat_const(np.dsplit(x[None,:,:],n),dynamic=True),
        T=mat_const(np.eye(m)),
        R=mat_const(np.eye(m) if dynamic else np.zeros((m,0))),
        Q=mat_var(1) if dynamic else mat_const(np.zeros((0,0))),
        c=mat_const(np.zeros((m,1))),
        a1=mat_const(np.zeros((m,1))),
        P1=mat_const(np.diag([np.inf]*m)))

def model_reg(x, dynamic=False):
    m,n  = x.shape
    return ssmodel(H=mat_var(1),
        Z=mat_const(np.dsplit(x[None,:,:],n),dynamic=True),
        T=mat_const(np.eye(m)),
        R=mat_const(np.eye(m) if dynamic else np.zeros((m,0))),
        Q=mat_var(1) if dynamic else mat_const(np.zeros((0,0))),
        c=mat_const(np.zeros((m,1))),
        a1=mat_const(np.zeros((m,1))),
        P1=mat_const(np.diag([np.inf]*m)))

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
        model1 = ssmodel(H=mat_var(1),
            Z=mat_const(np.zeros((1,0))),
            T=mat_const(np.zeros((0,0))),
            R=mat_const(np.zeros((0,0))),
            Q=mat_const(np.zeros((0,0))),
            c=mat_const(np.zeros((0,1))),
            a1=mat_const(np.zeros((0,1))),
            P1=mat_const(np.zeros((0,0))))

    model2 = model_seasonal(seasonal_type,s)

    models = [model1,model2]

    # if cycle, model = [model ssm_cycle]; end
    if x is not None: models.append(model_reg(x))

    return model_cat(models)

    # return {
    #     'H=model1['H'],
    #     'Z=mat_const(np.hstack([model1['Z']['mat'],model2['Z']['mat']])),
    #     'T=mat_const(blkdiag(model1['T']['mat'],model2['T']['mat'])),
    #     'R=mat_const(blkdiag(model1['R']['mat'],model2['R']['mat'])),
    #     'Q={
    #         'gaussian': True,
    #         'dynamic':  False,
    #         'constant': False,
    #         'shape': np.asarray(model1['Q']['shape']) + np.asarray(model2['Q']['shape']),
    #         'func': lambda x: np.asmatrix(blkdiag(model1['Q']['func'](x[:model1['Q']['nparam']]),model2['Q']['func'](x[model1['Q']['nparam']:]))),
    #         'nparam': model1['Q']['nparam'] + model2['Q']['nparam']},
    #     'c=mat_const(np.vstack([model1['c']['mat'],model2['c']['mat']])),
    #     'a1': mat_const(np.vstack([model1['a1']['mat'],model2['a1']['mat']])),
    #     'P1': mat_const(blkdiag(model1['P1']['mat'],model2['P1']['mat']))}

def model_mvllm(p, cov=(True,True)):
    # %SSM_MVLLM Create SSMODEL object for multivariate local level model.
    # %   model = SSM_MVLLM(p[, cov])
    # %       p is the number of variables.
    # %       cov specifies complete covariance if true, or complete independence if
    # %           false, cov[0] for observation disturbance, cov[1] for state transition disturbance
    #   Each of the p variables evolve independently, as indicated by identity matrices for Z,T,R, so the only source of dependence is in the disturbances. Hence it does not make much sense to have cov all False, since it is equivalent to running p local linear models separately.
    return ssmodel(H=mat_var(p, cov[0]),
        Z=mat_const(np.eye(p)),
        T=mat_const(np.eye(p)),
        R=mat_const(np.eye(p)),
        Q=mat_var(p, cov[1]),
        c=mat_const(np.zeros((p,1))),
        a1=mat_const(np.zeros((p,1))),
        P1=mat_const(np.diag([np.inf]*p)))

def model_mvllt(p, cov=(True,True,True)):
    # %SSM_MVLLT Create SSMODEL object for multivariate local level trend model.
    # %   model = SSM_MVLLT(p[, cov])
    # %       p is the number of variables.
    # %       cov specifies complete covariance if true, or complete independence if
    # %           false, extended to a vector where needed.
    return ssmodel(H=mat_var(p, cov[0]),
        Z=mat_const(np.kron(np.eye(p), [1,0])),
        T=mat_const(np.kron(np.eye(p), [[1,1],[0,1]])),
        R=mat_const(np.eye(p*2)),
        Q=mat_interlvar(p, 2, cov[1:]),
        c=mat_const(np.zeros((p*2,1))),
        a1=mat_const(np.zeros((p*2,1))),
        P1=mat_const(np.diag([np.inf]*(p*2))))

def model_mvseasonal(p, cov, seasonal_type, s):
    """Create multivariate seasonal component.
    model_mvseasonal(p, cov, seasonal_type, s)
    p is the number of variables.
    cov specifies complete covariance if true, or complete independence if false
    """
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
        Q   = ssmat(transform=False, dynamic=False, constant=False,
                    shape=(s*p,)*2,
                    func=lambda x: np.kron(np.diag(np.exp(2*np.asarray(x))),W),
                    nparam=p)
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

    return ssmodel(H=mat_var(p,True), Z=Z, T=T, R=R, Q=Q,
        c=mat_const(np.zeros((m,1))),
        a1=mat_const(np.zeros((m,1))),
        P1=mat_const(np.diag([np.inf]*m)))

def model_mvreg(p, x, dep=None):
    # %MAT_MVREG Create base matrices for multivariate regression component.
    # %   [Z Zdmmask Zdvec T R] = MAT_MVREG(p, x, dep)
    # %       p is the number of observation variables.
    # %       dep is a p*size(x, 1) logical matrix which specifies the dependence of
    # %           each observation with each regression variable.
    m0,n  = x.shape
    if dep is not None:
        dep = np.asarray(dep)
        m   = np.sum(dep)
        X   = [x[None,dep[i,:],:] for i in range(p)]
        Z   = mat_const([blkdiag(*[X[i][:,:,t] for i in range(p)]) for t in range(n)],dynamic=True)
    else:
        m   = p*m0
        Z   = mat_const(np.dsplit(np.kron(np.eye(p)[:,:,None],x[None,:,:]),n),dynamic=True)
    return ssmodel(H=mat_var(p,True), Z=Z,
        T=mat_const(np.eye(m)),
        R=mat_const(np.zeros((m,0))),
        Q=mat_const(np.zeros((0,0))),
        c=mat_const(np.zeros((m,1))),
        a1=mat_const(np.zeros((m,1))),
        P1=mat_const(np.diag([np.inf]*m)))

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
        model1 = ssmodel(H=mat_var(p,cov[0]),
            Z=mat_const(np.zeros((p,0))),
            T=mat_const(np.zeros((0,0))),
            R=mat_const(np.zeros((0,0))),
            Q=mat_const(np.zeros((0,0))),
            c=mat_const(np.zeros((0,1))),
            a1=mat_const(np.zeros((0,1))),
            P1=mat_const(np.zeros((0,0)))); cov = cov[1:]

    model2  = model_mvseasonal(p, cov[0], seasonal_type, s)

    models  = [model1,model2]

    # if cycle, model = [model ssm_mvcycle(p, cov(k+1))]; end

    if x is not None: models.append(model_mvreg(p, x))

    return model_cat(models)

def model_arma(p, q, arma_mean=False):
    # %MAT_ARMA Create base matrices for ARMA(p, q) models.
    # %   [Z T R P1 Tmask Rmask P1mask] = MAT_ARMA(p, q[, mean])
    # %   [T R P1 Tmask Rmask P1mask] = MAT_ARMA(p, q[, mean])
    # %       Set mean to true to create a model with mean.
    r   = max(p, q+1)
    H   = mat_const(np.zeros((1,1)),trans=False)
    if arma_mean:
        m    = r + 1
        Z    = mat_const([1.0] + [0.0]*r)
        T0   = np.bmat([[np.zeros((r,1)),np.eye(r)],[np.mat([0.0]*r+[1.0])]])
        R0   = np.mat([[1.0]] + [[0.0]]*r)
    else:
        m    = r
        Z    = mat_const([1.0] + [0.0]*(r-1))
        T0   = np.bmat([[np.zeros((r-1,1)),np.eye(r-1)],[np.zeros((1,r))]])
        R0   = np.mat([[1.0]] + [[0.0]]*(r-1))

    def psi_to_arma(x):
        # Bound variables: p, q, T0, R0, arma_mean, r
        x1  = np.asarray(x[:p])
        x2  = np.asarray(x[p:p+q])
        x3  = x[-1]
        if p == 1:
            T0[0,0]  = x1/np.sqrt(1 + x1**2)
        elif p == 2:
            Y        = x1/np.sqrt(1 + x1**2)
            T0[0,0]  = (1 - Y[1]) * Y[0]
            T0[1,0]  = Y[1]
        elif p > 2:
            T0[:p,0] = x1[:,None]
        if q == 1:
            R0[1,0]  = x2/np.sqrt(1 + x2**2)
        elif q == 2:
            Y        = x2/np.sqrt(1 + x2**2)
            R0[1,0]  = (1 - Y[1]) * Y[0]
            R0[2,0]  = Y[1]
        elif q > 2:
            R0[1:q+1,0] = x2[:,None]
        zetavar  = np.exp(2*x3)
        Q        = np.mat(zetavar)
        if arma_mean:
            Tr  = T0[:-1,:-1]
            Rr  = R0[:-1,:]
            P1  = zetavar*np.linalg.solve(np.eye(r*r) - np.kron(Tr,Tr),(Rr*Rr.T).reshape(r**2,1,order='F')).reshape(r,r,order='F')
            P1  = np.asmatrix(blkdiag(P1,np.inf))
        else:
            P1  = zetavar*np.linalg.solve(np.eye(r*r) - np.kron(T0,T0),(R0*R0.T).reshape(r**2,1,order='F')).reshape(r,r,order='F')
        return T0, R0, Q, P1

    return ssmodel(A={'target':('T','R','Q','P1'),'func':psi_to_arma,'nparam':p+q+1},
        H=H, Z=Z,
        T=ssmat(transform=True, dynamic=False, constant=False,
                   shape=(m,m), func=None, nparam=0),
        R=ssmat(transform=True, dynamic=False, constant=False,
                   shape=(m,1), func=None, nparam=0),
        Q=ssmat(transform=False, dynamic=False, constant=False,
                   shape=(1,1), func=None, nparam=0),
        c=mat_const(np.zeros((m,1))),
        a1=mat_const(np.zeros((m,1))),
        P1=ssmat(transform=False, dynamic=False, constant=False,
                   shape=(m,m), func=None, nparam=0))
