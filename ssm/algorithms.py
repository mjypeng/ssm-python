# -*- coding: utf-8 -*-

from .common import *
from scipy.optimize import minimize, show_options

DEFAULT_TOL = 10**-7

def _sigma(Sigma,u):
    # Function for incorporating covariance into independent Gaussian samples
    dgSigma = np.diag(Sigma)
    if (Sigma==np.diag(dgSigma)).all(): x = np.diag(np.sqrt(dgSigma)) * u
    else: # Sigma is not diagonal
        lmbda,U = np.linalg.eig(Sigma)
        x = U * (np.diag(np.sqrt(lmbda))) * u
        # [U Lambda] = eig(full(Sigma));
        # x = U * (np.diag(np.sqrt(np.diag(Lambda))) * u);
    return x

def _sample(N,n,p,m,r,H,Z,T,R,Q,c,a1,P1,stationary,Hdyn,Zdyn,RQdyn,Qdyn,cdyn):
    # function [y alpha eps eta] = _sample(N, n, p, m, r, Znl, Tnl, Z, T, Hdyn, Zdyn, Tdyn, Rdyn, Qdyn, cdyn, Hmat, Rmat, Qmat, cmat, a1, P1)
    # Sample unconditionally from state space model and parameters (i.e. unconditional on any observed data)
    # y is (p, N, n)
    # alpha is (m, N, n)
    # eps is (p, N, n)
    # eta is (r, N, n)
    #   All state space matrices would not be modified by _sample

    # %% Determine nonlinear functions %%
    # if Znl, Zdyn = false; else Zmat = getmat(Z)
    # if Tnl, Tdyn = false; else Tmat = getmat(T)
    c    = [np.tile(c[t],(1,N)) for t in range(n)] if cdyn else [np.tile(c[0],(1,N))]*n

    #-- Draw from Gaussian distribution --#
    alpha1  = np.random.normal(size=(m,N))
    eps     = np.random.normal(size=(p,N,n))
    eta     = np.random.normal(size=(r,N,n))

    #-- Initialization for sampling --#
    if Hdyn: eps[:,:,0] = _sigma(H[0],eps[:,:,0])
    else: eps = _sigma(H[0],eps.reshape((p,N*n))).reshape((p,N,n))
    if Qdyn: eta[:,:,0] = _sigma(Q[0],eta[:,:,0])
    else: eta = _sigma(Q[0],eta.reshape((r,N*n))).reshape((r,N,n))
    y       = np.zeros((p,N,n))
    alpha   = np.zeros((m,N,n))

    #-- Generate unconditional samples from the model (and given parameters) --#
    P1  = P1.copy()
    P1[P1 == np.inf] = 0
    alpha[:,:,0] = np.tile(a1,(1,N)) + _sigma(P1,alpha1)
    # if Znl, y[:,:,0] = getfunc(Z, alpha[:,:,0], 1);
    # else
    if Zdyn: y[:,:,0] = Z[0] * alpha[:,:,0]
    for t in range(1,n):
        if Hdyn: eps[:,:,t] = _sigma(H[t],eps[:,:,t])
        if Qdyn: eta[:,:,t] = _sigma(Q[t],eta[:,:,t])
        # if Tnl, alpha[:,:,t] = c + getfunc(T, alpha[:,:,t-1], t-1) + R * eta[:,:,t-1];
        # else 
        alpha[:,:,t] = c[t] + T[t] * alpha[:,:,t-1] + R[t] * eta[:,:,t-1]
        # if Znl, y[:,:,t] = getfunc(Z, alpha[:,:,t], t)
        if Zdyn: y[:,:,t] = Z[t] * alpha[:,:,t]
    # if ~Znl && 
    if not Zdyn:
        y = np.asarray(Z[0] * alpha.reshape((m,N*n))).reshape((p,N,n)) + eps # need to cast back to ndarray to preserve 3D (avoid matrix auto squeeze back to 2D)
    else:
        y = y + eps

    return y,alpha,eps,eta

def _kalman(mode, n,y,mis,anymis,allmis, H,Z,T,R,Q,c,a1,P1,stationary,RQdyn, tol,log_diag=False):
    # mode:
    #   0 - all output.
    #   1 - Kalman filter.
    #   2 - state smoother.
    #   3 - disturbance smoother.
    #   4 - loglikelihood.
    #   5 - fast smoother.
    #   6 - fast state smoother.
    #   7 - fast disturbance smoother.
    #   8 - loglikelihood gradient.
    # (Only a and v depends on the data y, the values of P, P_inf, d, ... etc. is
    # fixed for given model matrices (parameters).)
    # log_diag: whether to log kalman iteration diagnostics
    #   All data and state space matrices would not be modified by _kalman

    #-- Set output for operating mode --#
    Output_a, Output_P, Output_v, Output_invF, Output_K, Output_L, Output_Pinf, Output_F2, Output_L1, Output_logL_, Output_var_, Output_RQ, Output_QRt, Output_RQRt  = (False,)*14
    if mode in (0,'all'): # all output
        mode = 0
        Output_a, Output_P, Output_v, Output_invF, Output_K, Output_L, Output_Pinf, Output_F2, Output_L1, Output_logL_, Output_var_, Output_RQ, Output_QRt, Output_RQRt  = (True,)*14
    elif mode in (1,'kalman'): # Kalman filter
        mode = 1
        Output_a, Output_P, Output_v, Output_invF  = (True,)*4
    elif mode in (2,'statesmo'): # state smoother
        mode = 2
        Output_a, Output_P, Output_v, Output_invF, Output_L, Output_Pinf, Output_F2, Output_L1    = (True,)*8
    elif mode in (3,'disturbsmo'): # disturbance smoother
        mode = 3
        Output_v, Output_invF, Output_K, Output_L, Output_RQ, Output_QRt = (True,)*6
    elif mode in (4,'loglikelihood'): # loglikelihood
        mode = 4
        Output_logL_, Output_var_  = (True,)*2
    elif mode in (5,'fastsmo'): # fast smoother
        mode = 5
        Output_v, Output_invF, Output_K, Output_L, Output_L1, Output_QRt = (True,)*6
    elif mode in (6,'faststatesmo'): # fast state smoother
        mode = 6
        Output_v, Output_invF, Output_L, Output_L1, Output_RQRt  = (True,)*5
    elif mode in (7,'fastdisturbsmo'): # fast disturbance smoother
        mode = 7
        Output_v, Output_invF, Output_K, Output_L, Output_QRt   = (True,)*5
    elif mode in (8,'loglikgrad'): # loglikelihood gradient
        mode = 8
        Output_v, Output_invF, Output_K, Output_L, Output_logL_, Output_var_ = (True,)*6
    if log_diag: iter_log = ['']*n

    #-- Initialization --#
    a       = a1.copy()
    P       = P1.copy()
    D       = (P == np.inf)
    init    = D.any() # use exact diffuse initialization if init = true.
    if init:
        d     = n # 0-index: t is in [0,n-1]
        P[D]  = 0
        P_inf = D.astype(float)
    else:
        d     = -1
    converged   = False

    #-- Preallocate Output Results --#
    if Output_a:
        Result_a           = np.zeros((a.shape[0],n+1))
        Result_a[:,[0]]    = a
    if Output_P:
        Result_P           = np.zeros(P.shape + (n+1,))
        Result_P[:,:,0]    = P
    if Output_v:     Result_v     = [None]*n
    if Output_invF:  Result_invF  = [None]*n
    if Output_K:     Result_K     = [None]*n
    if Output_L:     Result_L     = [None]*n
    if Output_Pinf:
        Result_Pinf = [np.asarray(P_inf)] if init else [np.zeros_like(P)] # Length will be d
    if Output_F2:    Result_F2    = [] # length d
    if Output_L1:    Result_L1    = [] # length d
    if Output_logL_: Result_logL_ = 0
    if Output_var_:  Result_var_  = 0
    if Output_RQ:
        Result_RQ   = [None]*n if RQdyn else [R[0]*Q[0]]*n
    if Output_QRt:
        Result_QRt  = [None]*n if RQdyn else [Q[0]*R[0].T]*n
    if Output_RQRt:
        Result_RQRt = [None]*n if RQdyn else [RQRt[0]]*n

    Fns   = np.zeros(n,dtype=bool) # Is F_inf nonsingular for each iteration
    RQRt  = [R[t]*(Q[t]*R[t].T) for t in range(n)] if RQdyn else [R[0]*(Q[0]*R[0].T)]*n

    #-- Kalman filter loop --#
    for t in range(n):
        if converged: converged = not anymis[t] # Any missing observation throws off the steady state
        if converged:
            #-- Kalman filter after steady state reached (i.e. P is constant) --#
            if log_diag: iter_log[t] = 'converged'
            v   = y[:,t] - Z[t] * a
            a   = c[t] + T[t] * a + K * v
        elif allmis[t]:
            #-- Kalman filter when all observations are missing --#
            if log_diag: iter_log[t] = 'allmis'
            v     = np.zeros_like(v)
            F     = Z[t] * (P * Z[t].T) + H[t]
            invF  = F.I #### TODO: Ignore diffuse initialization for now
            K     = np.zeros_like(K)
            L     = T[t].copy()
            a     = c[t] + T[t] * a
            P     = T[t] * P * T[t].T + RQRt[t]
            if init:
                if log_diag: iter_log[t] += ',init'
                P_inf = T[t] * P_inf * T[t].T
        else:
            if anymis[t]:
                # "Disable" parts of state space matrices that corresponds to missing elements in the observation vector
                if log_diag: iter_log[t] = 'anymis,'
                Zt    = Z[t]
                Z[t]  = Z[t][~mis[:,t],:]
                Ht    = H[t]
                H[t]  = H[t][np.ix_(~mis[:,t],~mis[:,t])]
            if init:
                #-- Exact diffuse initial Kalman filter --#
                if log_diag: iter_log[t] += 'init'
                M       = P * Z[t].T
                M_inf   = P_inf * Z[t].T
                A_inf   = T[t] * P_inf
                if (abs(M_inf) < tol).all(): # F_inf is zero
                    if log_diag: iter_log[t] += ',F_inf=0'
                    F       = Z[t] * M + H[t]
                    invF    = F.I # The real invF
                    F2      = np.zeros_like(F2)
                    K       = T[t] * M * invF
                    K1      = np.zeros_like(K1)
                    L       = T[t] - K * Z[t]
                    L1      = np.zeros_like(L1)
                    P       = T[t] * P * L.T + RQRt[t]
                    P_inf   = A_inf * T[t].T
                else: # F_inf is assumed to be nonsingular
                    if log_diag: iter_log[t] += ',F_inf!=0'
                    Fns[t]  = True
                    invF    = (Z[t] * M_inf).I # This is actually invF1
                    F       = Z[t] * M + H[t]
                    F2      = -invF * F * invF
                    K       = T[t] * M_inf * invF
                    K1      = T[t] * (M * invF + M_inf * F2)
                    L       = T[t] - K * Z[t]
                    L1      = -K1 * Z[t]
                    P       = T[t] * P * L.T + A_inf * L1.T + RQRt[t]
                    P_inf   = A_inf * L.T
                if (abs(P_inf) < tol).all():
                    d    = t
                    init = False
            else:
                #-- Normal Kalman filter --#
                if log_diag: iter_log[t] += 'normal'
                M       = P * Z[t].T
                F       = Z[t] * M + H[t]
                invF    = F.I
                K       = T[t] * M * invF
                L       = T[t] - K * Z[t]
                prevP   = P.copy()
                P       = T[t] * P * L.T + RQRt[t]
                if stationary and (abs(P-prevP) < tol).all(): converged = True

            #-- Kalman data filter --#
            v  = y[~mis[:,t],t] - Z[t] * a # The convoluted indexing for y is required to ensure that a column vector is returned (instead of a row vector, which would mess up the shape of a)
            a  = c[t] + T[t] * a + K * v

            if anymis[t]:
                # "Restore" parts of state space matrices that corresponds to missing elements in the observation vector
                Z[t]  = Zt
                H[t]  = Ht

        #-- Store results for this iteration (time point t) --#
        if Output_a:    Result_a[:,[t+1]]  = a
        if Output_P:    Result_P[:,:,t+1]  = P
        if Output_v:    Result_v[t]        = v
        if Output_invF: Result_invF[t]     = invF
        if Output_K:    Result_K[t]        = K
        if Output_L:    Result_L[t]        = L
        if t <= d:
            if Output_Pinf: Result_Pinf.append(P_inf)
            if Output_F2:   Result_F2.append(F2)
            if Output_L1:   Result_L1.append(L1)
        if not allmis[t]:
            if Output_logL_:
                detinvF     = np.linalg.det(invF)
                if detinvF > 0:   Result_logL_ = Result_logL_ - np.log(detinvF)
                elif detinvF < 0: Result_logL_ = np.nan
            if Output_var_:
                if t > d or not Fns[t]: Result_var_ = Result_var_ + v.T*invF*v
        if RQdyn:
            if Output_RQ:   Result_RQ[t]   = R[t]*Q[t]
            if Output_QRt:  Result_QRt[t]  = Q[t]*R[t].T
            if Output_RQRt: Result_RQRt[t] = RQRt[t]

    #-- Output Results --#
    if mode == 0: # all output
        output = Result_a,Result_P,d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_Pinf,Result_F2,Result_L1,Result_logL_+Result_var_,Result_var_,Result_RQ,Result_QRt,Result_RQRt
    elif mode == 1: # Kalman filter
        output = Result_a,Result_P,d,Result_v,Result_invF
    elif mode == 2: # state smoother
        output = Result_a,Result_P,d,Fns,Result_v,Result_invF,Result_L,Result_Pinf,Result_F2,Result_L1
    elif mode == 3: # disturbance smoother
        output = d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_RQ,Result_QRt
    elif mode == 4: # loglikelihood
        output = Result_logL_+np.asmatrix(Result_var_)[0,0],np.asmatrix(Result_var_)[0,0]
    elif mode == 5: # fast smoother
        output = d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_L1,Result_QRt
    elif mode == 6: # fast state smoother
        output = d,Fns,Result_v,Result_invF,Result_L,Result_L1,Result_RQRt
    elif mode == 7: # fast disturbance smoother
        output = d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_QRt
    elif mode == 8: # loglikelihood gradient
        output = d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_logL_+Result_var_,Result_var_
    return output + (iter_log,) if log_diag else output

def _fastsmo(n,y,mis,anymis,allmis,m,H,Z,T,R,Q,c,a1,P1,stationary,RQdyn,tol):
    #   No state space matrices would be modified by _fastsmo

    #-- Kalman filter --#
    d,Fns,v,invF,K,L,L1,QRt = _kalman(5,n,y,mis,anymis,allmis,H,Z,T,R,Q,c,a1,P1,stationary,RQdyn,tol)

    #-- Initialization --#
    P1      = P1.copy()
    D       = (P1 == np.inf)
    P1[D]   = 0
    P1_inf  = D.astype(float)

    #-- Disturbance smoothing backwards recursion --#
    r   = np.zeros((m,1))
    r1  = np.zeros((m,1))
    epshat  = np.zeros((y.shape[0],n))
    etahat  = np.zeros((QRt[0].shape[0],n))
    for t in range(n-1,-1,-1):
        etahat[:,t] = QRt[t] * r
        if allmis[t]:
            #-- Disturbance smoothing when all observations are missing --#
            epshat[:,t] = 0
            r  = T[t].T * r
            if t <= d: r1 = T[t].T * r1
        else:
            if anymis[t]:
                # "Disable" parts of state space matrices that corresponds to missing elements in the observation vector
                Zt    = Z[t]
                Z[t]  = Z[t][~mis[:,t],:]
                Ht    = H[t]
                H[t]  = H[t][np.ix_(~mis[:,t],~mis[:,t])]
            if t > d or not Fns[t]:
                #-- Normal disturbance smoothing or when F_inf is zero --#
                epshat[~mis[:,t],t] = H[t] * (invF[t] * v[t] - K[t].T * r)
                r = Z[t].T * invF[t] * v[t] + L[t].T * r
                if t <= d: r1 = T[t].T * r1
            else:
                #-- Exact initial disturbance smoothing when F_inf is nonsingular --#
                epshat[~mis[:,t],t] = -H[t] * K[t].T * r
                r1  = Z[t].T * invF[t] * v[t] + L[t].T * r1 + L1[t].T * r
                r   = L[t].T * r
            if anymis[t]:
                # "Restore" parts of state space matrices that corresponds to missing elements in the observation vector
                Z[t]  = Zt
                H[t]  = Ht

    #-- Fast state smoothing --#
    alphahat       = np.zeros((m,n))
    alphahat[:,0]  = a1 + P1 * r + P1_inf * r1
    for t in range(0,n-1):
        alphahat[:,t+1] = c[t] + T[t] * alphahat[:,t] + R[t] * etahat[:,t]

    return alphahat,epshat,etahat

def _batchkalman(mode,n,N,y,H,Z,T,R,Q,c,a1,P1,stationary,RQdyn,tol):
    # mode:
    #   0 - all output.
    #   1 - Kalman filter.
    #   2 - fast smoother.
    #   3 - fast state smoother.
    #   4 - fast disturbance smoother.
    #   y is (p, N, n) and has no missing data.
    # (Only a and v depends on the data y, the values of P, P_inf, d, ... etc. is
    # fixed for given model matrices (parameters).)
    #   No data or state space matrices would be modified by this routine
    #   c is (m, N, n), replicated N times

    Output_a, Output_P, Output_v, Output_invF, Output_K, Output_L, Output_Pinf, Output_F2, Output_L1, Output_RQ, Output_QRt, Output_RQRt  = (False,)*12
    if mode in (0,'all'): # all output
        mode = 0
        Output_a, Output_P, Output_v, Output_invF, Output_K, Output_L, Output_Pinf, Output_F2, Output_L1, Output_RQ, Output_QRt, Output_RQRt  = (True,)*12
    elif mode in (1,'kalman'): # Kalman filter
        mode = 1
        Output_a, Output_P, Output_v, Output_invF  = (True,)*4
    elif mode in (2,'fastsmo'): # fast smoother
        mode = 2
        Output_v, Output_invF, Output_K, Output_L, Output_L1, Output_QRt  = (True,)*6
    elif mode in (3,'faststatesmo'): # fast state smoother
        mode = 3
        Output_v, Output_invF, Output_L, Output_L1, Output_RQRt  = (True,)*5
    elif mode in (4,'fastdisturbsmo'): # fast disturbance smoother
        mode = 4
        Output_v, Output_invF, Output_K, Output_L, Output_QRt   = (True,)*5

    #-- Prepare state space matrices --#
    a     = np.tile(a1.copy(),(1,N))
    P     = P1.copy()
    RQRt  = [R[t]*(Q[t]*R[t].T) for t in range(n)] if RQdyn else [R[0]*(Q[0]*R[0].T)]*n

    #-- Initialization --#
    D       = (P == np.inf)
    init    = D.any() # use exact initialization if init = true.
    if init:
        d     = n
        P[D]  = 0
        P_inf = D.astype(float)
    else:
        d     = -1
    converged   = False

    #-- Preallocate Output Results --#
    if Output_a:
        Result_a           = np.zeros((a.shape[0],N,n+1))
        Result_a[:,:,0]    = a
    if Output_P:
        Result_P           = np.zeros(P.shape + (n+1,))
        Result_P[:,:,0]    = P
    if Output_v:     Result_v     = [None] * n
    if Output_invF:  Result_invF  = [None] * n
    if Output_K:     Result_K     = [None] * n
    if Output_L:     Result_L     = [None] * n
    if Output_Pinf:
        Result_Pinf = [np.asarray(P_inf)] if init else [np.zeros_like(P)] # List length will be d
    if Output_F2:    Result_F2    = [] # length d
    if Output_L1:    Result_L1    = [] # length d
    if Output_RQ:
        if RQdyn: Result_RQ   = [None] * n
        else:     Result_RQ   = [R[0] * Q[0]] * n
    if Output_QRt:
        if RQdyn: Result_QRt  = [None] * n
        else:     Result_QRt  = [Q[0] * R[0].T] * n
    if Output_RQRt:
        if RQdyn: Result_RQRt = [None] * n
        else:     Result_RQRt = [RQRt] * n

    Fns = np.zeros(n,dtype=bool) # Is F_inf nonsingular for each iteration

    #-- Batch Kalman filter loop --#
    for t in range(n):
        if not converged:
            if init:
                #-- Exact initial Kalman filter --#
                M       = P * Z[t].T
                M_inf   = P_inf * Z[t].T
                A_inf   = T[t] * P_inf
                if (abs(M_inf) < tol).all(): # F_inf is zero
                    Fns[t]  = False
                    F       = Z[t] * M + H[t]
                    F2      = np.zeros_like(F2)
                    invF    = F.I
                    K       = T[t] * M * invF
                    K1      = np.zeros_like(K1)
                    L       = T[t] - K * Z[t]
                    L1      = np.zeros_like(L1)
                    P       = T[t] * P * L.T + RQRt[t]
                    P_inf   = A_inf * T[t].T
                else: # F_inf is assumed to be nonsingular
                    invF    = (Z[t] * M_inf).I
                    F       = Z[t] * M + H[t]
                    F2      = -invF * F * invF
                    K       = T[t] * M_inf * invF
                    K1      = T[t] * (M * invF + M_inf * F2)
                    L       = T[t] - K * Z[t]
                    L1      = -K1 * Z[t]
                    P       = T[t] * P * L.T + A_inf * L1.T + RQRt[t]
                    P_inf   = A_inf * L.T
                if (abs(P_inf) < tol).all():
                    d    = t
                    init = False
            else:
                #-- Normal Kalman filter --#
                M       = P * Z[t].T
                F       = Z[t] * M + H[t]
                invF    = F.I
                K       = T[t] * M * invF
                L       = T[t] - K * Z[t]
                prevP   = P.copy()
                P       = T[t] * P * L.T + RQRt[t]
                if stationary and (abs(P-prevP) < tol).all(): converged = True

        #-- Kalman data filter --#
        v   = y[:,:,t] - Z[t] * a
        a   = c[t] + T[t] * a + K * v

        #-- Store results for this time point --#
        if Output_a:    Result_a[:,:,t+1] = a
        if Output_P:    Result_P[:,:,t+1] = P
        if Output_v:    Result_v[t]    = v
        if Output_invF: Result_invF[t] = invF
        if Output_K:    Result_K[t]    = K
        if Output_L:    Result_L[t]    = L
        if t <= d:
            if Output_Pinf: Result_Pinf[:,:,t+1] = P_inf
            if Fns[t]:
                if Output_F2: Result_F2[t] = F2
                if Output_L1: Result_L1[t] = L1
        if RQdyn:
            if Output_RQ:   Result_RQ[t]   = R[t] * Q[t]
            if Output_QRt:  Result_QRt[t]  = Q[t] * R[t].T
            if Output_RQRt: Result_RQRt[t] = RQRt[t]

    #-- Output Results --#
    if mode == 0: # all output
        return Result_a,Result_P,d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_Pinf,Result_F2,Result_L1,Result_RQ,Result_QRt,Result_RQRt
    elif mode == 1: # Kalman filter
        return Result_a,Result_P,d,Result_v,Result_invF
    elif mode == 2: # fast smoother
        return d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_L1,Result_QRt
    elif mode == 3: # fast state smoother
        return d,Fns,Result_v,Result_invF,Result_L,Result_L1,Result_RQRt
    elif mode == 4: # fast disturbance smoother
        return d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_QRt

def _batchsmo(mode,n,N,y,m,H,Z,T,R,Q,c,a1,P1,stationary,RQdyn,cdyn,tol):
    # y is (p, N, n)
    # alphahat is (m, N, n)
    # epshat is (p, N, n)
    # etahat is (r, N, n)
    # mode 0: return alphahat,epshat,etahat,Result_r
    # mode 1: return alphahat,epshat,etahat
    #   This routine does not modify any data or state space matrices

    #-- Prepare state space matrices --#
    c   = [np.tile(c[t],(1,N)) for t in range(n)] if cdyn else [np.tile(c[0],(1,N))]*n

    #-- Kalman filter --#
    d,Fns,v,invF,K,L,L1,QRt = _batchkalman(2,n,N,y,H,Z,T,R,Q,c,a1,P1,stationary,RQdyn,tol=tol)

    #-- Initialization --#
    a1      = np.tile(a1.copy(),(1,N))
    P1      = P1.copy()
    D       = (P1 == np.inf)
    P1[D]   = 0
    P1_inf  = D.astype(float)

    #-- Disturbance smoothing backwards recursion --#
    if mode == 0: Result_r = [None]*n
    r   = np.zeros((m,N))
    r1  = np.zeros((m,N))
    epshat  = np.zeros((y.shape[0],N,n))
    etahat  = np.zeros((QRt[0].shape[0],N,n))
    for t in range(n-1,-1,-1):
        if mode == 0: Result_r[t] = r
        etahat[:,:,t] = QRt[t] * r
        if t > d or not Fns[t]:
            #-- Normal disturbance smoothing or when F_inf is zero --#
            epshat[:,:,t] = H[t] * (invF[t] * v[t] - K[t].T * r)
            r = Z[t].T * invF[t] * v[t] + L[t].T * r
            if t <= d:
                r1 = T[t].T * r1
        else:
            #-- Exact initial disturbance smoothing when F_inf is nonsingular --#
            epshat[:,:,t] = -H[t] * K[t].T * r
            r1  = Z[t].T * invF[t] * v[t] + L[t].T * r1 + L1[t].T * r
            r   = L[t].T * r

    #-- Fast state smoothing --#
    alphahat           = np.zeros((m,N,n))
    alphahat[:,:,0]    = a1 + P1 * r + P1_inf * r1
    for t in range(0,n-1):
        alphahat[:,:,t+1] = c[t] + T[t] * alphahat[:,:,t] + R[t] * etahat[:,:,t]

    if mode == 0:
        return alphahat,epshat,etahat,Result_r
    elif mode == 1:
        return alphahat,epshat,etahat

def kalman(y,model,tol=DEFAULT_TOL,log_diag=False):
    #-- Prepare state space matrices and data --#
    n, p, y, mis, anymis, allmis  = prepare_data(y)
    H, Z, T, R, Q, c, a1, P1, stationary, RQdyn  = prepare_model(model,n)

    a, P, d, v, invF = _kalman(1,n,y,mis,anymis,allmis,H,Z,T,R,Q,c,a1,P1,stationary,RQdyn,tol=tol,log_diag=log_diag)

    # Build ndarrays from list output
    v   = np.array(np.concatenate(v,1))
    F   = np.concatenate([np.array(x[:,:,None]) for x in invF],2)**-1

    # Suppress output during diffuse initialization
    a[:,:d+1]     = np.nan
    P[:,:,:d+1]   = np.nan
    v[:,:d+1]     = np.nan
    F[:,:,:d+1]   = np.nan

    return a, P, v, F

def loglik(y,model,tol=DEFAULT_TOL):
    #-- Prepare state space matrices and data --#
    n, p, y, mis, anymis, allmis  = prepare_data(y)
    nmis    = n - np.sum(allmis)
    H, Z, T, R, Q, c, a1, P1, stationary, RQdyn  = prepare_model(model,n)

    #-- Calculate loglikelihood --#
    _logL,_fvar = _kalman(4,n,y,mis,anymis,allmis,H,Z,T,R,Q,c,a1,P1,stationary,RQdyn,tol=tol,log_diag=False)
    logL        = -(nmis*p*np.log(2*np.pi) + _logL) / 2
    fvar        = _fvar / (n*p - np.sum(P1 == np.inf))

    return logL, fvar

def estimate(y,model,x0,method=None,options=None,tol=DEFAULT_TOL):
    """
    model is modified inplace but reference returned for convenience
    """
    #-- Prepare state space matrices and data --#
    n, p, y, mis, anymis, allmis  = prepare_data(y)
    nmis    = n - sum(allmis)
    w       = model.nparam

    #-- Estimate model parameters --#
    _nloglik = lambda x: _kalman(4,n,y,mis,anymis,allmis,*prepare_model(set_param(model,x),n),tol=tol,log_diag=False)[0]

    result  = minimize(_nloglik,x0,method=method,options=options)

    model   = set_param(model,result.x)

    result.logL  = -(nmis*p*np.log(2*np.pi) + result.fun) / 2
    result.AIC   = (-2*result.logL + 2*(w + np.sum(model.P1.mat == np.inf)))/nmis
    result.BIC   = (-2*result.logL + np.log(nmis)*(w + np.sum(model.P1.mat == np.inf)))/nmis
    if not result.success: sys.stderr.write('ssm.estimate:warning:minimizer failed to exit successfully\n')

    return model,result

def statesmo(y,model,mode=1,tol=DEFAULT_TOL):
    # mode:
    #   0 - all output.
    #   1 - state smoother.
    #   2 - ARMA EM. %%%% TODO: Ignore diffuse initialization for now.

    #-- Prepare state space matrices and data --#
    n, p, y, mis, anymis, allmis  = prepare_data(y)
    H, Z, T, R, Q, c, a1, P1, stationary, RQdyn  = prepare_model(model,n)

    #-- Kalman filter --#
    a, P, d, Fns, v, invF, L, P_inf, F2, L1 = _kalman(2,n,y,mis,anymis,allmis,H,Z,T,R,Q,c,a1,P1,stationary,RQdyn,tol)

    #-- Preallocate Output Results --#
    Output_r, Output_N  = (False,)*2
    if mode in (0,'all'): # all output
        mode = 0
        Output_r, Output_N  = (True,)*2
    elif mode in (1,'statesmo'): # state smoother
        mode = 1
        Output_r, Output_N  = (True,)*2
    elif mode == 2: # ARMA EM
        Output_N  = True
    if Output_r: Result_r  = [None]*n
    if Output_N: Result_N  = [None]*n

    #-- State smoothing backwards recursion --#
    m   = model.m
    r   = np.matrix(np.zeros((m,1)))
    r1  = np.matrix(np.zeros((m,1)))
    N   = np.matrix(np.zeros((m,m)))
    N1  = np.matrix(np.zeros((m,m)))
    N2  = np.matrix(np.zeros((m,m)))
    alphahat    = np.zeros((m,n))
    V           = np.zeros((m,m,n))
    for t in range(n-1,-1,-1):
        #-- Store output --#
        if Output_r: Result_r[t] = r
        if Output_N: Result_N[t] = N
        if allmis[t]:
            #-- State smoothing when all observations are missing --#
            r   = T[t].T * r
            N   = T[t].T * N * T[t]
            if t > d:
                alphahat[:,[t]] = P[:,:,t] * r
                V[:,:,t]        = P[:,:,t] * N * P[:,:,t]
            else:
                r1  = T[t].T * r1
                N1  = T[t].T * N1 * T[t]
                N2  = T[t].T * N2 * T[t]
                alphahat[:,[t]] = P[:,:,t] * r + P_inf[t] * r1
                P_infN1P        = P_inf[t] * N1 * P[:,:,t]
                V[:,:,t]        = P[:,:,t] * N * P[:,:,t] + P_infN1P.T + P_infN1P + P_inf[t] * N2 * P_inf[t]
        else:
            if anymis[t]: Z[t] = Z[t][~mis[:,t],:]
            if t > d:
                #-- Normal state smoothing --#
                M   = Z[t].T * invF[t]
                r   = M * v[t] + L[t].T * r
                N   = M * Z[t] + L[t].T * N * L[t]
                alphahat[:,[t]] = P[:,:,t] * r
                V[:,:,t]        = P[:,:,t] * N * P[:,:,t]
            else:
                #-- Exact initial state smoothing --#
                if Fns[t]:
                    r1  = Z[t].T * invF[t] * v[t] + L[t].T * r1 + L1[t].T * r
                    r   = L[t].T * r
                    LN  = L[t].T * N1 + L1[t].T * N
                    N2  = Z[t].T * F2[t] * Z[t] + LN * L1[t] + (L[t].T * N2 + L1[t].T * N1) * L[t]
                    N1  = Z[t].T * invF[t] * Z[t] + LN * L[t]
                    N   = L[t].T * N * L[t]
                else: # F_inf[t] = 0
                    M   = Z[t].T * invF[t]
                    r   = M * v[t] + L[t].T * r
                    r1  = T[t].T * r1
                    N   = M * Z[t] + L[t].T * N * L[t]
                    N1  = T[t].T * N1 * L[t]
                    N2  = T[t].T * N2 * T[t]
                alphahat[:,[t]] = P[:,:,t] * r + P_inf[t] * r1
                P_infN1P        = P_inf[t] * N1 * P[:,:,t]
                V[:,:,t]        = P[:,:,t] * N * P[:,:,t] + P_infN1P.T + P_infN1P + P_inf[t] * N2 * P_inf[t]

    alphahat    = a[:,:n] + alphahat
    V           = P[:,:,:n] - V

    #-- Output Results --#
    if mode == 0: # all output
        return L,P,alphahat,V,Result_r,Result_N
    elif mode == 1: # state smoother
        # Build ndarrays from list output
        Result_r   = np.array(np.concatenate(Result_r,1))
        Result_N   = np.concatenate([np.array(x[:,:,None]) for x in Result_N],2)
        return alphahat,V,Result_r,Result_N
    elif mode == 2: # ARMA EM
        return L,P,alphahat,V,Result_N

def disturbsmo(y,model,mode=1,tol=DEFAULT_TOL):
    # if ndims(y) > 2
    #     if any(mis[:]), warning('ssm:ssmodel:disturbsmo:BatchMissing', 'Batch operations with missing data not supported for MATLAB code, set ''usec'' to true.')
    #     %% Data preprocessing %%
    #     N   = size(y, 3);
    #     y   = permute(y, (0,2,1));
    #     %% Batch smoother %%
    #     [temp epshat etahat Result_r] = batchsmo_int(n, N, y, ...
    #         ~issta(model.H), ~issta(model.Z), ~issta(model.T), ~issta(model.R), ~issta(model.Q), ~issta(model.c), ...
    #         getmat(model.H), getmat(model.Z), getmat(model.T), getmat(model.R), getmat(model.Q), getmat(model.c), ...
    #         model.a1.mat, model.P1.mat, opt.tol);
    #     %% Result postprocessing %%
    #     epshat   = permute(epshat, (0,2,1));
    #     etahat   = permute(etahat, (0,2,1));
    #     if nargout > 2
    #         %% Kalman filter %%
    #         [d Fns v invF K L RQ QRt] = kalman_int(3, n, y, mis, anymis, allmis, Hdyn, Zdyn, Tdyn, ~issta(model.R), Qdyn, ~issta(model.c), Hmat, Zmat, Tmat, getmat(model.R), Qmat, getmat(model.c), a1, model.P1.mat, opt.tol);
    #         %% Disturbance smoothing backwards recursion %%
    #         p   = size(y, 1);
    #         m   = size(a1, 1);
    #         rr  = size(model.Q, 1);
    #         N   = np.zeros(m, m);
    #         epsvarhat   = np.zeros(p, p, n);
    #         etavarhat   = np.zeros(rr, rr, n);
    #         Result_N    = cell(1, n);
    #         for t = n : -1 : 1
    #             Result_N[t]     = N;
    #             if Hdyn, H = Hmat[t].copy()
    #             if Zdyn, Z = Zmat[t].copy()
    #             if Tdyn, T = Tmat[t].copy()
    #             if Qdyn, Q = Qmat[t].copy()
    #             etavarhat[:,:,t] = Q - QRt[t] * N * RQ[t];
    #             if allmis[t]
    #                 %% Disturbance smoothing when all observations are missing %%
    #                 epsvarhat[:,:,t] = H; %%%% TODO: What is epsvarhat when all missing?
    #                 N = T.T * N * T
    #             else
    #                 if anymis[t], Z(mis[:,t],:)=[]; H(mis[:,t],:)=[]; H(:,mis[:,t])=[]
    #                 if t > d || ~Fns[t]
    #                     %% Normal disturbance smoothing or when F_inf is zero %%
    #                     epsvarhat(~mis[:,t], ~mis[:,t], t) = H - H * (invF[t] + K[t].T * N * K[t]) * H;
    #                     M = Z.T * invF[t];
    #                     N = M * Z + L[t].T * N * L[t];
    #                 else
    #                     %% Exact initial disturbance smoothing when F_inf is nonsingular %%
    #                     epsvarhat(~mis[:,t], ~mis[:,t], t) = H - H * K[t].T * N * K[t] * H;
    #                     N = L[t].T * N * L[t];
    #                 end
    #                 if anymis[t], if ~Zdyn, Z = Zmat, if ~Hdyn, H = Hmat, end
    #             end
    #         end
    #     end
    # else # ndims(y) <= 2

    # mode 0: all output: epshat,etahat,epsvarhat,etavarhat,Result_r,Result_N
    # mode 1: no Result_r,Result_N
    # mode 2: only epshat,etahat

    #-- Prepare state space matrices and data --#
    n, p, y, mis, anymis, allmis  = prepare_data(y)
    H, Z, T, R, Q, c, a1, P1, stationary, RQdyn  = prepare_model(model,n)

    if mode in (0,1):
        #-- Kalman filter --#
        d,Fns,v,invF,K,L,RQ,QRt = _kalman(3,n,y,mis,anymis,allmis,H,Z,T,R,Q,c,a1,P1,stationary,RQdyn,tol)

        #-- Disturbance smoothing backwards recursion --#
        m   = model.m
        rr  = Q[0].shape[0]
        r   = np.matrix(np.zeros((m,1)))
        N   = np.matrix(np.zeros((m,m)))
        epshat      = np.zeros((p,n))
        epsvarhat   = np.zeros((p,p,n))
        etahat      = np.zeros((rr,n))
        etavarhat   = np.zeros((rr,rr,n))
        if mode == 0:
            Result_r = [None]*n
            Result_N = [None]*n
        for t in range(n-1,-1,-1):
            if mode == 0:
                Result_r[t] = r
                Result_N[t] = N
            etahat[:,[t]]    = QRt[t] * r
            etavarhat[:,:,t] = Q[t] - QRt[t] * N * RQ[t]
            if allmis[t]:
                #-- Disturbance smoothing when all observations are missing --#
                epshat[:,t] = 0
                epsvarhat[:,:,t] = H[t].copy() #### TODO: What is epsvarhat when all missing?
                r = T[t].T * r
                N = T[t].T * N * T[t]
            else:
                if anymis[t]:
                    # "Disable" parts of state space matrices that corresponds to missing elements in the observation vector
                    Z[t]   = Z[t][~mis[:,t],:]
                    H[t]   = H[t][np.ix_(~mis[:,t],~mis[:,t])]
                if t > d or not Fns[t]:
                    #-- Normal disturbance smoothing or when F_inf is zero --#
                    epshat[~mis[:,t],t] = H[t] * (invF[t] * v[t] - K[t].T * r)
                    epsvarhat[np.ix_(~mis[:,t],~mis[:,t],[t])] = H[t] - H[t] * (invF[t] + K[t].T * N * K[t]) * H[t]
                    M = Z[t].T * invF[t]
                    r = M * v[t] + L[t].T * r
                    N = M * Z[t] + L[t].T * N * L[t]
                else:
                    #-- Exact initial disturbance smoothing when F_inf is nonsingular --#
                    epshat[~mis[:,t],t] = -H[t] * K[t].T * r
                    epsvarhat[np.ix_(~mis[:,t],~mis[:,t],[t])] = H[t] - H[t] * K[t].T * N * K[t] * H[t]
                    r = L[t].T * r
                    N = L[t].T * N * L[t]
    # elif mode == 2:
    #     [epshat etahat] = fastdisturbsmo_int(n, y, mis, anymis, allmis, Hdyn, Zdyn, Tdyn, ~issta(model.R), Qdyn, ~issta(model.c), Hmat, Zmat, Tmat, getmat(model.R), Qmat, getmat(model.c), a1, model.P1.mat, opt.tol);

    if mode == 0:
        return epshat,etahat,epsvarhat,etavarhat,Result_r,Result_N
    elif mode == 1:
        return epshat,etahat,epsvarhat,etavarhat
    elif mode == 2:
        return epshat,etahat

def simsmo(N,y,model,antithetic=1,tol=DEFAULT_TOL):
    # function [alphatilde epstilde etatilde alphaplus] = simsmo(y, model, N, varargin)
    # Znl     = isa(model.Z, 'ssfunc');
    # Tnl     = isa(model.T, 'ssfunc');
    # if opt.usec && ~Znl && ~Tnl
    #     [alphatilde epstilde etatilde alphaplus] = simsmo_int_c(y, N, getmat_c(model.H), Hdyn, getmat_c(model.Z), Zdyn, ...
    #         getmat_c(model.T), Tdyn, getmat_c(model.R), Rdyn, getmat_c(model.Q), Qdyn, getmat_c(model.c), cdyn, ...
    #         getmat_c(model.a1), getmat_c(model.P1), opt.antithetic, opt.tol, opt.tol, opt.inv, true);

    #-- Get the current matrix values --#
    n, p, y, mis, anymis, allmis  = prepare_data(y)
    m  = model.m
    r  = model.r
    H, Z, T, R, Q, c, a1, P1, stationary, RQdyn  = prepare_model(model,n)
    Hdyn = model.H.dynamic
    Zdyn = model.Z.dynamic
    Qdyn = model.Q.dynamic
    cdyn = model.c.dynamic

    #-- Data preprocessing --#
    Nsamp = int(np.ceil(N/2.0)) if antithetic >= 1 else N

    #-- Unconditional sampling --#
    yplus,alphaplus,epsplus,etaplus = _sample(Nsamp,n,p,m,r,H,Z,T,R,Q,c,a1,P1,stationary,Hdyn,Zdyn,RQdyn,Qdyn,cdyn)
    # yplus,alphaplus,epsplus,etaplus = _sample(Nsamp,n,p,m,r, Znl, Tnl, model.Z, model.T, Hdyn, Zdyn, Tdyn, Rdyn, Qdyn, cdyn, Hmat, Rmat, Qmat, cmat, a1, P1);

    #-- Fast (and batch) state and disturbance smoothing --#
    alphahat,epshat,etahat = _fastsmo(n,y,mis,anymis,allmis,m,H,Z,T,R,Q,c,a1,P1,stationary,RQdyn,tol=tol)
    alphaplushat,epsplushat,etaplushat = _batchsmo(1,n,Nsamp,yplus,m,H,Z,T,R,Q,c,a1,P1,stationary,RQdyn,cdyn,tol=tol)

    #-- Calculate sampled disturbances, states and observations --#
    if antithetic >= 1:
        # The "[:,:,None]" is required to add the 3rd dimension before tiling, else the end result would remain 2D
        epstilde    = np.tile(epshat[:,:,None],(1,1,2*Nsamp)) + np.hstack([-epsplushat + epsplus, epsplushat - epsplus]).transpose((0,2,1))
        etatilde    = np.tile(etahat[:,:,None],(1,1,2*Nsamp)) + np.hstack([-etaplushat + etaplus, etaplushat - etaplus]).transpose((0,2,1))
        alphatilde  = np.tile(alphahat[:,:,None],(1,1,2*Nsamp)) + np.hstack([-alphaplushat + alphaplus, alphaplushat - alphaplus]).transpose((0,2,1))
        if (N % 2) == 1:
            epstilde   = epstilde[:,:,:-1]
            etatilde   = etatilde[:,:,:-1]
            alphatilde = alphatilde[:,:,:-1]
    else: # antithetic <= 0
        epstilde    = np.tile(epshat[:,:,None],(1,1,N)) - (epsplushat - epsplus).transpose((0,2,1))
        etatilde    = np.tile(etahat[:,:,None],(1,1,N)) - (etaplushat - etaplus).transpose((0,2,1))
        alphatilde  = np.tile(alphahat[:,:,None],(1,1,N)) - (alphaplushat - alphaplus).transpose((0,2,1))
    alphaplus = alphaplus.transpose((0,2,1))

    return alphatilde,epstilde,etatilde,alphaplus

def signal(alpha, model, t0=0, mcom=None):
    # %@SSMODEL/SIGNAL Retrieve signal components.
    # %       alpha is the state sequence.
    # %       model is the linear Gaussian model to use.
    # %       t0 is optional time offset for dynamic Z.
    # %       ycom is p*n*M where M is number of signal components, unless p == 1,
    # %           in which case ycom is M*n.

    # if opt.usec
    #     ycom    = signal_int_c(alpha, model.mcom, getmat_c(model.Z), ~issta(model.Z), t0, true);
    # else
    n       = alpha.shape[1]
    if mcom == 'all':
        mcom  = [model.m]
    elif mcom is None:
        mcom  = model.mcom
    ncom    = len(mcom)
    mcom    = np.cumsum([0] + mcom)
    Zmat    = model.Z.mat
    if model.Z.dynamic:
        p       = Zmat[0].shape[0]
        ycom    = np.zeros((p, n, ncom))
        for t in range(n):
            Z   = Zmat[t0 + t]
            for i in range(ncom):
                ycom[:,[t],i] = Z[:,mcom[i]:mcom[i+1]]*alpha[mcom[i]:mcom[i+1],[t]]
    else:
        p       = Zmat.shape[0]
        ycom    = np.zeros((p, n, ncom))
        for i in range(ncom):
            ycom[:,:,i] = Zmat[:,mcom[i]:mcom[i+1]]*alpha[mcom[i]:mcom[i+1],:]

    return ycom.transpose((2,1,0)).squeeze() if p == 1 else ycom

def signalvar(V, model, t0=0, mcom=None):
    """Retrieve signal component variances.
    V is the state variance sequence.
    model is the linear Gaussian model to use.
    t0 is optional time offset for dynamic Z.
    ycomvar is p*p*n*M where M is number of signal components, unless p == 1, in which case ycomvar is M*n.
    """
    n       = V.shape[2]
    p       = model.p
    if mcom == 'all':
        mcom  = [model.m]
    elif mcom is None:
        mcom  = model.mcom
    ncom    = len(mcom)
    mcom    = np.cumsum([0] + mcom)
    ycom    = np.zeros((p, p, n, ncom))
    for t in range(n):
        if model.Z.dynamic:
            Z   = model.Z.mat[t0 + t]
        else:
            Z   = model.Z.mat
        for i in range(ncom):
            ycom[:,:,[t],i] = Z[:,mcom[i]:mcom[i+1]]*V[mcom[i]:mcom[i+1],mcom[i]:mcom[i+1],t]*Z[:,mcom[i]:mcom[i+1]].T

    return ycom.transpose((3,2,0,1)).squeeze() if p == 1 else ycom
