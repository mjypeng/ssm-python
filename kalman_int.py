# -*- coding: utf-8 -*-

import numpy as np

def kalman_int(mode,n,y,Hdyn,Zdyn,Tdyn,Rdyn,Qdyn,cdyn,Hmat,Zmat,Tmat,Rmat,Qmat,cmat,a,P,tol):
    # function varargout = kalman_int(mis, anymis, allmis, , )
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

    Output_a     = False
    Output_P     = False
    Output_v     = False
    Output_invF  = False
    Output_K     = False
    Output_L     = False
    Output_Pinf  = False
    Output_F2    = False
    Output_L1    = False
    Output_logL_ = False
    Output_var_  = False
    Output_RQ    = False
    Output_QRt   = False
    Output_RQRt  = False

    if mode == 0: # all output
        Output_a     = True
        Output_P     = True
        Output_v     = True
        Output_invF  = True
        Output_K     = True
        Output_L     = True
        Output_Pinf  = True
        Output_F2    = True
        Output_L1    = True
        Output_logL_ = True
        Output_var_  = True
        Output_RQ    = True
        Output_QRt   = True
        Output_RQRt  = True
    elif mode == 1: # Kalman filter
        Output_a     = True
        Output_P     = True
        Output_v     = True
        Output_invF  = True
    elif mode == 2: # state smoother
        Output_a     = True
        Output_P     = True
        Output_v     = True
        Output_invF  = True
        Output_L     = True
        Output_Pinf  = True
        Output_F2    = True
        Output_L1    = True
    elif mode == 3: # disturbance smoother
        Output_v     = True
        Output_invF  = True
        Output_K     = True
        Output_L     = True
        Output_RQ    = True
        Output_QRt   = True
    elif mode == 4: # loglikelihood
        Output_logL_ = True
        Output_var_  = True
    elif mode == 5: # fast smoother
        Output_v     = True
        Output_invF  = True
        Output_K     = True
        Output_L     = True
        Output_L1    = True
        Output_QRt   = True
    elif mode == 6: # fast state smoother
        Output_v     = True
        Output_invF  = True
        Output_L     = True
        Output_L1    = True
        Output_RQRt  = True
    elif mode == 7: # fast disturbance smoother
        Output_v     = True
        Output_invF  = True
        Output_K     = True
        Output_L     = True
        Output_QRt   = True
    elif mode == 8: # loglikelihood gradient
        Output_v     = True
        Output_invF  = True
        Output_K     = True
        Output_L     = True
        Output_logL_ = True
        Output_var_  = True

    #-- Initialization --#
    D       = (P == np.inf)
    init    = D.any() # use exact initialization if init = true.
    if init:
        d     = n # 0-index: t is in [0,n-1]
        P[D]  = 0
        P_inf = D.astype(float)
    else:
        d     = 0
    stationary  = not (Hdyn or Zdyn or Tdyn or Rdyn or Qdyn) # c does not effect convergence of P
    converged   = False
    RQdyn       = Rdyn or Qdyn
    if not Hdyn: H = Hmat
    if not Zdyn: Z = Zmat
    if not Tdyn: T = Tmat
    if not Rdyn: R = Rmat
    if not Qdyn: Q = Qmat
    if not cdyn: c = cmat
    RQRt = R*(Q*R.T)
    if not RQdyn: RQRt = R*(Q*R.T)

    #-- Preallocate Output Results --#
    if Output_a:
        Result_a           = np.matrix(np.zeros((a.shape[0],n+1)))
        Result_a[:,0]      = a
    if Output_P:
        Result_P           = np.zeros(P.shape + (n+1,))
        Result_P[:,:,0]    = P
    if Output_v:     Result_v     = [None]*n
    if Output_invF:  Result_invF  = [None]*n
    if Output_K:     Result_K     = [None]*n
    if Output_L:     Result_L     = [None]*n
    if Output_Pinf:
        Result_Pinf = [P_inf] if init else [np.matrix(np.zeros(P.shape))] # Length will be d
    if Output_F2:    Result_F2    = [] # length d
    if Output_L1:    Result_L1    = [] # length d
    if Output_logL_: Result_logL_ = 0
    if Output_var_:  Result_var_  = 0
    if Output_RQ:
        if RQdyn: Result_RQ   = [None]*n
        else:     Result_RQ   = [R*Q]*n
    if Output_QRt:
        if RQdyn: Result_QRt  = [None]*n
        else:     Result_QRt  = [Q*R.T]*n
    if Output_RQRt:
        if RQdyn: Result_RQRt = [None]*n
        else:     Result_RQRt = [RQRt]*n

    Fns     = np.zeros((1,n),dtype=bool) # Is F_inf nonsingular for each iteration

    #-- Kalman filter loop --#
    for t in range(n):
        if Hdyn: H = Hmat[t]
        if Zdyn: Z = Zmat[t]
        if Tdyn: T = Tmat[t]
        if Rdyn: R = Rmat[t]
        if Qdyn: Q = Qmat[t]
        if cdyn: c = cmat[t]
        if RQdyn: RQRt = R*(Q*R.T)
    #     if converged, converged = ~anymis(t); end % Any missing observation throws off the steady state
        if converged:
            #-- Kalman filter after steady state reached --#
            v   = y[:,t] - Z*a
            a   = c + T*a + K*v
    #     elseif allmis(t)
    #         %% Kalman filter when all observations are missing %%
    #         v(:)    = 0;
    #         F       = Z*(P*Z') + H;
    #         invF    = inv(F); %%%% TODO: Ignore diffuse initialization for now
    #         K(:)    = 0;
    #         L       = T;
    #         a       = c + T*a;
    #         P       = T*P*T' + RQRt;
    #         if init, P_inf = T*P_inf*T'; end
        else:
    #         if anymis(t), Z(mis(:, t), :)=[]; H(mis(:, t), :)=[]; H(:, mis(:, t))=[]; end
            if init:
                #-- Exact initial Kalman filter --#
                M       = P*Z.T
                M_inf   = P_inf*Z.T
                A_inf   = T*P_inf
                if (abs(M_inf) < tol).all(): # F_inf is zero
                    F       = Z*M + H
                    invF    = F.I # The real invF
                    F2[:]   = 0
                    K       = T*M*invF
                    K1[:]   = 0
                    L       = T - K*Z
                    L1[:]   = 0
                    P       = T*P*L.T + RQRt
                    P_inf   = A_inf*T.T
                else: # F_inf is assumed to be nonsingular
                    Fns[t]  = True
                    invF    = (Z*M_inf).I # This is actually invF1
                    F       = Z*M + H
                    F2      = -invF*F*invF
                    K       = T*M_inf*invF
                    K1      = T*(M*invF + M_inf*F2)
                    L       = T - K*Z
                    L1      = -K1*Z
                    P       = T*P*L.T + A_inf*L1.T + RQRt
                    P_inf   = A_inf*L.T
                if (abs(P_inf) < tol).all():
                    d    = t
                    init = False
            else:
                #-- Normal Kalman filter --#
                M       = P*Z.T
                F       = Z*M + H
                invF    = F.I
                K       = T*M*invF
                L       = T - K*Z
                prevP   = P
                P       = T*P*L.T + RQRt
                if stationary and (abs(P-prevP) < tol).all(): converged = True

            #-- Kalman data filter --#
            v = y[:,t] - Z*a
            a = c + T*a + K*v
    #         v   = y(~mis(:, t), t) - Z*a;
    #         if anymis(t), if ~Zdyn, Z = Zmat; end, if ~Hdyn, H = Hmat; end, end
    #     end
        #-- Store results for this time point --#
        if Output_a:    Result_a[:,t+1]    = a
        if Output_P:    Result_P[:,:,t+1]  = P
        if Output_v:    Result_v[t]        = v
        if Output_invF: Result_invF[t]     = invF
        if Output_K:    Result_K[t]        = K
        if Output_L:    Result_L[t]        = L
        if t <= d:
            if Output_Pinf: Result_Pinf.append(P_inf)
            if Output_F2:   Result_F2.append(F2)
            if Output_L1:   Result_L1.append(L1)
        if True: # ~allmis(t)
            if Output_logL_:
                detinvF     = np.linalg.det(invF)
                if detinvF > 0:   Result_logL_ = Result_logL_ - np.log(detinvF)
                elif detinvF < 0: Result_logL_ = np.nan
            if Output_var_:
                if t > d or not Fns[t]: Result_var_ = Result_var_ + v.T*invF*v
        if RQdyn:
            if Output_RQ:   Result_RQ[t]   = R*Q
            if Output_QRt:  Result_QRt[t]  = Q*R.T
            if Output_RQRt: Result_RQRt[t] = RQRt

    #-- Output Results --#
    if mode == 0: # all output
        return Result_a,Result_P,d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_Pinf,Result_F2,Result_L1,Result_logL_+Result_var_,Result_var_,Result_RQ,Result_QRt,Result_RQRt
    elif mode == 1: # Kalman filter
        return Result_a,Result_P,d,Result_v,Result_invF
    elif mode == 2: # state smoother
        return Result_a,Result_P,d,Fns,Result_v,Result_invF,Result_L,Result_Pinf,Result_F2,Result_L1
    elif mode == 3: # disturbance smoother
        return d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_RQ,Result_QRt
    elif mode == 4: # loglikelihood
        return Result_logL_+Result_var_,Result_var_
    elif mode == 5: # fast smoother
        return d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_L1,Result_QRt
    elif mode == 6: # fast state smoother
        return d,Fns,Result_v,Result_invF,Result_L,Result_L1,Result_RQRt
    elif mode == 7: # fast disturbance smoother
        return d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_QRt
    elif mode == 8: # loglikelihood gradient
        return d,Fns,Result_v,Result_invF,Result_K,Result_L,Result_logL_+Result_var_,Result_var_

# nile data
y = np.matrix([1120,1160,963,1210,1160,1160,813,1230,1370,1140,995,935,1110,994,1020,960,1180,799,958,1140,1100,1210,1150,1250,1260,1220,1030,1100,774,840,874,694,940,833,701,916,692,1020,1050,969,831,726,456,824,702,1120,1100,832,764,821,768,845,864,862,698,845,744,796,1040,759,781,865,845,944,984,897,822,1010,771,676,649,846,812,742,801,1040,860,874,848,890,744,749,838,1050,918,986,797,923,975,815,1020,906,901,1170,912,746,919,718,714,740])

a,P,d,v,invF = kalman_int(1,y.shape[1],y,False,False,False,False,False,False,np.matrix(1),np.matrix(1),np.matrix(1),np.matrix(1),np.matrix(0.1),np.matrix(0),np.matrix(0),np.matrix(np.inf),10**-4)
