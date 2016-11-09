# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ssmodel as ssm

mpl.rcParams['figure.figsize'] = (16,10)
SILENT_OUTPUT = bool(sys.argv[1] if len(sys.argv)>1 else 0)
if SILENT_OUTPUT:
    run_name = '_'+sys.argv[2] if len(sys.argv)>2 else ''
    fout  = open('demo_nile'+run_name+'_out.txt','w')
    mpl.rcParams['savefig.bbox']        = 'tight'
    mpl.rcParams['savefig.dpi']         = 150
    mpl.rcParams['savefig.format']      = 'png'
    mpl.rcParams['savefig.pad_inches']  = 0.1
else:
    fout  = sys.stdout

fout.write('\n')

#-- Load data --#
y       = np.loadtxt('data/nile.dat')[:,None].T
time    = range(1871,1971)

#-- Maximum loglikelihood estimation --#
llm       = ssm.model_llm()
opt_x     = ssm.estimate(y,llm,np.log([10000,5000])/2,method='Nelder-Mead')[0]
llm       = ssm.set_param(llm,opt_x)
logL,fvar = ssm.loglik(y,llm)

fout.write("Loglikelihood = %g, variance = %g.\n" % (logL,fvar))
fout.write("epsilon variance = %g, eta variance = %g.\n" % (llm['H']['mat'][0,0],llm['Q']['mat'][0,0]))

#-- Kalman filtering --#
a,P,v,F  = ssm.kalman(y,llm)
# Reshape output for plotting
a     = a.squeeze()
P     = P.squeeze()
sqrtP = np.sqrt(P)
v     = v.squeeze()
F     = F.squeeze()

fig = plt.figure(num='Filtered state')
ax1 = plt.subplot(221)
plt.plot(time,y.tolist()[0],'r:',label='nile')
plt.plot(time+[1971],a,'b-',label='filt. state')
plt.plot(time+[1971],a+1.645*sqrtP,'g:',label='90% conf. +')
plt.plot(time+[1971],a-1.645*sqrtP,'g:',label='90% conf. -')
plt.title('Filtered state'); plt.ylim([450,1400])
plt.legend()

ax2 = plt.subplot(222)
plt.plot(time+[1971],P)
plt.title('Filtered state variance'); plt.ylim([5000,17500])

ax3 = plt.subplot(223)
plt.plot(time,v)
plt.title('Prediction errors'); plt.ylim([-450,400])

ax4 = plt.subplot(224)
plt.plot(time,F)
plt.title('Prediction error variance'); plt.ylim([20000,32500])

if SILENT_OUTPUT:
    plt.savefig('demo_nile'+run_name+'_out01.png')
    plt.close()
else:
    plt.show()

#-- State smoothing --#
alphahat,V,r,N    = ssm.statesmo(1,y,llm)
# Reshape output for plotting
alphahat = alphahat.squeeze()
V        = V.squeeze()
r        = r.squeeze()
N        = N.squeeze()

fig = plt.figure(num='Smoothed state')
ax1 = plt.subplot(221)
plt.plot(time,y.tolist()[0],'r:',label='nile')
plt.plot(time,alphahat,label='smo. state')
plt.plot(time,alphahat+1.645*np.sqrt(V),'g:',label='90% conf. +')
plt.plot(time,alphahat-1.645*np.sqrt(V),'g:',label='90% conf. -')
plt.title('Smoothed state'); plt.ylim([450,1400]); plt.legend()

ax2 = plt.subplot(222)
plt.plot(time,V)
plt.title('Smoothed state variance'); plt.ylim([2300,4100])
ax3 = plt.subplot(223)
plt.plot(time,r)
plt.title('Smoothing cumulant'); plt.ylim([-0.036,0.024])
ax4 = plt.subplot(224)
plt.plot(time,N)
plt.title('Smoothing variance cumulant'); plt.ylim([0,0.000105])

if SILENT_OUTPUT:
    plt.savefig('demo_nile'+run_name+'_out02.png')
    plt.close()
else:
    plt.show()

#-- Disturbance smoothing --#
epshat,etahat,epsvarhat,etavarhat = ssm.disturbsmo(1,y,llm)
# Reshape output for plotting
epshat = epshat.squeeze()
etahat = etahat.squeeze()
epsvarhat = epsvarhat.squeeze()
etavarhat = etavarhat.squeeze()

fig = plt.figure(num='Smoothed disturbances')
ax1 = plt.subplot(221)
plt.plot(time,epshat)
plt.title('Observation error'); plt.ylim([-360,280])
ax2 = plt.subplot(222)
plt.plot(time,epsvarhat)
plt.title('Observation error variance'); plt.ylim([2300,4100])
ax3 = plt.subplot(223)
plt.plot(time,etahat)
plt.title('State error'); plt.ylim([-50,35])
ax4 = plt.subplot(224)
plt.plot(time,etavarhat)
plt.title('State error variance'); plt.ylim([1225,1475])

if SILENT_OUTPUT:
    plt.savefig('demo_nile'+run_name+'_out03.png')
    plt.close()
else:
    plt.show()

#-- Simulation smoothing --#
NN = 5 if SILENT_OUTPUT else 1
for ii in range(NN):
    alphatilde,epstilde,etatilde,alphaplus = ssm.simsmo(1,y,llm)
    alphatilde  = alphatilde.squeeze()
    epstilde    = epstilde.squeeze()
    etatilde    = etatilde.squeeze()
    alphaplus   = alphaplus.squeeze()

    fig = plt.figure(num='Simulation')
    ax1 = plt.subplot(221)
    plt.plot(time,alphahat,label='samp. state')
    plt.scatter(time,alphaplus+alphahat[0]-alphaplus[0],8,'r','s','filled',label='nile')
    plt.title('Unconditioned sampled state'); plt.legend()
    ax2 = plt.subplot(222)
    plt.plot(time, alphahat,label='disp. samp. state')
    plt.scatter(time,alphatilde, 8, 'r', 's', 'filled',label='nile')
    plt.title('Conditioned sampled state'); plt.ylim([740,1160]); plt.legend()
    ax3 = plt.subplot(223)
    plt.plot(time, epshat,label='smo. obs. disturb.')
    plt.scatter(time,epstilde,8,'r','s','filled',label='samp. obs. disturb.')
    plt.title('Conditioned sampled observation error'); plt.ylim([-440,280]); plt.legend()
    ax4 = plt.subplot(224)
    plt.plot(time, etahat,label='smo. state disturb.')
    plt.scatter(time,etatilde,8,'r','s','filled',label='samp. state disturb.')
    plt.title('Conditioned sampled state error'); plt.ylim([-440,280]); plt.legend()

    if SILENT_OUTPUT:
        plt.savefig('demo_nile'+run_name+'_out04-'+str(ii)+'.png')
        plt.close()
    else:
        plt.show()

#-- Missing Observations --#
ymis    = y.astype(float).copy()
ymis[:,range(21,41)+range(61,81)] = np.nan
amis,Pmis        = ssm.kalman(ymis,llm)[:2]
alphahatmis,Vmis = ssm.statesmo(1,ymis,llm)[:2]
amis        = amis.squeeze()
Pmis        = Pmis.squeeze()
alphahatmis = alphahatmis.squeeze()
Vmis        = Vmis.squeeze()

fig = plt.figure(num='Filtering and smoothing of data with missing observations')
ax1 = plt.subplot(221)
plt.plot(time,ymis.tolist()[0],'r:',label='nile w/ miss. values')
plt.plot(time+[1971],amis,label='filt. state')
plt.title('Filtered state (extrapolation)'); plt.ylim([450,1400]); plt.legend()
ax2 = plt.subplot(222)
plt.plot(time+[1971],Pmis)
plt.title('Filtered state variance'); plt.ylim([4000,36000])
ax3 = plt.subplot(223)
plt.plot(time, ymis.tolist()[0], 'r:',label='nile w/ miss. values')
plt.plot(time, alphahatmis,label='smo. state')
plt.title('Smoothed state (interpolation)'); plt.ylim([450,1400]); plt.legend()
ax4 = plt.subplot(224)
plt.plot(time, Vmis)
plt.title('Filtered state (extrapolation)'); plt.ylim([2000,10000])

if SILENT_OUTPUT:
    plt.savefig('demo_nile'+run_name+'_out05.png')
    plt.close()
else:
    plt.show()

#-- Forecasting (equivalent to future missing values) --#
yforc   = np.hstack([y,np.tile(np.nan,(1,50))])
aforc,Pforc,vforc,Fforc = ssm.kalman(yforc,llm)
# Reshape output for plotting
aforc     = aforc.squeeze()
Pforc     = Pforc.squeeze()
sqrtPforc = np.sqrt(Pforc)
vforc     = vforc.squeeze()
Fforc     = Fforc.squeeze()

fig = plt.figure(num='Forecasting')
ax1 = plt.subplot(221)
plt.plot(time+range(1972,2022), yforc.tolist()[0], 'r:',label='nile')
plt.plot(time+range(1972,2023), aforc,label='forecast')
plt.plot(time+range(1972,2023), np.hstack([np.tile(np.nan,len(time)),aforc[-51:]+0.675*sqrtPforc[-51:]]), 'g:',label='50% conf. +')
plt.title('State forecast'); plt.xlim([1868,2026]); plt.ylim([450,1400])
plt.plot(time+range(1972,2023), np.hstack([np.tile(np.nan,len(time)),aforc[-51:]-0.675*sqrtPforc[-51:]]), 'g:',label='50% conf. -')
plt.title('State forecast'); plt.xlim([1868,2026]); plt.ylim([450,1400]); plt.legend()
ax2 = plt.subplot(222)
plt.plot(time+range(1972,2023), Pforc)
plt.title('State variance'); plt.xlim([1868,2026]); plt.ylim([4000,80000])
ax3 = plt.subplot(223)
plt.plot(time+range(1972,2023), aforc)
plt.title('Observation forecast'); plt.xlim([1868,2026]); plt.ylim([700,1200])
ax4 = plt.subplot(224)
plt.plot(time+range(1972,2022), Fforc)
plt.title('Observation forecast variance'); plt.xlim([1868,2026]); plt.ylim([20000,96000])

if SILENT_OUTPUT:
    plt.savefig('demo_nile'+run_name+'_out06.png')
    plt.close()
else:
    plt.show()

fout.write('\n')

if SILENT_OUTPUT: fout.close()
