# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.linalg import sqrtm
from scipy import stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import ssm

mpl.rcParams['figure.figsize'] = (16,10)
SILENT_OUTPUT = bool(sys.argv[1] if len(sys.argv)>1 else 0)
if SILENT_OUTPUT:
    run_name = '_'+sys.argv[2] if len(sys.argv)>2 else ''
    fout  = open('demo_seatbelt'+run_name+'_out.txt','w')
    mpl.rcParams['savefig.bbox']        = 'tight'
    mpl.rcParams['savefig.dpi']         = 150
    mpl.rcParams['savefig.format']      = 'png'
    mpl.rcParams['savefig.pad_inches']  = 0.1
    fig_count  = 0
    def _output_fig(count):
        count  += 1
        plt.savefig('demo_seatbelt'+run_name+("_out%02d" % count)+'.png')
        plt.close()
        return count
    output_fig  = lambda : _output_fig(fig_count)
else:
    fout        = sys.stdout
    output_fig  = plt.show

fout.write('\n')

#-- Load data --#
seatbelt = np.loadtxt('data/seatbelt.dat').T
time     = pd.date_range(pd.to_datetime('19690101'),pd.to_datetime('19850101'),freq='MS')

#-- Analysis of drivers series --#
y       = seatbelt[[0],:]

#-- Estimation of basic structural time series model --#
bstsm   = ssm.model_stsm('level', 'trig1', 12) #ssm.model_cat([ssm.model_llm(),ssm.model_seasonal('trig1', 12)])
bstsm   = ssm.estimate(y, bstsm, np.log([0.003,0.0009,5e-7])/2, method='Nelder-Mead',options={'disp':not SILENT_OUTPUT})[0]
fout.write("epsilon variance = %g, eta variance = %g, omega variance = %g.\n\n" % (bstsm['H']['mat'][0,0],bstsm['Q']['mat'][0,0],bstsm['Q']['mat'][1,1]))

a,P,v,F         = ssm.kalman(y,bstsm)
alphahat,V,r,N  = ssm.statesmo(y,bstsm)
#-- Retrieve components --#
ycom        = ssm.signal(a, bstsm)
lvl         = ycom[0,:]
seas        = ycom[1,:]
ycomhat     = ssm.signal(alphahat, bstsm)
lvlhat      = ycomhat[0,:]
seashat     = ycomhat[1,:]

irr,etahat,epsvarhat,etavarhat = ssm.disturbsmo(y,bstsm)
irr       = irr.squeeze()
epsvarhat = epsvarhat.squeeze()

fig = plt.figure(num='Estimated Components')
ax1 = plt.subplot(311)
plt.plot(time[:-1], y.tolist()[0], 'r:', label='drivers')
plt.plot(time[:-1], lvlhat, label='est. level')
plt.title('Level'); plt.ylim([6.875, 8]); plt.legend()
ax2 = plt.subplot(312)
plt.plot(time[:-1], seashat)
plt.title('Seasonal'); plt.ylim([-0.16, 0.28])
ax3 = plt.subplot(313)
plt.plot(time[:-1], irr)
plt.title('Irregular'); plt.ylim([-0.15, 0.15])

fig_count  = output_fig()

fig = plt.figure(num='Data and level')
plt.plot(time, lvl, label='filtered level')
plt.plot(time[:-1], lvlhat, ':', label='smoothed level')
plt.scatter(time[:-1], y.tolist()[0],c='r',marker='+', label='drivers')
plt.ylim([6.95,7.9]); plt.legend()

fig_count  = output_fig()

#-- Calculate standardized residuals --#
u       = irr/np.sqrt(epsvarhat)
r       = np.zeros((12,y.shape[1]))
for t in range(y.shape[1]): r[:,[t]] = np.asmatrix(sqrtm(etavarhat[:,:,t])).I*etahat[:,[t]]
comres  = ssm.signal(r, bstsm)
lvlres  = comres[0,:].squeeze()

fig = plt.figure(num='Residuals')
ax1 = plt.subplot(311)
plt.plot(time[:-1], np.asarray(y).squeeze() - lvl[:-1] - seas[:-1])
plt.title('One-step ahead prediction residuals'); plt.xlim([time[0],time[-2]]); plt.ylim([-0.35,0.25])
ax2 = plt.subplot(312)
plt.plot(time[:-1], u)
plt.title('Auxiliary irregular residuals'); plt.xlim([time[0],time[-2]]); plt.ylim([-4.5,4.5])
ax3 = plt.subplot(313)
plt.plot(time[:-1], lvlres)
plt.title('Auxiliary level residuals'); plt.xlim([time[0],time[-2]]); plt.ylim([-2.5,1.5])

fig_count  = output_fig()

#-- Adding explanatory variables and intervention to the model --#
petrol   = seatbelt[[4],:]
bstsmir  = ssm.model_cat([bstsm,ssm.model_intv(y.shape[1],'step',169),ssm.model_reg(petrol)])
bstsmir,res = ssm.estimate(y, bstsmir, np.log([0.004,0.00027,1e-6])/2, method='Nelder-Mead',options={'disp':not SILENT_OUTPUT})
logL     = res['logL']

alphahatir,Vir  = ssm.statesmo(y,bstsmir)[:2]
irrir           = ssm.disturbsmo(y,bstsmir)[0]
irrir   = irrir.squeeze()
ycomir  = ssm.signal(alphahatir, bstsmir)
lvlir   = np.sum(ycomir[[0,2,3],:],0).squeeze()
seasir  = ycomir[1,:].squeeze()

fout.write('[Analysis on drivers series]\n')
fout.write("Loglikelihood: %g\n" % logL)
fout.write("Irregular variance: %g\n" % bstsmir['H']['mat'][0,0])
fout.write("Level variance: %g\n" % bstsmir['Q']['mat'][0,0])
fout.write("Seasonal variance: %g\n" % bstsmir['Q']['mat'][1,1])
fout.write('Variable             Coefficient     R. m. s. e.     t-value\n')
tval = alphahatir[-1,0]/np.sqrt(Vir[-1,-1,-1])
pval = stats.t.sf(np.abs(tval),y.shape[1] - bstsmir.m)*2
fout.write("petrol coefficient   %-14.5f  %-14.5f  %.5f [%.5f]\n" % (alphahatir[-1,0],np.sqrt(Vir[-1,-1,-1]),tval,pval))
tval = alphahatir[-2,0]/np.sqrt(Vir[-2,-2,-1])
pval = stats.t.sf(np.abs(tval),y.shape[1] - bstsmir.m)*2
fout.write("level shift at 83.2  %-14.5f  %-14.5f  %.5f [%.5f]\n\n" % (alphahatir[-2,0],np.sqrt(Vir[-2,-2,-1]),tval,pval))

fig = plt.figure(num='Estimated Components w/ intervention and regression')
plt.subplot(311)
plt.plot(time[:-1], np.asarray(y).squeeze(), 'r:', label='drivers')
plt.plot(time[:-1], lvlir, label='est. level')
plt.title('Level'); plt.xlim([time[0],time[-1]]); plt.ylim([6.875,8])
plt.legend()
plt.subplot(312)
plt.plot(time[:-1], seasir)
plt.title('Seasonal'); plt.xlim([time[0],time[-1]]); plt.ylim([-0.16,0.28])
plt.subplot(313)
plt.plot(time[:-1], irrir)
plt.title('Irregular'); plt.ylim([-0.15,0.15])

fig_count  = output_fig()

irr,etahat,epsvarhat,etavarhat = ssm.disturbsmo(y,bstsmir)
r       = np.zeros((12,y.shape[1]))
for t in range(y.shape[1]): r[:,[t]] = np.asmatrix(sqrtm(etavarhat[:,:,t])).I*etahat[:,[t]]
comres  = ssm.signal(r, bstsm, [1,11])
lvlres  = comres[0,:].squeeze()

fig = plt.figure(num='Estimated Components w/ intervention and regression')
ax1 = plt.subplot(211)
plt.plot(time[:-1], irrir)
plt.title('Irregular'); plt.xlim([time[0],time[-1]]); plt.ylim([-0.15,0.15])
ax2 = plt.subplot(212)
plt.plot(time[:-1], lvlres)
plt.title('Normalized level residuals'); plt.xlim([time[0],time[-1]]); plt.ylim([-1.5,1])

fig_count  = output_fig()

#-- Analysis of both front and rear seat passengers bivariate series --#
y2  = seatbelt[1:3,:]

#-- Bivariate basic structural time series model with regression variables --#
# petrol and kilometer travelled, before intervention
bibstsm    = ssm.model_mvstsm(2,[True,True,False],'level','trig fixed',12,x=seatbelt[3:5,:])
bibstsm,res = ssm.estimate(y2[:,:169],bibstsm,np.log([0.1,0.1,0.05,0.02,0.02,0.01])/2, method='Nelder-Mead',options={'disp':not SILENT_OUTPUT})
# opt_x,logL = ssm.estimate(y2[:,:169],bibstsm,np.log([0.00531,0.0083,0.00441,0.000247,0.000229,0.000218])/2)[:2]
logL  = res['logL']
Qirr  = bibstsm['H']['mat']
Qlvl  = bibstsm['Q']['mat']

fout.write('[Parameters estimated w/o intervention on front and rear seat bivariate series]\n')
fout.write("Loglikelihood: %g.\n" % logL)
fout.write('Irregular disturbance   Level disturbance\n')
fline   = "%-10.5g  %-10.5g  %-10.5g  %-10.5g\n"
fout.write(fline % (Qirr[0,0],Qirr[0,1],Qlvl[0,0],Qlvl[0,1]))
fout.write(fline % (Qirr[1,0],Qirr[1,1],Qlvl[1,0],Qlvl[1,1]))
fout.write('\n')

alphahat    = ssm.statesmo(y2[:,:169],bibstsm)[0]
comhat      = ssm.signal(alphahat, bibstsm)
lvlhat      = comhat[:,:,0]
seashat     = comhat[:,:,1]
reghat      = comhat[:,:,2]

fig  = plt.figure(num='Estimated components w/o intervention on front and rear seat bivariate series')
ax1  = plt.subplot(221)
plt.plot(time[:169], lvlhat[0,:]+reghat[0,:])
plt.scatter(time[:169], y2[0,:169], 8, 'r', 's', 'filled')
plt.title('Front seat passenger level (w/o seasonal)'); plt.xlim([time[0],time[168]]); plt.ylim([6,7.25])
ax2  = plt.subplot(222)
plt.plot(time[:169], lvlhat[0,:])
plt.title('Front seat passenger level'); plt.xlim([time[0],time[168]]); # plt.ylim([3.84,4.56])
ax3  = plt.subplot(223)
plt.plot(time[:169], lvlhat[1,:]+reghat[1,:])
plt.scatter(time[:169], y2[1,:169], 8, 'r', 's', 'filled')
plt.title('Rear seat passenger level (w/o seasonal)'); plt.xlim([time[0],time[168]]); plt.ylim([5.375,6.5])
ax4  = plt.subplot(224)
plt.plot(time[:169], lvlhat[1,:])
plt.title('Rear seat passenger level'); plt.xlim([time[0],time[168]]); # plt.ylim([1.64,1.96])

fig_count  = output_fig()

#-- Add intervention to both series --#
bibstsm2i  = ssm.model_cat([bibstsm,ssm.model_mvreg(2,ssm.x_intv(y2.shape[1],'step',169),[[True],[True]])])
bibstsm2i,res  = ssm.estimate(y2, bibstsm2i, np.log([0.1,0.1,0.05,0.02,0.02,0.01])/2, method='Nelder-Mead',options={'disp':not SILENT_OUTPUT})
# opt_x,logL2i  = ssm.estimate(y2, bibstsm2i, np.log([0.0054,0.00857,0.00445,0.000256,0.000232,0.000225])/2)[:2]
logL2i = res['logL']
Qirr  = bibstsm2i['H']['mat']
Qlvl  = bibstsm2i['Q']['mat']
alphahat2i,V2i    = ssm.statesmo(y2, bibstsm2i)[:2]

fout.write('[Parameters estimated w/ intervention on both series]\n')
fout.write("Loglikelihood: %g.\n" % logL2i)
fout.write('Irregular disturbance   Level disturbance\n')
fout.write(fline % (Qirr[0,0],Qirr[0,1],Qlvl[0,0],Qlvl[0,1]))
fout.write(fline % (Qirr[1,0],Qirr[1,1],Qlvl[1,0],Qlvl[1,1]))
fout.write('Level shift intervention:\n')
fout.write('        Coefficient     R. m. s. e.     t-value    p-value\n')
tval = alphahat2i[-2,-1]/np.sqrt(V2i[-2,-2,-1])
pval = stats.t.sf(np.abs(tval),y.shape[1] - bstsmir.m)*2
fout.write("front   %-14.5f  %-14.5f  %.5f   %.5f\n" % (alphahat2i[-2,-1],np.sqrt(V2i[-2,-2,-1]),tval,pval))
tval = alphahat2i[-1,-1]/np.sqrt(V2i[-1,-1,-1])
pval = stats.t.sf(np.abs(tval),y.shape[1] - bstsmir.m)*2
fout.write("rear    %-14.5f  %-14.5f  %.5f   %.5f\n\n" % (alphahat2i[-1,-1],np.sqrt(V2i[-1,-1,-1]),tval,pval))

#-- Add intervention only to front seat passenger series --#
bibstsmi  = ssm.model_cat([bibstsm,ssm.model_mvreg(2,ssm.x_intv(y2.shape[1],'step',169),[[True],[False]])])
bibstsmi,res  = ssm.estimate(y2, bibstsmi, np.log([0.1,0.1,0.05,0.02,0.02,0.01])/2, method='Nelder-Mead',options={'disp':not SILENT_OUTPUT}) #np.log([0.00539,0.00856,0.00445,0.000266,0.000235,0.000232])/2)[:2]
logLi  = res['logL']
Qirr  = bibstsmi['H']['mat']
Qlvl  = bibstsmi['Q']['mat']
alphahati,Vi  = ssm.statesmo(y2, bibstsmi)[:2]

fout.write('[Parameters estimated w/ intervention only on front seat series]\n')
fout.write("Loglikelihood: %g.\n" % logLi)
fout.write('Irregular disturbance   Level disturbance\n')
fout.write(fline % (Qirr[0,0],Qirr[0,1],Qlvl[0,0],Qlvl[0,1]))
fout.write(fline % (Qirr[1,0],Qirr[1,1],Qlvl[1,0],Qlvl[1,1]))
fout.write('Level shift intervention:\n')
fout.write('        Coefficient     R. m. s. e.     t-value    p-value\n')
tval = alphahati[-1,-1]/np.sqrt(Vi[-1,-1,-1])
pval = stats.t.sf(np.abs(tval),y.shape[1] - bstsmir.m)*2
fout.write("front   %-14.5f  %-14.5f  %.5f   %.5f\n\n" % (alphahati[-1,-1],np.sqrt(Vi[-1,-1,-1]),tval,pval))

comhati   = ssm.signal(alphahati, bibstsmi)
lvlhati   = comhati[:,:,0]
seashati  = comhati[:,:,1]
reghati   = comhati[:,:,2]
intvhati  = comhati[:,:,3]

fig  = plt.figure(num='Estimated components w/ intervention only on front seat series')
ax1  = plt.subplot(221)
plt.plot(time[:-1], lvlhati[0,:]+reghati[0,:]+intvhati[0,:],label='est. level')
plt.scatter(time[:-1], y2[0,:].squeeze(), 8, 'r', 's', 'filled', label='front seat')
plt.title('Front seat passenger level (w/o seasonal)')
plt.xlim([time[0],time[-2]]); plt.ylim([6,7.25]); plt.legend()
ax2  = plt.subplot(222)
plt.plot(time[:-1], lvlhati[0,:]+intvhati[0,:])
plt.title('Front seat passenger level')
plt.xlim([time[0],time[-2]]); # plt.ylim([3.84,4.56])
ax3  = plt.subplot(223)
plt.plot(time[:-1], lvlhati[1,:]+reghati[1,:], label='est. level')
plt.scatter(time[:-1], y2[1,:].squeeze(), 8, 'r', 's', 'filled', label='rear seat')
plt.title('Rear seat passenger level (w/o seasonal)')
plt.xlim([time[0],time[-2]]); plt.ylim([5.375,6.5]); plt.legend()
ax4  = plt.subplot(224)
plt.plot(time[:-1], lvlhati[1,:])
plt.title('Rear seat passenger level')
plt.xlim([time[0],time[-2]]); # plt.ylim([1.64,1.96])

fig_count  = output_fig()

if SILENT_OUTPUT: fout.close()
