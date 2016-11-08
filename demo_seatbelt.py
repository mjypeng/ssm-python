# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.linalg import sqrtm
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import ssmodel as ssm

mpl.rcParams['figure.figsize'] = (16,10)
SILENT_OUTPUT = bool(sys.argv[1] if len(sys.argv)>1 else 0)
if SILENT_OUTPUT:
    run_name = '_'+sys.argv[2] if len(sys.argv)>2 else ''
    fout  = open('demo_seatbelt'+run_name+'_out.txt','w')
    mpl.rcParams['savefig.bbox']        = 'tight'
    mpl.rcParams['savefig.dpi']         = 150
    mpl.rcParams['savefig.format']      = 'png'
    mpl.rcParams['savefig.pad_inches']  = 0.1
else:
    fout  = sys.stdout

fout.write('\n')

#-- Load data --#
seatbelt = np.loadtxt('data/seatbelt.dat').T
time     = pd.date_range(pd.to_datetime('19690101'),pd.to_datetime('19850101'),freq='MS')

#-- Analysis of drivers series --#
y       = seatbelt[[0],:]
n       = y.shape[1]
mis     = np.array(np.isnan(y))
anymis  = np.any(mis,0)
allmis  = np.all(mis,0)

#-- Estimation of basic structural time series model --#
bstsm   = ssm.model_stsm('level', 'trig1', 12)
# bstsm   = ssm.model_cat([ssm.model_llm(),ssm.model_seasonal('trig1', 12)])
opt_x   = ssm.estimate(y, bstsm, np.log([0.003,0.0009,5e-7])/2)[0]
# bstsm       = estimate(y, bstsm, [0.003 0.0009 5e-7], [], 'fmin', 'bfgs', 'disp', 'off');
fout.write("epsilon variance = %g, eta variance = %g, omega variance = %g.\n" % (np.exp(2*opt_x[0]),np.exp(2*opt_x[1]),np.exp(2*opt_x[2])))

a,P,d,v,invF  = ssm.kalman_int(1,n,y,mis,anymis,allmis,ssm.set_param(bstsm,opt_x))
a[:,:d+1]     = np.nan
P[:,:,:d+1]   = np.nan

alphahat,V,r,N  = ssm.statesmo_int(1,n,y,mis,anymis,allmis,ssm.set_param(bstsm,opt_x))
#-- Retrieve components --#
ycom        = ssm.signal(a, bstsm, [0,1,12])
lvl         = ycom[0,:].squeeze()
seas        = ycom[1,:].squeeze()
ycomhat     = ssm.signal(alphahat, bstsm, [0,1,12])
lvlhat      = ycomhat[0,:].squeeze()
seashat     = ycomhat[1,:].squeeze()

irr,etahat,epsvarhat,etavarhat = ssm.disturbsmo_int(1,n,y,mis,anymis,allmis,ssm.set_param(bstsm,opt_x))
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

plt.show()

fig = plt.figure(num='Data and level')
plt.plot(time, lvl, label='filtered level')
plt.plot(time[:-1], lvlhat, ':', label='smoothed level')
plt.scatter(time[:-1], y.tolist()[0],c='r',marker='+', label='drivers')
plt.ylim([6.95,7.9]); plt.legend()

plt.show()

#-- Calculate standardized residuals --#
u       = irr/np.sqrt(epsvarhat)
r       = np.zeros((12,y.shape[1]))
for t in range(y.shape[1]): r[:,[t]] = np.asmatrix(sqrtm(etavarhat[:,:,t])).I*etahat[:,[t]]
comres  = ssm.signal(r, bstsm, [0,1,12])
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

plt.show()

#-- Adding explanatory variables and intervention to the model --#
petrol   = seatbelt[[4],:]
bstsmir  = ssm.model_cat([bstsm,ssm.model_intv(y.shape[1],'step',169),ssm.model_reg(petrol)])
opt_x,logL  = ssm.estimate(y, bstsmir, np.log([0.004,0.00027,1e-6])/2)[:2]

alphahatir,Vir  = ssm.statesmo_int(1,n,y,mis,anymis,allmis,ssm.set_param(bstsmir,opt_x))[:2]
irrir           = ssm.disturbsmo_int(1,n,y,mis,anymis,allmis,ssm.set_param(bstsmir,opt_x))[0]
irrir   = irrir.squeeze()
ycomir  = ssm.signal(alphahatir, bstsmir, [0,1,12,13,14])
lvlir   = np.sum(ycomir[[0,2,3],:],0).squeeze()
seasir  = ycomir[1,:].squeeze()

fout.write('[Analysis on drivers series]\n')
fout.write("Loglikelihood: %g\n" % logL)
fout.write("Irregular variance: %g\n" % np.exp(opt_x[0]))
fout.write("Level variance: %g\n" % np.exp(opt_x[1]))
fout.write("Seasonal variance: %g\n" % np.exp(opt_x[2]))
fout.write('Variable             Coefficient     R. m. s. e.     t-value\n')
fout.write("petrol coefficient   %-14.5g  %-14.5g  %g\n" % (alphahatir[-1,0],np.sqrt(Vir[-1,-1,-1]),alphahatir[-1,0]/np.sqrt(Vir[-1,-1,-1])))
fout.write("level shift at 83.2  %-14.5g  %-14.5g  %g\n\n" % (alphahatir[-2,0],np.sqrt(Vir[-2,-2,-1]),alphahatir[-2,0]/np.sqrt(Vir[-2,-2,-1])))

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
plt.show()

irr,etahat,epsvarhat,etavarhat = ssm.disturbsmo_int(1,n,y,mis,anymis,allmis,ssm.set_param(bstsmir,opt_x))
r       = np.zeros((12,y.shape[1]))
for t in range(y.shape[1]): r[:,[t]] = np.asmatrix(sqrtm(etavarhat[:,:,t])).I*etahat[:,[t]]
comres  = ssm.signal(r, bstsm, [0,1,12])
lvlres  = comres[0,:].squeeze()

fig = plt.figure(num='Estimated Components w/ intervention and regression')
ax1 = plt.subplot(211)
plt.plot(time[:-1], irrir)
plt.title('Irregular'); plt.xlim([time[0],time[-1]]); plt.ylim([-0.15,0.15])
ax2 = plt.subplot(212)
plt.plot(time[:-1], lvlres)
plt.title('Normalized level residuals'); plt.xlim([time[0],time[-1]]); plt.ylim([-1.5,1])

plt.show()

#-- Analysis of both front and rear seat passengers bivariate series --#
y2  = seatbelt[1:3,:]
n       = y2.shape[1]
mis     = np.array(np.isnan(y2))
anymis  = np.any(mis,0)
allmis  = np.all(mis,0)

#-- Bivariate basic structural time series model with regression variables --#
# petrol and kilometer travelled, before intervention
bibstsm    = ssm.model_mvstsm(2,[True,True,False],'level','trig fixed',12,x=seatbelt[3:5,:])
opt_x,logL = ssm.estimate(y2[:,:169],bibstsm,np.log([0.00531,0.0083,0.00441,0.000247,0.000229,0.000218])/2)[:2]
Qirr  = ssm.f_psi_to_cov(2)(opt_x[:3])
Qlvl  = ssm.f_psi_to_cov(2)(opt_x[3:6])

fout.write('[Parameters estimated w/o intervention on front and rear seat bivariate series]\n')
fout.write("Loglikelihood: %g.\n" % logL)
fout.write('Irregular disturbance   Level disturbance\n')
fline   = "%-10.5g  %-10.5g  %-10.5g  %-10.5g\n"
fout.write(fline % (Qirr[0,0],Qirr[0,1],Qlvl[0,0],Qlvl[0,1]))
fout.write(fline % (Qirr[1,0],Qirr[1,1],Qlvl[1,0],Qlvl[1,1]))
fout.write('\n')

alphahat    = ssm.statesmo_int(1,n,y2[:,:169],mis,anymis,allmis,ssm.set_param(bibstsm,opt_x))[0]
comhat      = signal(alphahat, bibstsm, [0,1,12,14]);
lvlhat      = comhat(:, :, 1);
seashat     = comhat(:, :, 2);
reghat      = comhat(:, :, 3);
# figure('Name', 'Estimated components w/o intervention on front and rear seat bivariate series');
# subplot(2, 2, 1), plot(time(1:169), lvlhat(1, :)+reghat(1, :)), hold all, scatter(time(1:169), y2(1, 1:169), 8, 'r', 's', 'filled'), hold off, title('Front seat passenger level (w/o seasonal)'), xlim([68 85]), ylim([6 7.25]);
# subplot(2, 2, 2), plot(time(1:169), lvlhat(1, :)), title('Front seat passenger level'), xlim([68 85]),% ylim([3.84 4.56]);
# subplot(2, 2, 3), plot(time(1:169), lvlhat(2, :)+reghat(2, :)), hold all, scatter(time(1:169), y2(2, 1:169), 8, 'r', 's', 'filled'), hold off, title('Rear seat passenger level (w/o seasonal)'), xlim([68 85]), ylim([5.375 6.5]);
# subplot(2, 2, 4), plot(time(1:169), lvlhat(2, :)), title('Rear seat passenger level'), xlim([68 85]),% ylim([1.64 1.96]);
# if ispc, set(gcf, 'WindowStyle', 'docked'); end

# % Add intervention to both series
# bibstsm2i           = [bibstsm ssm_mvintv(2, size(y2, 2), 'step', 170)];
# [bibstsm2i logL2i]  = estimate(y2, bibstsm2i, [0.0054 0.00857 0.00445 0.000256 0.000232 0.000225], [], 'fmin', 'bfgs', 'disp', 'off');
# [alphahat2i V2i]    = statesmo(y2, bibstsm2i);
# fprintf(1, '[Parameters estimated w/ intervention on both series]\n');
# fprintf(1, 'Loglikelihood: %g.\n', logL2i);
# fprintf(1, 'Irregular disturbance   Level disturbance\n');
# fprintf(1, fline, bibstsm2i.param([1 3 4 6]));
# fprintf(1, fline, bibstsm2i.param([3 2 6 5]));
# fprintf(1, 'Level shift intervention:\n');
# fprintf(1, '        Coefficient     R. m. s. e.     t-value\n');
# fprintf(1, 'front   %-14.5g  %-14.5g  %g\n', alphahat2i(end-1, end), realsqrt(V2i(end-1, end-1, end)), alphahat2i(end-1, end)/realsqrt(V2i(end-1, end-1, end)));
# fprintf(1, 'rear    %-14.5g  %-14.5g  %g\n\n', alphahat2i(end, end), realsqrt(V2i(end, end, end)), alphahat2i(end, end)/realsqrt(V2i(end, end, end)));

# % Add intervention only to front seat passenger series
# bibstsmi            = [bibstsm ssm_mvintv(2, size(y2, 2), {'step' 'null'}, 170)];
# [bibstsmi logLi]    = estimate(y2, bibstsmi, [0.00539 0.00856 0.00445 0.000266 0.000235 0.000232]);
# [alphahati Vi]      = statesmo(y2, bibstsmi);
# fprintf(1, '[Parameters estimated w/ intervention only on front seat series]\n');
# fprintf(1, 'Loglikelihood: %g.\n', logLi);
# fprintf(1, 'Irregular disturbance   Level disturbance\n');
# fprintf(1, fline, bibstsmi.param([1 3 4 6]));
# fprintf(1, fline, bibstsmi.param([3 2 6 5]));
# fprintf(1, 'Level shift intervention:\n');
# fprintf(1, '        Coefficient     R. m. s. e.     t-value\n');
# fprintf(1, 'front   %-14.5g  %-14.5g  %g\n\n', alphahati(end, end), realsqrt(Vi(end, end, end)), alphahati(end, end)/realsqrt(Vi(end, end, end)));

# comhati     = signal(alphahati, bibstsmi);
# lvlhati     = comhati(:, :, 1);
# seashati    = comhati(:, :, 2);
# reghati     = comhati(:, :, 3);
# intvhati    = comhati(:, :, 4);
# figure('Name', 'Estimated components w/ intervention only on front seat series');
# subplot(2, 2, 1), plot(time(1:end-1), lvlhati(1, :)+reghati(1, :)+intvhati(1, :), 'DisplayName', 'est. level'), hold all, scatter(time(1:end-1), y2(1, :), 8, 'r', 's', 'filled', 'DisplayName', 'front seat'), hold off, title('Front seat passenger level (w/o seasonal)'), xlim([68 85]), ylim([6 7.25]), legend('show');
# subplot(2, 2, 2), plot(time(1:end-1), lvlhati(1, :)+intvhati(1, :)), title('Front seat passenger level'), xlim([68 85]),% ylim([3.84 4.56]);
# subplot(2, 2, 3), plot(time(1:end-1), lvlhati(2, :)+reghati(2, :), 'DisplayName', 'est. level'), hold all, scatter(time(1:end-1), y2(2, :), 8, 'r', 's', 'filled', 'DisplayName', 'rear seat'), hold off, title('Rear seat passenger level (w/o seasonal)'), xlim([68 85]), ylim([5.375 6.5]), legend('show');
# subplot(2, 2, 4), plot(time(1:end-1), lvlhati(2, :)), title('Rear seat passenger level'), xlim([68 85]),% ylim([1.64 1.96]);
# if ispc, set(gcf, 'WindowStyle', 'docked'); end

# % figure('Name', 'Estimated components w/ intervention only on front seat series');
# % subplot(2, 1, 1), plot(time(1:end-1), lvlhati(1, :)+reghati(1, :)+intvhati(1, :), 'DisplayName', 'est. level'), hold all, scatter(time(1:end-1), y2(1, :), 8, 'r', 's', 'filled', 'DisplayName', 'front seat'), hold off, title('Front seat passenger level (w/o seasonal)'), xlim([68 85]), ylim([6 7.25]);
# % subplot(2, 1, 2), plot(time(1:end-1), lvlhati(2, :)+reghati(2, :), 'DisplayName', 'est. level'), hold all, scatter(time(1:end-1), y2(2, :), 8, 'r', 's', 'filled', 'DisplayName', 'rear seat'), hold off, title('Rear seat passenger level (w/o seasonal)'), xlim([68 85]), ylim([5.375 6.5]);
# % if ispc, set(gcf, 'WindowStyle', 'docked'); end

if SILENT_OUTPUT: fout.close()
