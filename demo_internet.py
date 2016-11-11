# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import ssm

fout  = sys.stdout

fout.write('\n')

internet  = np.loadtxt('data/internet.dat').T
y         = internet[[1],1:]

P  = 5
Q  = 5

#-- Model selection for complete series --#
fout.write('AIC for ARMA(p, q) models on complete data:\n')
fout.write('    ')
for q in range(Q+1): fout.write('%-12d' % q)
fout.write('\n')

logL    = np.zeros((P+1, Q+1))
AIC     = np.zeros((P+1, Q+1))
arma    = np.tile(None,(P+1, Q+1))
for p in range(P+1):
    fout.write('%-4d' % p)
    for q in range(Q+1):
        model      = ssm.model_arma(p,q)
        arma[p,q],res = ssm.estimate(y,model,[0.1]*(model['nparam']-1)+[np.log(10)/2],method='Nelder-Mead',options={'maxiter':4000,'maxfev':3000})
        logL[p,q]  = res.logL
        AIC[p,q]   = res.AIC
        fout.write("%-12g" % AIC[p,q])
    fout.write('\n')

i  = AIC.argmin(); temp = AIC.flat[i]; AIC.flat[i] = np.inf
fout.write("ARMA(%d, %d) found to be the best model, " % np.unravel_index(i,AIC.shape))
j  = AIC.argmin(); AIC.flat[i] = temp
fout.write("ARMA(%d, %d) is the second best.\n\n" % np.unravel_index(j,AIC.shape))

#-- Model selection for data with missing values --#
ymis  = y.copy()
ymis[:,[6,16,26,36,46,56,66,72,73,74,75,76,86,96]] = np.nan
fout.write('AIC for ARMA(p, q) models on data w/ missing observations:\n')
fout.write('    ')
for q in range(Q+1): fout.write("%-12d" % q)
fout.write('\n')

logLmis     = np.zeros((P+1,Q+1))
AICmis      = np.zeros((P+1,Q+1))
armamis     = np.tile(None,(P+1,Q+1))
for p in range(P+1):
    fout.write('%-4d' % p)
    for q in range(Q+1):
        model  = ssm.model_arma(p,q)
        armamis[p,q],res  = ssm.estimate(ymis,model,[-0.1]*(model['nparam']-1) + [np.log(1.0)/2],method='Nelder-Mead',options={'maxiter':4000,'maxfev':3000})
        logLmis[p,q]      = res.logL
        AICmis[p,q]       = res.AIC
        fout.write("%-12g" % AICmis[p,q])
    fout.write('\n')

i  = AICmis.argmin(); temp = AICmis.flat[i]; AICmis.flat[i] = np.inf
fout.write("ARMA(%d, %d) found to be the best model, " % np.unravel_index(i,AICmis.shape))
j  = AICmis.argmin(); AICmis.flat[i] = temp
fout.write("ARMA(%d, %d) is the second best.\n\n" % np.unravel_index(j,AICmis.shape))

#-- Forecast with ARMA(1, 1) on the complete data --#
armafore,res  = ssm.estimate(y,ssm.model_arma(1,1),[0.1,0.1,np.log(0.1)/2],method='Nelder-Mead',options={'maxiter':4000,'maxfev':3000})
yf            = ssm.signal(ssm.kalman(np.bmat([y,np.tile(np.nan,(1,20))]), armafore)[0], armafore)
fig  = plt.figure(num='Internet series forecast')
plt.plot(yf.squeeze(), label='forecast')
plt.scatter(range(1,y.shape[1]+1), y.squeeze(), 10, 'r', 's', 'filled', label='data')
plt.title('Internet series forecast'); plt.ylim([-15,15]); plt.legend()

plt.show()

#-- Forecast with ARMA(1, 1) on the data w/ missing values --#
armafore,res  = ssm.estimate(ymis, ssm.model_arma(1,1),[0.1,0.1,np.log(0.1)/2],method='Nelder-Mead',options={'maxiter':4000,'maxfev':3000})
ymisf         = ssm.signal(ssm.kalman(np.bmat([ymis,np.tile(np.nan,(1,20))]), armafore)[0], armafore)
fig  = plt.figure(num='Internet series in-sample one-step and out-of-sample forecasts')
plt.plot(ymisf.squeeze(), label='forecast')
plt.scatter(range(1,ymis.shape[1]+1), ymis.squeeze(), 10, 'r', 's', 'filled', label='data')
plt.title('Internet series in-sample one-step and out-of-sample forecasts')
plt.ylim([-15,15])

plt.show()
