import numpy as np
import matplotlib.pyplot as plt

f= np.linspace(-3,8.5,100)
v = np.linspace(0,1.2,100)
F,V=np.meshgrid(f,v)
 
T_global = 4.3/8.5 * F  # ~4.3 degree warming for rcp8.5   (just for comparing to marzeion)
 
gamma = 1
 
S0 = 1
M0 = 1
#Vdot = (V)**.8 * S0 - (V)**1.5 * M0*(1+0.4*F)  # strong glacier sensitivity to cooling
Vdot = (V)**.8 * (S0 - M0*(1+0.4*F + np.log(V)*2))
#Vdot = (V)**.8 * (S0 - M0*np.exp(0.3*F + (V-1)*3))
 
plt.pcolormesh(T_global,V,Vdot,cmap='bwr',vmin=-S0,vmax=S0)
plt.colorbar()
plt.xlabel('global mean temperature')
plt.ylabel('V/V0')
 
#https://www.nature.com/articles/s41558-018-0093-1