import numpy as np
import matplotlib.pyplot as plt



y_exct = np.load('y_true.npy')                                          

print(y_exct.shape)
y = np.load('y_obs.npy')                                                

y_e = np.load('assimilated_ensemble.npy')
y_obs_alltime = np.load('simualated_all_time_obs.npy') 

n_ensemble = np.shape(y_e)[0]  


                     
y_alltime = np.transpose(y_obs_alltime, (1,0,2))
y_e_tr_obs = np.transpose(y_e, (1,0,2))# use while plotiing against N_obs
y_e_trans_spatial = np.transpose(y_e, (1,2,0)) # use while plotiing against X






xi =39
y_e_mean_obs = np.mean(y_e_tr_obs[:,:,xi], axis=1)

plt.plot(y_alltime[:,:,xi], 'g-')
#plt.plot(y_exct[:,xi], 'r-')
plt.show()
        



