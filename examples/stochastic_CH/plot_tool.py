import numpy as np
import matplotlib.pyplot as plt



y_exct = np.load('y_true.npy')                                          

#print(y_exct.shape)
y = np.load('y_obs.npy')                                                

y_e = np.load('assimilated_ensemble.npy')
y_obs_alltime = np.load('simualated_all_time_obs.npy') 

n_ensemble = np.shape(y_e)[0]  

y_e_tr_obs = np.transpose(y_e, (1,0,2))# use while plotiing against N_obs
                     
y_alltime = np.transpose(y_obs_alltime, (1,0,2))

y_e_trans_spatial = np.transpose(y_obs_alltime, (2,1,0)) # use while plotiing against X


y_exct_alltime = np.load('y_true_alltime.npy') 
y_exct_alltime_trans_spatial = np.transpose(y_exct_alltime)

y_exct_obs_alltime = np.load('y_obs_alltime.npy') 

print(y_e_trans_spatial.shape)

xi =7
ensemble = 40
N = 10
y_e_mean_obs = np.mean(y_e_tr_obs[:,:,xi], axis=1)

# plt.plot(y_alltime[:,:,xi], 'g-')
# plt.plot(y_exct_alltime[:,xi], 'r-')
plt.plot(y_e_trans_spatial[:,N, :], '-y')
plt.plot(y_exct_alltime_trans_spatial[:,N], 'b-')
# plt.plot(y_exct_obs_alltime[:,xi], '-o')
plt.show()
        



