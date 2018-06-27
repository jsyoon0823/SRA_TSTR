#%%

import numpy as np
import matplotlib.pyplot as plt

#%%   

Output = np.zeros([10,4])

NN = 11

for i in range(NN):
    
    File_name = '/home/jinsung/Documents/Jinsung/2018_Research/KDD/Final/'+ str(i) +'.csv'
    Temp = np.loadtxt(File_name)
   
    Output = Temp + Output
   
Output = Output / float(NN) 
   
Result_Kappa = Output[:,:2]
Result = Output[:,2:4]
   

#%%   
fig, ax = plt.subplots( nrows=1, ncols=1 )

X = np.asarray(range(10))*0.05 + 0.05
    
plt.plot(X, Result[:,0], c='r', label='TSTR')
plt.plot(X, Result_Kappa[:,0], c='b', label = 'SRA')
        
plt.xlabel('Flipping Probability (p)')
plt.ylabel('TSTR & SRA Performance')
plt.xlim([0,0.5])
plt.ylim([0,1])
plt.legend(loc='lower left')
    
plt.grid()
    
#%%
plt.savefig('/home/jinsung/Documents/Jinsung/2018_Research/KDD/Figure/AUROC.pdf')
plt.close(fig)    
    
    
#%%
    #%%
fig, ax = plt.subplots( nrows=1, ncols=1 )

X = np.asarray(range(10))*0.05 + 0.05
    
plt.plot(X, Result[:,1], c='r', label='TSTR')
plt.plot(X, Result_Kappa[:,1], c='b', label='SRA')
        
plt.xlabel('Flipping Probability (p)')
plt.ylabel('TSTR & SRA Performance')
plt.xlim([0,0.5])
plt.ylim([0,1])
plt.legend(loc='lower left')
    
plt.grid()
    
#%%
plt.savefig('/home/jinsung/Documents/Jinsung/2018_Research/KDD/Figure/AUPRC.pdf')
plt.close(fig)    
    
#%%   
fig, ax = plt.subplots( nrows=1, ncols=1 )

X = np.asarray(range(10))*0.05 + 0.05
    
plt.plot(X, Result[:,0], c='r', label='TSTR')
plt.plot(X, Result_Kappa[:,0], c='b', label = 'SRA')
        
plt.xlabel('Flipping Probability (p)')
plt.ylabel('TSTR & SRA Performance')
plt.xlim([0,0.3])
plt.ylim([0,1])
plt.legend(loc='lower left')
    
plt.grid()
    
#%%
plt.savefig('/home/jinsung/Documents/Jinsung/2018_Research/KDD/Figure/AUROC_0.3.pdf')
plt.close(fig)    
    
    
#%%
    #%%
fig, ax = plt.subplots( nrows=1, ncols=1 )

X = np.asarray(range(10))*0.05 + 0.05
    
plt.plot(X, Result[:,1], c='r', label='TSTR')
plt.plot(X, Result_Kappa[:,1], c='b', label='SRA')
        
plt.xlabel('Flipping Probability (p)')
plt.ylabel('TSTR & SRA Performance')
plt.xlim([0,0.3])
plt.ylim([0,1])
plt.legend(loc='lower left')
    
plt.grid()
    
#%%
plt.savefig('/home/jinsung/Documents/Jinsung/2018_Research/KDD/Figure/AUPRC_0.3.pdf')
plt.close(fig)    
    