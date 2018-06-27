
import numpy as np
import matplotlib.pyplot as plt


for iteration in range(10):
    
#    runfile('/home/jinsung/Documents/Jinsung/2018_Research/KDD/Synthetic_Main.py', wdir='/home/jinsung/Documents/Jinsung/2018_Research/KDD')
    runfile('/home/jinsung/Documents/Jinsung/2018_Research/KDD/Synthetic_Main_Final.py', wdir='/home/jinsung/Documents/Jinsung/2018_Research/KDD')
    
    # Data Loading
    
    File_name = '/home/jinsung/Documents/Jinsung/2018_Research/KDD/Result/maggic_p_0.csv'
    Original = np.loadtxt(File_name)
    
    Orig_AUC = Original[:,0]
    Orig_APR = Original[:,2]
    
    L = 12
    
    
    #%%
    def kappa (Order_A, Order_B):
        L = len(Order_A)
        
        nom = 0    
        
        for i in range(L):
            for j in range(L):
                Temp_A = Order_A[i] - Order_A[j]
                Temp_B = Order_B[i] - Order_B[j]
                
                Temp = Temp_A * Temp_B
                
                if (Temp >= 0):
                    nom = nom + 1
                    
        kappa_stat = round(float(nom - L)  / (L*(L-1)),3)
        
        return kappa_stat
        
    #%% 
    Result_Kappa = np.zeros([10,4])
        
    for i in range(10):
        File_name = '/home/jinsung/Documents/Jinsung/2018_Research/KDD/Result/maggic_p_'+str( (i+1) * 5)+'_New.csv'
        New = np.loadtxt(File_name)
        
        New_AUC = New[:,0]
        New_APR = New[:,2]
        
        Result_Kappa[i,0] = kappa(Orig_AUC, New_AUC)
        Result_Kappa[i,1] = kappa(Orig_APR, New_APR)
        
    #%%
    for i in range(10):
        File_name = '/home/jinsung/Documents/Jinsung/2018_Research/KDD/Result/maggic_p_'+str( (i+1) * 5)+'.csv'
        New = np.loadtxt(File_name)
        
        New_AUC = New[:,0]
        New_APR = New[:,2]
        
        Result_Kappa[i,2] = np.mean(New_AUC)
        Result_Kappa[i,3] = np.mean(New_APR)    
        
    
    file_name = '/home/jinsung/Documents/Jinsung/2018_Research/KDD/Final/'+ str(iteration) + '.csv'    
    np.savetxt(file_name, Result_Kappa)


