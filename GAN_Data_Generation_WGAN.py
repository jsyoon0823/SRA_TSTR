'''
Jinsung Yoon (05/13/2018)
NIPS DP GAN First Model with PATE
'''

#%% Packages
import tensorflow as tf
import numpy as np

from tqdm import tqdm

#%% Function Start

def GAN_Data_Generation(X_train, Y_train, X_test, Y_test, bin_idx, epsilon, T_No):

    #%% Parameters
    # Batch size    
    mb_size = 64
    # Random variable dimension
    z_dim = 10
    # Hidden unit dimensions
    h_dim = 20
    # Conditioning dimension
    C_dim = 1
        
    
    lam = 10
    lr = 1e-4
    
    lamda = float(1)/float(T_No) * np.sqrt( float(4*mb_size*1000) / epsilon )    
    
    lamda = float(lamda)
    

    #%% Data Preprocessing
    Y_train = np.asarray(Y_train)
    Y_train = np.reshape(Y_train, (len(Y_train),1))
    X_train = np.asarray(X_train)
        
    Y_test = np.asarray(Y_test)
    Y_test = np.reshape(Y_test,(len(Y_test),1))

    # Parameters
    X_dim = len(X_train[0,:])
    Train_No = len(X_train[:,0])
    Test_No = len(Y_test)
    
    #%% Data Normalization
    Min_Val = np.min(X_train,0)
    
    X_train = X_train - Min_Val
    
    Max_Val = np.max(X_train,0)
    
    X_train = X_train / (Max_Val + 1e-8)
    
    #%% Algorithm Start

    #%% Necessary Functions

    # Xavier Initialization Definition
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape = size, stddev = xavier_stddev)    
        
    # Sample from uniform distribution
    def sample_Z(m, n):
        return np.random.uniform(-1., 1., size = [m, n])
        
    # Sample from the real data
    def sample_X(m, n):
        return np.random.permutation(m)[:n]  
     
    #%% Placeholder
    
    # Feature
    X = tf.placeholder(tf.float32, shape = [None, X_dim])   
    # Random Variable    
    Z = tf.placeholder(tf.float32, shape = [None, z_dim])
    # Conditional Variable
    C = tf.placeholder(tf.float32, shape = [None, C_dim])
    # Conditional Variable
    M = tf.placeholder(tf.float32, shape = [None, C_dim])
      
    #%% Discriminator
    # Discriminator
    D_W1 = tf.Variable(xavier_init([X_dim + C_dim, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim,1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_W2, D_b1, D_b2]
    
    #%% Generator

    G_W1 = tf.Variable(xavier_init([z_dim + C_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim,X_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))
    
    theta_G = [G_W1, G_W2, G_b1, G_b2]

    #%% Functions
    def generator(z, c):
        inputs = tf.concat(axis=1, values = [z,c])
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_log_prob = tf.nn.sigmoid(tf.matmul(G_h1, G_W2) + G_b2)
        
        return G_log_prob
    
    def discriminator(x,c):
        inputs = tf.concat(axis=1, values = [x,c])
        D_h1 = tf.nn.relu(tf.matmul(inputs,D_W1) + D_b1)
        out = (tf.matmul(D_h1, D_W2) + D_b2)
        
        return out
    
    #%% 
    # Structure
    G_sample = generator(Z,C)
    D_real = discriminator(X,C)
    D_fake = discriminator(G_sample,C)

    #%%
    D_entire = tf.concat(axis = 0, values = [D_real, D_fake])    
    
    #%%

    # Replacement of Clipping algorithm to Penalty term
    # 1. Line 6 in Algorithm 1
    eps = tf.random_uniform([mb_size, 1], minval = 0., maxval = 1.)
    X_inter = eps*X + (1. - eps) * G_sample

    # 2. Line 7 in Algorithm 1
    grad = tf.gradients(discriminator(X_inter,C), [X_inter,C])[0]
    grad_norm = tf.sqrt(tf.reduce_sum((grad)**2 + 1e-8, axis = 1))
    grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

    # Loss function
    D_loss = tf.reduce_mean((1-M) * D_entire) - tf.reduce_mean(M * D_entire) + grad_pen
    G_loss = -tf.reduce_mean(D_fake)

    # Solver
    D_solver = (tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5).minimize(D_loss, var_list = theta_D))
    G_solver = (tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5).minimize(G_loss, var_list = theta_G))
            
    #%%
    # Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


        
    #%%
    # Iterations
    for it in tqdm(range(50000)):

        for _ in range(5):
            #%% Teacher Training            
            Z_mb = sample_Z(mb_size, z_dim)            
            
            # Teacher 1
            X_idx = sample_X(Train_No,mb_size)        
            X_mb = X_train[X_idx,:]  
            
            C_mb = Y_train[X_idx,:]  
            
            #%%
            
            M_real = np.ones([mb_size,])
            M_fake = np.zeros([mb_size,])

            M_entire = np.concatenate((M_real, M_fake),0)
            
            laplace = np.random.laplace(loc=0.0, scale=lamda, size = mb_size*2)
    
            M_entire = M_entire + laplace
            
            M_entire = (M_entire > 0.5)            
            
            M_mb = np.reshape(M_entire.astype(float), (2*mb_size,1))
            
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X: X_mb, Z: Z_mb, C: C_mb, M: M_mb})
            
                    
        #%% Generator Training
                    
        Z_mb = sample_Z(mb_size, z_dim)   
        
        C_idx = sample_X(Train_No,mb_size)        
            
        C_mb = Y_train[C_idx,:]  
                    
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {Z: Z_mb, C: C_mb})

    #%%       

    #%% Output Generation (Train Set)
    
    X_train_New = sess.run([G_sample], feed_dict = {Z: sample_Z(Train_No, z_dim), C: Y_train})

    X_train_New = X_train_New[0]
    
    #### Renormalization
        
    X_train_New = X_train_New * (Max_Val + 1e-8)
    
    X_train_New = X_train_New + Min_Val

    #### Rounding   
         
    X_train_New[:,bin_idx:] = np.round(X_train_New[:,bin_idx:])
    
    #### Y Return
    
    Y_train_New = np.reshape(Y_train, (len(Y_train[:,0]),))    
    
    #%% Output Generation (Test Set)
    
    X_test_New = sess.run([G_sample], feed_dict = {Z: sample_Z(Test_No, z_dim), C: Y_test})

    X_test_New = X_test_New[0]
    
    #### Renormalization
        
    X_test_New = X_test_New * (Max_Val + 1e-8)
    
    X_test_New = X_test_New + Min_Val

    #### Rounding   
         
    X_test_New[:,bin_idx:] = np.round(X_test_New[:,bin_idx:])
    
    #### Y Return
    
    Y_test_New = np.reshape(Y_test, (len(Y_test[:,0]),))    
    
    #%% Return
    
    return X_train_New, Y_train_New, X_test_New, Y_test_New