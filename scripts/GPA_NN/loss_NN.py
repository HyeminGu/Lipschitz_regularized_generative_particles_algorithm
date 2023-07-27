#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid tensorflow warning
import tensorflow as tf
import numpy as np

def f_Lip_divergence(P, Q, params={'f':'KL', 'L':1.0, 'alpha':None}):
    # P, Q: numpy arrays
    
    # Prepare data & data_par ------------------------------------
    if P.ndim == 3: # grayscale image
        P = np.expand_dims(P, axis=3)
    if Q.ndim == 3: # grayscale image
        Q = np.expand_dims(Q, axis=3)
    
    data_par = {'P_label': None, 'Q_label': None,}
    data_par['N_samples_P'], data_par['N_samples_Q'] = P.shape[0], Q.shape[0]
    data_par['mb_size_P'], data_par['mb_size_Q'] = 200, 200
    
    N_dim = P.shape[1:]
    if N_dim == (): # dim=1
        N_dim = 1
    elif len(N_dim) == 1:
        N_dim = N_dim[0]
    
    Q = tf.constant(Q, dtype=tf.float32) # constant
    P = tf.constant(P, dtype=tf.float32) # constant
    
    # Prepare loss_par -------------------------------------------
    calc_Wasserstein1 = False
    loss_par = {'reverse': False, 'lamda': 0.0}
    if params['f'] == 'KL':
        loss_par.update({'f': params['f'], 'formulation': 'DV', 'par': []})
    elif params['f'] == 'W1':
        calc_Wasserstein1 = True
    elif params['alpha'] != None:
        loss_par.update({'f': params['f'], 'formulation': 'LT', 'par': [params['alpha']]})
    else: # alpha=2 divergence
        loss_par.update({'f': params['f'], 'formulation': 'LT', 'par': [2.0]})
        
    # Prepare Neural network structure and NN_par -----------------
    from lib.construct_NN import check_nn_topology, initialize_NN, model
    NN_par = {'N_dim':N_dim, 'N_conditions': 1, 'constraint': 'hard', 'L': params['L'], 'eps': None}
    
    if type(N_dim) == type([]): # image data
        NN_par.update({'NN_model': 'cnn-fnn', 'activation_ftn': ['leaky_relu', 'relu'],})
        N_fnn_layers = [int(np.prod(N_dim))]
        N_cnn_layers = [128, 128, 128]
    else:
        NN_par.update({'NN_model': 'fnn', 'activation_ftn': ['relu'],})
        if N_dim <= 10:
            N_fnn_layers = [32, 32, 32]
        elif N_dim <= 100:
            N_fnn_layers = [64, 64, 64]
        else:
            N_fnn_layers = [128, 128, 128]
        N_cnn_layers = None
    
    N_fnn_layers, N_cnn_layers, NN_par['activation_ftn'] = check_nn_topology(NN_par['NN_model'], N_fnn_layers, N_cnn_layers, N_dim, NN_par['activation_ftn'])
    NN_par.update({'N_cnn_layers':N_cnn_layers, 'N_fnn_layers':N_fnn_layers,})
    
    W, b = initialize_NN(NN_par)
    # scalar optimal value optimization for f-divergence
    nu = tf.Variable(0.0, dtype=tf.float32)
    parameters = {'W':W, 'b':b, 'nu':nu} # Learnable parameters for the discriminator phi
    phi = model(NN_par)  # discriminator

    
    # Prepare Neural network training hyper-parameters ------------
    from lib.train_NN import train_disc, train_wasserstein1
    lr_phi, epochs_phi, optimizer = 0.0005, 100, 'adam'
    lr_phi = tf.Variable(lr_phi, trainable=False) # lr for training a discriminator function
    
    # Train loss --------------------------------------------------
    if calc_Wasserstein1 == True:
        parameters, current_loss, _ = train_wasserstein1(parameters, phi, P, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer, print_vals=True)
    else:
        parameters, current_loss, _ = train_disc(parameters, phi, P, Q, lr_phi, epochs_phi, loss_par, NN_par, data_par, optimizer, print_vals=True)
        
    return current_loss
    

# ------------------------------------------------------------------------------
# test code
if __name__ == "__main__":
    import sys
    # python3 loss_NN.py f_Lip_divergence MNIST
    if sys.argv[1] == "MNIST":
        from tensorflow.keras.datasets.mnist import load_data
        
        (x_train, y_train), (x_test, y_test) = load_data(path="mnist.npz")
        x_train = x_train/255.0
        x_train = np.expand_dims(x_train, axis=3)
        x_train = np.float32(x_train)
        print("KL-Lip=1 divergence: ", f_Lip_divergence(x_train[:100], x_train[100:200]))
        print("KL-Lip=10 divergence: ", f_Lip_divergence(x_train[:100], x_train[100:200],
        params={'f':'KL', 'L':10.0, 'alpha':None}))
        print("KL divergence: ", f_Lip_divergence(x_train[:100], x_train[100:200],
        params={'f':'KL', 'L':None, 'alpha':None}))
        
        print("alpha=10.0-Lip=1 divergence: ", f_Lip_divergence(x_train[:100], x_train[100:200],
        params={'f':'alpha', 'L':1.0, 'alpha':10.0}))
        print("alpha=10.0-Lip=1 divergence: ", f_Lip_divergence(x_train[:100], x_train[100:200],
        params={'f':'alpha', 'L':10.0, 'alpha':10.0}))
        print("alpha=10.0 divergence: ", f_Lip_divergence(x_train[:100], x_train[100:200],
        params={'f':'alpha', 'L':None, 'alpha':10.0}))
    if sys.argv[1] == "Gaussian":
        P = np.random.normal(loc=0.0, scale=1.0, size=(500,2)) # 500 2D samples from P
        Q = np.random.normal(loc=10.0, scale=2.0, size=(300,2)) # 300 2D samples from Q

        print("KL-Lip=1 divergence: ", f_Lip_divergence(P,Q))
        print("KL-Lip=10 divergence: ", f_Lip_divergence(P,Q,
        params={'f':'KL', 'L':10.0, 'alpha':None}))
        print("KL divergence: ", f_Lip_divergence(P,Q,
        params={'f':'KL', 'L':None, 'alpha':None}))
        
        print("alpha=10.0-Lip=1 divergence: ", f_Lip_divergence(P,Q,
        params={'f':'alpha', 'L':1.0, 'alpha':10.0}))
        print("alpha=10.0-Lip=1 divergence: ", f_Lip_divergence(P,Q,
        params={'f':'alpha', 'L':10.0, 'alpha':10.0}))
        print("alpha=10.0 divergence: ", f_Lip_divergence(P,Q,
        params={'f':'alpha', 'L':None, 'alpha':10.0}))
        
