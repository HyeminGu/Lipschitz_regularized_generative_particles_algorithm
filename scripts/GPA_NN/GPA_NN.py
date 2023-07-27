#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid tensorflow warning
import tensorflow as tf
import numpy as np
import re
import sys

main_dir = os.getcwd()
if "GPA_NN" not in main_dir:
    os.chdir("scripts/GPA_NN")
sys.path.append('../../')

# input parameters --------------------------------------
from scripts.util.input_args import input_params
p, _ = input_params()

import yaml
from yaml.loader import SafeLoader
yaml_file = "../../configs/{dataset}-{generative_model}.yaml".format(dataset=p.dataset, generative_model=p.generative_model)


with open(yaml_file, 'r') as f:
    param = yaml.load(f, Loader=SafeLoader)
    #print(param)
    
updated_param = vars(p)
for param_key, param_val in updated_param.items():
    if type(param_val) == type(None):
        continue
    param[param_key] = param_val
    
if param['alpha']:
    par = [param['alpha']]
    param['exptype'] = '%s=%05.2f-%s' % (param['f'], param['alpha'], param['Gamma'])
else: 
    par = []
    param['exptype'] = '%s-%s' % (param['f'], param['Gamma'])
if param['L'] == None:
    param['expname'] = '%s_%s' % (param['exptype'], 'inf')
else:
    param['expname'] = '%s_%.4f' % (param['exptype'], param['L'])

# Data generation ----------------------------------------
from scripts.util.generate_data import generate_data
param, X_, Y_, X_label, Y_label = generate_data(param)
       
if param['dataset'] in ['BreastCancer','Labeled_disease']:
    Q = tf.constant(X_/10.0, dtype=tf.float32) # constant
    P = tf.Variable(Y_/10.0, dtype=tf.float32) # variable
else:
    Q = tf.constant(X_, dtype=tf.float32) # constant
    P = tf.Variable(Y_, dtype=tf.float32) # variable
    
if param['N_conditions'] >1:
    Q_label = tf.constant(X_label, dtype=tf.float32)
    P_label = tf.constant(Y_label, dtype=tf.float32)
    
else:
    Q_label, P_label = None, None
    

data_par = {'P_label': P_label, 'Q_label': Q_label, 'mb_size_P': param['mb_size_P'], 'mb_size_Q': param['mb_size_Q'], 'N_samples_P': param['N_samples_P'], 'N_samples_Q': param['N_samples_Q'], }

print("Data prepared.")


# Discriminator learning  -----------------------------------------
# Discriminator construction using Neural Network
from lib.construct_NN import check_nn_topology, initialize_NN, model

N_fnn_layers, N_cnn_layers, param['activation_ftn'] = check_nn_topology(param['NN_model'], param['N_fnn_layers'], param['N_cnn_layers'], param['N_dim'], param['activation_ftn'])

NN_par = {'NN_model':param['NN_model'], 'activation_ftn':param['activation_ftn'], 'N_dim': param['N_dim'], 'N_cnn_layers':N_cnn_layers, 'N_fnn_layers':N_fnn_layers, 'N_conditions': param['N_conditions'], 'constraint': param['constraint'], 'L': param['L'], 'eps': param['eps']}

W, b = initialize_NN(NN_par)
phi = model(NN_par)  # discriminator

# scalar optimal value optimization for f-divergence
nu = tf.Variable(0.0, dtype=tf.float32)

parameters = {'W':W, 'b':b, 'nu':nu} # Learnable parameters for the discriminator phi

# Train setting
from lib.train_NN import train_disc
lr_phi = tf.Variable(param['lr_phi'], trainable=False) # lr for training a discriminator function

# (Discriminator) Loss ----------------------------------------------
loss_par = {'f': param['f'], 'formulation': param['formulation'], 'par': par, 'reverse': param['reverse'], 'lamda': param['lamda']}

# Transporting particles --------------------------------------------
# ODE solver setting
from lib.transport_particles import calc_vectorfield, solve_ode
dPs = []
if param['ode_solver'] in ['forward_euler', 'AB2', 'AB3', 'AB4', 'AB5']:
    aux_params = []
else:
    aux_params = {'parameters': parameters, 'phi': phi, 'Q': Q, 'lr_phi': lr_phi,'epochs_phi': param['epochs_phi'], 'loss_par': loss_par, 'NN_par': NN_par, 'data_par': data_par, 'optimizer': param['optimizer']}

# Applying mobility to particles
if param['mobility'] == 'bounded':
    from lib.construct_NN import bounded_relu  # mobility that bounding particles (For image data)
        
# Train setting
lr_P_init = param['lr_P'] # Assume that deltat = deltat(t)
if param['ode_solver'] == "DOPRI5": # deltat = deltat(x,t)
    lr_P_init = [param['lr_P']]*param['N_samples_P']
    # Low dimensional example=> rank 2, Image example=> rank 4
    for i in range(1, tf.rank(P)):
        lr_P_init = np.expand_dims(lr_P_init, axis=i)
lr_P = tf.Variable(lr_P_init, trainable=False)
lr_Ps = []


# Evaluating Wasserstein-1 metric ----------------------------------
if param['calc_Wasserstein1'] == True:
    NN_par2 = {'NN_model':param['NN_model'], 'activation_ftn':param['activation_ftn'], 'N_dim': param['N_dim'], 'N_cnn_layers':N_cnn_layers, 'N_fnn_layers':N_fnn_layers, 'N_conditions': param['N_conditions'], 'constraint': param['constraint'], 'L': 1.0, 'eps': param['eps']}

    W2, b2 = initialize_NN(NN_par2)
    phi2 = model(NN_par2)  # discriminator for Wasserstein 1 metric
    parameters2 = {'W':W2, 'b':b2} # Learnable parameters for the discriminator phi2
    
    # Train setting
    from lib.train_NN import train_wasserstein1


# Save & plot settings -----------------------------------------------
# Metrics to calculate
from scripts.util.evaluate_metric import calc_ke, calc_grad_phi
if np.prod(param['N_dim']) <= 12:
    from scripts.util.evaluate_metric import calc_sinkhorn
if np.prod(param['N_dim']) > 100:
    from scripts.util.evaluate_metric import calc_fid
trajectories = []
vectorfields = []
divergences = []
wasserstein1s = []
KE_Ps = []
FIDs = []

# saving/plotting parameters
if param['save_iter'] >= param['epochs']:
    param['save_iter'] = 1

if param['plot_result'] == True:
    from scripts.util.plot_result import plot_result

if not os.path.exists(main_dir + '/assets/' + param['dataset']):
    os.makedirs(main_dir + '/assets/' + param['dataset'])

param['expname'] = param['expname']+'_%04d_%04d_%02d_%s' % (param['N_samples_Q'], param['N_samples_P'], param['random_seed'], param['exp_no'])
filename = main_dir + '/assets/' + param['dataset']+'/%s.pickle' % (param['expname'])

if param['plot_intermediate_result'] == True:
    if 'gaussian' in param['dataset'] and 'Extension' not in param['dataset']:
         r_param = param['sigma_Q']
    elif 'student_t' in param['dataset']:
        r_param = param['nu']
    elif param['dataset'] == 'Extension_of_gaussian':
        r_param = param['a']
    else:
        r_param = None
    
# additional plots for simple low dimensional dynamics
if param['N_dim'] == 1:
    xx = np.linspace(-3, 13, 300)
    xx = tf.constant(np.reshape(xx, (-1,1)), dtype=tf.float32)
    phis = []
elif param['N_dim'] == 2:#'2D' in param['dataset']:
    if "student_t" in param['dataset']:
        nx = 50
        xx = np.linspace(-30, 30, nx)
        yy = np.linspace(-30, 30, nx)
    else:
        nx = 40
        xx = np.linspace(-3, 9, nx)
        yy = np.linspace(-3, 9, nx)
    XX, YY = np.meshgrid(xx, yy)
    xx = np.concatenate((np.reshape(XX, (-1,1)), np.reshape(YY, (-1,1))), axis=1)
    xx = tf.constant(xx, dtype=tf.float32)
    phis = []
    
    
# Train ---------------------------------------------------------------
import matplotlib.pyplot as plt
import time 
t0 = time.time()

for it in range(1, param['epochs']+1): # Loop for updating particles P
    parameters, current_loss, dW_norm = train_disc(parameters, phi, P, Q, lr_phi, param['epochs_phi'], loss_par, NN_par, data_par, param['optimizer'], print_vals=True)
    
    if param['calc_Wasserstein1'] == True:
        parameters2, current_wass1, _ = train_wasserstein1(parameters2, phi2, P, Q, lr_phi, param['epochs_phi'], NN_par2, data_par, param['optimizer'], print_vals=True)
    
    dPs.append( calc_vectorfield(phi, P, parameters, NN_par, loss_par, data_par) )
    
    if param['ode_solver'] == "DOPRI5": # deltat adust
        P, dPs, dP, lr_P = solve_ode(P, lr_P, dPs, param['ode_solver'], aux_params) # update P
    else:
        P, dPs, dP = solve_ode(P, lr_P, dPs, param['ode_solver'], aux_params) # update P

    if param['mobility'] == 'bounded':
        P.assign(bounded_relu(P))
     
    lr_Ps.append(lr_P.numpy())
    # adjust learning rates
    #if it>=100:
    #    lr_P = decay_learning_rate(lr_P, param['lr_P_decay'], {'epochs': param['epochs']-100, 'epoch': it-100, 'KE_P': KE_P})
        
    # save results
    divergences.append(current_loss)
    KE_P = calc_ke(dP, param['N_samples_P'])
    KE_Ps.append(KE_P)
    grad_phi = calc_grad_phi(dP)
    #print("grad", grad_phi)
    if param['calc_Wasserstein1'] == True:
        wasserstein1s.append(current_wass1)
    
    if param['epochs']<=100 or it%param['save_iter'] == 0:
        if param['dataset'] in ['BreastCancer','Labeled_disease']:
            trajectories.append(P.numpy()*10)
        else:
            trajectories.append(P.numpy())
        if np.prod(param['N_dim']) < 500:
            vectorfields.append(dP.numpy())
        elif np.prod(param['N_dim']) >= 784:  # image data
            FIDs.append( calc_fid(pred=P.numpy(), real=Q.numpy()) )
    
    
    if it % (param['epochs']/10) == 0:
        display_msg = 'iter %6d: loss = %.10f, norm of dW = %.2f, kinetic energy of P = %.10f, average learning rate for P = %.6f' % (it, current_loss, dW_norm, KE_P, tf.math.reduce_mean(lr_P).numpy())
        if len(FIDs) > 0 :
            display_msg = display_msg + ', FID = %.3f' % FIDs[-1]   
        print(display_msg)
        print("grad", grad_phi)
        
        if param['plot_intermediate_result'] == True:
            data = {'trajectories': trajectories, 'divergences': divergences, 'wasserstein1s':wasserstein1s, 'KE_Ps': KE_Ps, 'FIDs':FIDs, 'X_':X_, 'Y_':Y_, 'X_label':X_label, 'Y_label':Y_label, 'dt': lr_Ps, 'dataset': param['dataset'], 'r_param': r_param, 'vectorfields': vectorfields, 'save_iter':param['save_iter']}
            if param['N_dim'] ==2:
                data.update({'phi': phi, 'W':W, 'b':b, 'NN_par':NN_par})
            plot_result(filename, intermediate=True, epochs = it, iter_nos = None, data = data, show=False)
        
        
        if np.prod(param['N_dim']) == 1:
            zz = phi(xx,None, W,b,NN_par).numpy()
            zz = np.reshape(zz, -1)
            phis.append(zz)
            
        

total_time = time.time() - t0
print(f'total time {total_time:.3f}s')

if '1D' in p.dataset:
    import matplotlib.pyplot as plt
    for i, yy in enumerate(phis):
        color_gradient = (max(-25/4*(i/9)**2+0.85,0), max(-25/4*(i/9-1/2)**2+0.85,0), max(-25/4*(i/9-1)**2+0.85,0))
        plt.plot(xx, yy, label='t=%.2f' %((i+1)*param['epochs']/10*param['lr_P']), color = color_gradient)
    plt.legend()
    plt.title(r'$\phi_t$')
    f = filename.split('.pickle')
    plt.savefig(f[0]+"-phis.png")
    plt.show()
    
        


# Save result ------------------------------------------------------
import pickle
if param['N_dim'] == 1:
    X_ = np.concatenate((X_, np.zeros(shape=X_.shape)), axis=1)
    Y_ = np.concatenate((Y_, np.zeros(shape=Y_.shape)), axis=1)
    
    trajectories = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in trajectories]
    vectorfields = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in vectorfields]
  
if param['L'] == None:
    param['L'] = 'inf'
param.update({'X_': X_, 'Y_': Y_, 'lr_Ps':lr_Ps,})
result = {'trajectories': trajectories, 'vectorfields': vectorfields, 'divergences': divergences, 'KE_Ps': KE_Ps, 'FIDs': FIDs, 'wasserstein1s': wasserstein1s}

if param['dataset'] in ['BreastCancer','Labeled_disease']:
    np.savetxt(main_dir + "/data/gene_expression_example/GPL570/"+param['dataset']+'/output_norm_dataset_dim_%d.csv' % param['N_dim'], trajectories[-1], delimiter=",")
        
# Save trained data
with open(filename,"wb") as fw:
    pickle.dump([param, result] , fw)
print("Results saved at:", filename)

# Plot final result ------------------------------------
if param['plot_result'] == True:
    plot_result(filename, intermediate=False, show=False)
