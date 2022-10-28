#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid tensorflow warning
import tensorflow as tf
import numpy as np
import re

# input parameters --------------------------------------
from util.input_args import input_params
p = input_params()
param = vars(p)

if p.alpha:    
    par = [p.alpha]
    p.exptype = '%s=%05.2f-%s' % (p.f, p.alpha, p.Gamma)
else: 
    par = []
    p.exptype = '%s-%s' % (p.f, p.Gamma)
if p.L == None:
    p.expname = '%s_%s' % (p.exptype, 'inf')
else:
    p.expname = '%s_%.2f' % (p.exptype, p.L)   

# Data generation ----------------------------------------
from util.generate_data import generate_data
p, X_, Y_, X_label, Y_label = generate_data(p)
       
Q = tf.constant(X_/10.0, dtype=tf.float32) # constant
P = tf.Variable(Y_/10.0, dtype=tf.float32) # variable

if p.N_conditions >1:
    Q_label = tf.constant(X_label, dtype=tf.float32)
    P_label = tf.constant(Y_label, dtype=tf.float32)
else:
    Q_label, P_label = None, None  

# Witness function construction using Neural Network -----------------
from util.construct_NN import check_nn_topology, initialize_NN, model

N_fnn_layers, N_cnn_layers, p.activation_ftn = check_nn_topology(p.NN_model, p.N_fnn_layers, p.N_cnn_layers, p.N_dim, p.activation_ftn)

NN_par = {'NN_model':p.NN_model, 'activation_ftn':p.activation_ftn, 'N_cnn_layers':N_cnn_layers, 'N_fnn_layers':N_fnn_layers, 'N_conditions': p.N_conditions, 'constraint': p.constraint, 'L': p.L}

W, b = initialize_NN(NN_par)
phi = model(NN_par) 


# ML-trained Mobility function construction --------------------------
if p.mobility == "ML":   
    NN_par_m = {'NN_model':p.NN_model+'_mobility', 'activation_ftn':p.mobility_activation_ftn, 'N_cnn_layers':N_cnn_layers, 'N_fnn_layers':N_fnn_layers, 'N_conditions': p.N_conditions, 'constraint': p.constraint, 'L': p.mobility_L,'K': p.mobility_bound}    
    
    W_m, b_m = initialize_NN(NN_par_m)
    mu = model(NN_par_m)


# Loss & Loss_first_variation ------------------------------
loss_par = {'f': p.f, 'formulation': p.formulation, 'par': par, 'reverse': p.reverse, }

def f_star(g, loss_par):
    if loss_par['f'] == 'KL':
        if loss_par['formulation'] =='LT': # f = xlogx 
            return tf.math.exp(g-1)
        elif loss_par['formulation'] =='LT2': # f = xlogx -x +1
            return tf.math.exp(g)-1
    elif loss_par['f'] == 'alpha':
        alpha = loss_par['par'][0]
        if loss_par['formulation'] == 'LT': # f = (x^alpha-1)/(alpha*(alpha-1))
            return 1/alpha*(1/(alpha-1)+tf.math.pow((alpha-1)*tf.nn.relu(g), alpha/(alpha-1)))
            
def divergence(phi, P, Q, P_label, Q_label, W, b, NN_par, loss_par):
    if loss_par['reverse'] == False:
        g1, g2 = phi(P, P_label, W, b, NN_par), phi(Q,Q_label, W,b,NN_par)
    else:
        g1, g2 = phi(Q,Q_label, W,b,NN_par), phi(P, P_label, W, b, NN_par)
        
    if loss_par['formulation'] == 'DV':
        return tf.reduce_mean(g1)-tf.math.log(tf.reduce_mean(tf.math.exp(g2)))
    else: # LT
        return tf.reduce_mean(g1)-tf.reduce_mean(f_star(g2, loss_par))
       

def first_variation(phi, P, Q, P_label, Q_label, W, b, NN_par, loss_par):
    g = phi(P, P_label, W, b, NN_par)
    if loss_par['reverse'] == False: # D(P||Q)
        return g
    else: # D(Q||P)
        if loss_par['f'] == 'KL':
            if loss_par['formulation'] == 'LT': # f = xlogx 
                return -tf.math.exp(g-1)
            elif formulation == 'LT2': # f = xlogx -x +1
                return -tf.math.exp(g)
            elif formulation == 'DV':
                return -tf.math.exp(g)/tf.reduce_mean(tf.math.exp(g))  
        elif loss_par['f'] == 'alpha':
            alpha = loss_par['par'][0]
            if loss_par['formulation'] == 'LT': # f = (x^alpha-1)/(alpha*(alpha-1))
                return -tf.math.multiply(1/alpha, tf.math.add(1/(alpha-1), tf.math.pow((alpha-1)*tf.nn.relu(g), alpha/(alpha-1))) )
                
def gradient_penalty(phi, P, Q, P_label, Q_label, W, b, NN_par, lamda):
    if NN_par['constraint'] == 'soft':
        L = NN_par['L']
        if tf.shape(P)[0] < tf.shape(Q)[0]:
            idx = np.random.shuffle(range(tf.shape(Q)[0]))
            idx = idx[0:tf.shape(P)[0]]
            Q = Q[idx]
            if Q_label != None:
                Q_label = Q_label[idx]
        elif tf.shape(P)[0] > tf.shape(Q)[0]:
            idx = np.random.shuffle(range(tf.shape(P)[0]))
            idx = idx[0:tf.shape(Q)[0]]
            P = P[idx]
            if P_label != None:
                P_label = P_label[idx]
        
        t = np.random.binomial(1, 0.5, size=tf.shape(Q)[0])
        T = tf.constant(np.expand_dims(t, axis=tuple(range(1, tf.rank(Q)))), dtype=tf.float32)
        R = tf.multiply(T,P)+tf.multiply(1-T,Q)
        if P_label != None:
            T_label = tf.constant(np.expand_dims(t, axis=1), dtype=tf.float32)
            R_label = tf.multiply(T_label,P_label)+tf.multiply(1-T_label,Q_label)
        else: 
            R_label = None
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(R)
            phi_R = phi(R,R_label, W,b,NN_par)            
        grad_phi = tape.gradient(phi_R, R)
        return -lamda*tf.math.reduce_mean(tf.nn.relu(tf.math.square(grad_phi,2)/L**2-1.0))
    else:
        return tf.constant(0.0, dtype=tf.float32)
        

# Train -------------------------------------------------
from util.train_NN import decay_learning_rate, update_NN
lr_NN = tf.Variable(p.lr_NN, trainable=False)
lr_P = tf.Variable(p.lr_P, trainable=False)
lr_Ps = []

from util.evaluate_metric import calc_fid, calc_ke
trajectories = []
vectorfields = []
mobilities = []
divergences = []
KE_Ps = []
FIDs = []


if p.N_dim == 1:
    xx = np.linspace(-10, 10, 300)
    xx = tf.constant(np.reshape(xx, (-1,1)), dtype=tf.float32)
    phis = []
    if p.mobility == 'ML':
        mus = []
elif p.N_dim == 2:#'2D' in p.dataset:
    xx = np.linspace(-10, 10, 40)
    yy = np.linspace(-10, 10, 40)
    XX, YY = np.meshgrid(xx, yy)
    xx = np.concatenate((np.reshape(XX, (-1,1)), np.reshape(YY, (-1,1))), axis=1)
    xx = tf.constant(xx, dtype=tf.float32)
    phis = []
    if p.mobility == 'ML':
        mus = []

import time 
t0 = time.time()
for it in range(1, p.epochs+1): 
    #if p.mobility == 'ML':
    #    W_m, b_m = initialize_NN(NN_par_m)    
    loss = tf.add(divergence(phi, P, Q, P_label, Q_label, W, b, NN_par, loss_par), 10.0*gradient_penalty(phi, P, Q, P_label, Q_label, W, b, NN_par, p.lamda))
    current_loss = loss.numpy()
    for in_it in range(p.epochs_nn): # train NN to get phi* 
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([W, b])
            loss = tf.add(divergence(phi, P, Q, P_label, Q_label, W, b, NN_par, loss_par), 10.0*gradient_penalty(phi, P, Q, P_label, Q_label, W, b, NN_par, p.lamda)) 
        dW, db = tape.gradient(loss, [W,b])
        
        W, b, dW_norm = update_NN(W, b, dW, db, lr_NN, NN_par, descent=False, calc_dW_norm=True)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(P)
        loss_first_variation = first_variation(phi, P, Q, P_label, Q_label, W, b, NN_par,loss_par)         
    dP = tape.gradient(loss_first_variation, P) # update P
    
    if p.mobility == 'ML':
        for in_it in range(p.epochs_nn): 
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch([W_m,b_m])
                if p.mobility_activation_ftn[-1] == 'linear':
                    m = tf.math.square(mu(P,P_label, W_m,b_m,NN_par_m))
                else:
                    m = mu(P,P_label, W_m,b_m,NN_par_m)
                P_tilde = P - lr_P*m*dP
                loss_tilde = tf.add(divergence(phi, P_tilde, Q, P_label, Q_label, W, b, NN_par, loss_par), p.mobility_lamda*tf.math.square(m)/loss)
            dW_m, db_m = tape.gradient(loss_tilde, [W_m, b_m]) 
            
            W_m, b_m, dW_norm_m = update_NN(W_m, b_m, dW_m, db_m, lr_NN, NN_par_m, descent=True)
            
            if p.mobility_activation_ftn[-1] == 'linear':
                m = tf.math.square(mu(P,P_label, W_m,b_m,NN_par_m))
            else:
                m = mu(P,P_label, W_m,b_m,NN_par_m)
    elif p.mobility == 'potential':
        if p.formulation == "DV":
            shift = tf.math.log(tf.reduce_mean(tf.math.exp(phi(Q,Q_label, W,b,NN_par))))
        else: # LT
            shift = tf.reduce_mean(f_star(phi(Q,Q_label, W,b,NN_par), loss_par))
        shift = tf.math.minimum(0, shift)
        m = phi(P,P_label, W,b,NN_par)-shift
    else:
        m = tf.constant(1.0, dtype=tf.float32)    
        
    P.assign(P - lr_P*m*dP)
    
    # save results
    divergences.append(current_loss)
    KE_P = calc_ke(dP, m, p.N_samples_P)
    KE_Ps.append(KE_P)    
    
    if p.epochs<=100 or it%p.save_iter == 0:
        trajectories.append(P.numpy()*10)
        mobilities.append(tf.reshape(m,-1).numpy())
        if np.prod(p.N_dim) < 500:
            vectorfields.append(dP.numpy())
        elif np.prod(p.N_dim) >= 784:  # image data
            if 'MNIST' in p.dataset:
                fid_model_name = 'autoencoder_mnist'
            elif 'CIFAR10' in p.dataset:
                fid_model_name = 'autoencoder_cifar10'
            FIDs.append( calc_fid(pred=P.numpy(), real=Q.numpy(), model_name=fid_model_name) )
            
    # adjust learning rates
    lr_Ps.append(lr_P.numpy())
    if it>=100:
        lr_P = decay_learning_rate(lr_P, p.lr_P_decay, {'epochs': p.epochs-100, 'epoch': it-100, 'KE_P': KE_P})
    
    # display intermediate results
    if it % (p.epochs/10) == 0:
        display_msg = 'iter %6d: loss = %.10f, grad_norm of W = %.2f, kinetic energy of P = %.10f, learning rate for P = %.6f' % (it, current_loss, dW_norm, KE_P, lr_P.numpy())
        if len(FIDs) > 0 :
            display_msg = display_msg + ', FID = %.3f' % FIDs[-1]   
        display_msg = display_msg + ', mobility maximum: %.2f minimum: %.2f' % (tf.math.reduce_max(m).numpy(), tf.math.reduce_min(m).numpy())
        print(display_msg)
        
        if p.N_dim == 1 or p.N_dim == 2:
            zz = phi(xx,None, W,b,NN_par).numpy()
            zz = np.reshape(zz, -1)
            phis.append(zz)
            if p.mobility == 'ML':
                if p.mobility_activation_ftn[-1] =='linear':
                    zz = tf.square(mu(xx,None, W_m,b_m,NN_par_m)).numpy()
                else:
                    zz = mu(xx,None, W_m,b_m,NN_par_m).numpy()
                zz = np.reshape(zz, -1)
                mus.append(zz)
            

total_time = time.time() - t0
print(f'total time {total_time:.3f}s')

# Save result ------------------------------------------------------
import pickle
if not os.path.exists(p.dataset):
    os.makedirs(p.dataset)

if '1D' in p.dataset:
    X_ = np.concatenate((X_, np.zeros(shape=X_.shape)), axis=1)
    Y_ = np.concatenate((Y_, np.zeros(shape=Y_.shape)), axis=1)
    
    trajectories = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in trajectories]
    vectorfields = [np.concatenate((x, np.zeros(shape=x.shape)), axis=1) for x in vectorfields]
        
param.update({'X_': X_, 'Y_': Y_, 'lr_Ps':lr_Ps})

p.expname = p.expname+'_%04d_%04d_%02d_%s' % (p.N_samples_Q, p.N_samples_P, p.random_seed, p.exp_no)        
filename = p.dataset+'/%s.pickle' % (p.expname)
    
result = {'trajectories': trajectories, 'vectorfields': vectorfields, 'divergences': divergences, 'KE_Ps': KE_Ps, 'FIDs': FIDs, 'mobilities': mobilities}

if p.dataset in ['BreastCancer',]:
    np.savetxt("gene_expression_example/GPL570/"+p2.dataset+'/output_norm_dataset_dim_%d.csv' % p.N_dim, trajectories[-1], delimiter=",")
        
## Save trained data
with open(filename,"wb") as fw:
    pickle.dump([param, result] , fw)
print("Results saved at:", filename)
#--------------
#plot
if p.plot_result == True:
    plot_velocity = True
    if len(vectorfields)==0 or '1D' in p.dataset or '2D' in p.dataset:
        plot_velocity = False
    
    quantile = True
    
    if 'MNIST' in filename or 'CIFAR10' in filename:
        from plot_result import plot_losses, plot_trajectories_img, plot_trained_img, images_to_animation, plot_tiled_images
        epochs = 0
        iter_nos = None
        
        images_to_animation(trajectories=None, N_samples_P=None, dt=None, physical_time=True, pick_samples = None, epochs=epochs, save_gif=True, filename = filename)
        plot_trajectories_img(X_ = None,Y_=None, trajectories = None, dt = None, pick_samples=None, epochs=epochs, iter_nos = iter_nos, physical_time=True, filename=filename)
        plot_trained_img(X_ = None, trajectories = None, pick_samples=None, epochs=0, filename=filename)
        
        plot_losses(loss_type='divergences', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
        plot_losses(loss_type='KE_Ps', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
        plot_losses(loss_type='FIDs', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
        
        if 'all' in filename: # conditional gpa
            plot_tiled_images(print_multiplier=10, last_traj=None, last_digit=None, epochs = 0, data = None, data_label=None, filename=filename)
            
    ## 2D embedded in high dimensions examples 
    elif 'submnfld' in filename:
        from plot_result import plot_losses, plot_trajectories, plot_speeds, plot_orth_axes_saturation, plot_initial_data
        iter_nos = None
        exp_alias_ = None
        track_velocity = True
        iscolor = True
        quantile = True
        
        plot_initial_data(proj_axes = [5,6], x_lim = [None,None],y_lim = [None,None], filename=filename)
        plot_trajectories(trajectories=None, dt=None, X_=None, Y_=None, r_param=None, vectorfields = [], mobilities = [], proj_axes = [5,6], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1, track_velocity=track_velocity, arrow_scale = 1, iscolor=iscolor, quantile=quantile, exp_alias_ = exp_alias_, x_lim = [None,None],y_lim = [None,None],  filename = filename)
        plot_orth_axes_saturation(N_dim = None, Y_ = None, trajectories = None, save_iter = 1,dt = None, proj_axes=[5,6], epochs=0, iter_nos=None, physical_time = True, filename = filename) 
        plot_speeds(vectorfields =  None, mobilities = None, N_dim = None, save_iter=1, dt=None, plot_scale='semilogy', proj_axes = [5,6], physical_time=True, epochs=0, filename = filename)
        plot_losses(loss_type='divergences', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
        plot_losses(loss_type='KE_Ps', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
    
    ## low dimensional example
    elif filename != "" :  
        from plot_result import plot_trajectories, plot_losses, plot_initial_data
        iter_nos = None
        exp_alias_ = None
        track_velocity = False
        iscolor = False
        quantile = True
    
        plot_initial_data(proj_axes = [0,1], x_lim = [None,None],y_lim = [None,None], filename=filename)
        plot_trajectories(trajectories=None, dt=None, X_=None, Y_=None, r_param=None, vectorfields = [], mobilities = [], proj_axes = [0,1], pick_samples =None, epochs = 0, iter_nos = None, physical_time=True, save_iter = 1, track_velocity=track_velocity, arrow_scale = 1, iscolor=iscolor, quantile=quantile, exp_alias_ = exp_alias_, x_lim = [None,None],y_lim = [None,None],  filename = filename)
        plot_losses(loss_type='divergences', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
        plot_losses(loss_type='KE_Ps', loss_state=None, plot_scale='semilogy', fitting_line=False, save_iter = 1, dt = None, iter_nos = None, exp_alias_=None, epochs=0, ylims=None, physical_time=True, filename=filename)
            
        if '1D' in p.dataset:
            #from plot_result import plot_velocities_1D
            #plot_velocities_1D(filename, plot_scale='semilogy', physical_time=True, epochs=0)
            import matplotlib.pyplot as plt
            for i, yy in enumerate(phis):
                plt.plot(xx, yy, label='t=%.2f' %((i+1)*p.epochs/10*p.lr_P))
            plt.legend()
            plt.title(r'$\phi_n, n=10*k$')
            f = filename.split('.pickle')
            plt.savefig(f[0]+"-phis.png")
            plt.show()
            
            if p.mobility == 'ML':
                for i, yy in enumerate(mus):
                    plt.plot(xx, yy, label='t=%.2f'%((i+1)*p.epochs/10*p.lr_P))
                plt.legend()
                plt.title(r'$\mu_n, n=10*k$')
                f = filename.split('.pickle')
                plt.savefig(f[0]+"-mus.png")
                plt.show()
                
        elif '2D' in p.dataset:
            #from plot_result import plot_velocities_2D
            #plot_velocities_2D(filename, plot_scale='semilogy', physical_time=True, epochs=0)
            
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = Axes3D(juufig)
            ZZ = np.reshape(phis[-1],(40,40))
            ax.plot_surface(XX, YY, ZZ)
            #for i, zz in enumerate(phis):
            #    Axes3D.plot_surface(XX, YY, zz, label=i)
            #plt.legend()
            ax.set_title(r'$\phi$')
            f = filename.split('.pickle')
            plt.savefig(f[0]+"-phis.png")
            plt.show()
            
            if p.mobility == 'ML':
                fig = plt.figure()
                ax = Axes3D(fig)
                ZZ = np.reshape(mus[-1],(40,40))
                ax.plot_surface(XX, YY, ZZ)
                ax.set_title(r'$\mu$')
                f = filename.split('.pickle')
                plt.savefig(f[0]+"-mus.png")
                plt.show()
