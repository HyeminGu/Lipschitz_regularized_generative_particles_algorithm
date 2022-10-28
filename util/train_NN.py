import tensorflow as tf
from util.construct_NN import spectral_normalization

# learning rate decay
def decay_learning_rate(lr, schedule, param_dict):
    if schedule == 'rational':
        maxiter = param_dict['epochs']
        it = param_dict['epoch']
        lr.assign(lr*(1-it/maxiter))
    elif schedule == 'step':
        KE_P = param_dict['KE_P']
        KE_P_lim = 1e-4
        KE_P_alpha_lim = 0.001
        if KE_P <= KE_P_lim + 2*KE_P_alpha_lim:
            lr.assign(lr/2)
    return lr
    
 
def update_NN(W, b, dW, db, lr_NN, NN_par, descent=True, calc_dW_norm=False):
    dW_norm = 0
    if descent == False:
        lr_NN = - lr_NN
    for l in range(len(W)):
        if calc_dW_norm == True:
            dW_norm = max(dW_norm, tf.norm(dW[l]))
        W[l].assign(W[l] - lr_NN*dW[l])    
        if NN_par['constraint'] == 'hard' and NN_par['L'] != None: # spectral normalization
            spectral_normalization(W[l], NN_par['L']**(1/len(W)))
        if db[l] != None: # fnn  
            b[l].assign(b[l] - lr_NN*db[l])
            
    return W, b, dW_norm