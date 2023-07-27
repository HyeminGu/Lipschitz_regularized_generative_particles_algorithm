import tensorflow as tf
import numpy as np

#@tf.function
def f_star(g, loss_par): # f^*(g) = inf_x <x,g>-f(x)
    threshold = 0.01
    if loss_par['f'] == 'KL': # f = xlogx -x +1
        return tf.math.exp(g)-1
    elif loss_par['f'] == 'alpha':
        alpha = loss_par['par'][0] # f = (x^alpha-1)/(alpha*(alpha-1))
        return 1/alpha*(1/(alpha-1)+tf.math.pow((alpha-1)*tf.nn.relu(g), alpha/(alpha-1)))
    elif loss_par['f'] == 'reverse_KL': # f = -logx
        max_g = tf.math.reduce_max(g)
        if max_g > threshold: # numerical stability
            return -1 - tf.math.log(-g+threshold)
        else:
            return -1 - tf.math.log(-g)
    elif loss_par['f'] == 'JS': # Jensen-Shannon
        max_exp_g = tf.math.reduce_max(tf.math.exp(g))
        if max_exp_g > 2.0-threshold: # numerical stability
            return -tf.math.log( 2 - tf.math.exp(g) + threshold )
        else:
            return -tf.math.log( 2 - tf.math.exp(g) )
            

# --------------------------

def E_phi(x, x_label, N_x, mb_size_x, phi, W, b, NN_par):
    g1 = []
    if NN_par['N_conditions'] > 1: # indexing for x_label
        for n in range(int(N_x/mb_size_x)):
            g1.append(tf.math.reduce_sum(phi(x[n*mb_size_x:(n+1)*mb_size_x], x_label[n*mb_size_x:(n+1)*mb_size_x], W,b, NN_par)))
        if int(N_x/mb_size_x) * mb_size_x < N_x:
            g1.append( tf.reduce_sum(phi(x[int(N_x/mb_size_x):],x_label[n*mb_size_x:(n+1)*mb_size_x], W,b, NN_par)) )
        
    else: # not indexing for x_label=None
        for n in range(int(N_x/mb_size_x)):
            g1.append(tf.math.reduce_sum(phi(x[n*mb_size_x:(n+1)*mb_size_x], x_label, W,b, NN_par)))
        if int(N_x/mb_size_x) * mb_size_x < N_x:
            g1.append( tf.reduce_sum(phi(x[int(N_x/mb_size_x):],x_label, W,b, NN_par)) )
            
    return tf.add_n(g1)/N_x

    
def E_fstar_phi(x, x_label, N_x, mb_size_x, phi, nu, W, b, NN_par, loss_par):
    g2 = []
    if NN_par['N_conditions'] > 1: # indexing for x_label
        for n in range(int(N_x/mb_size_x)):
            if loss_par['formulation'] == 'DV':
                g2.append(tf.math.reduce_sum(tf.math.exp(phi(x[n*mb_size_x:(n+1)*mb_size_x], x_label[n*mb_size_x:(n+1)*mb_size_x], W,b, NN_par))))
            else: # LT
                g2.append(tf.math.reduce_sum(f_star(phi(x[n*mb_size_x:(n+1)*mb_size_x], x_label[n*mb_size_x:(n+1)*mb_size_x], W,b, NN_par)-nu, loss_par)))
                
        if int(N_x/mb_size_x) * mb_size_x < N_x:
            if loss_par['formulation'] == 'DV':
                g2.append( tf.reduce_sum(tf.math.exp(phi(x[int(N_x/mb_size_x):],x_label[int(N_x/mb_size_x):], W,b, NN_par))) )
            else: # LT
                g2.append( tf.reduce_sum(f_star(phi(x[int(N_x/mb_size_x):],x_label[int(N_x/mb_size_x):], W,b, NN_par)-nu, loss_par)) )

    else: # don't do indexing for x_label=None
        for n in range(int(N_x/mb_size_x)):
            if loss_par['formulation'] == 'DV':
                g2.append(tf.reduce_sum(tf.math.exp(phi(x[n*mb_size_x:(n+1)*mb_size_x],x_label, W,b, NN_par))))
            else: # LT
                g2.append(tf.reduce_sum(f_star(phi(x[n*mb_size_x:(n+1)*mb_size_x],x_label, W,b, NN_par)-nu, loss_par)))
            
        if int(N_x/mb_size_x) * mb_size_x < N_x:
            if loss_par['formulation'] == 'DV':
                g2.append( tf.reduce_sum(tf.math.exp(phi(x[int(N_x/mb_size_x):],x_label, W,b, NN_par))) )
            else: # LT
                g2.append( tf.reduce_sum(f_star(phi(x[int(N_x/mb_size_x):],x_label, W,b, NN_par)-nu, loss_par)) )
    
    return tf.add_n(g2)/N_x


def divergence_mb(phi, nu, P, Q, W, b, NN_par, loss_par, data_par):
    N_P, N_Q, mb_size_P, mb_size_Q = data_par['N_samples_P'], data_par['N_samples_Q'], data_par['mb_size_P'], data_par['mb_size_Q']
    P_label, Q_label = data_par['P_label'], data_par['Q_label']
   

    if loss_par['reverse'] == False: # D(P||Q)
        g1 = E_phi(P, P_label, N_P, mb_size_P, phi, W, b, NN_par)
        g2 = E_fstar_phi(Q, Q_label, N_Q, mb_size_Q, phi, nu, W, b, NN_par, loss_par)
    else: # D(Q||P)
        g1 = E_phi(Q, Q_label,  N_Q, mb_size_Q, phi, W, b, NN_par)
        g2 = E_fstar_phi(P, P_label, N_P, mb_size_P, phi, nu, W, b, NN_par, loss_par)
        
    
    if loss_par['formulation'] == 'DV':
        return g1 - tf.math.log(g2)
    else:
        return g1 - g2 - nu
        
        
def wasserstein1_mb(phi, P, Q, W, b, NN_par, data_par):
    N_P, N_Q, mb_size_P, mb_size_Q = data_par['N_samples_P'], data_par['N_samples_Q'], data_par['mb_size_P'], data_par['mb_size_Q']
    P_label, Q_label = data_par['P_label'], data_par['Q_label']
    
    g1 = E_phi(P, P_label, N_P, mb_size_P, phi, W, b, NN_par)
    g2 = E_phi(Q, Q_label, N_Q, mb_size_Q, phi, W, b, NN_par)
    
    return g1 - g2
        


def calc_grad_phi(dP_dt):
    return np.mean(np.linalg.norm(dP_dt, axis=1))


# @tf.function : might not be compatible to soft lipschitz constraint loss
def gradient_penalty(phi, P, Q, W, b, NN_par, data_par, lamda):
    P_label, Q_label = data_par['P_label'], data_par['Q_label']
    if NN_par['constraint'] == 'soft':
        L = NN_par['L']
        
        '''
        N_tot = min((200, P.shape[0], Q.shape[0]))
        N_P = int(N_tot*P.shape[0]/(P.shape[0]+Q.shape[0]))
        N_Q = int(N_tot*Q.shape[0]/(P.shape[0]+Q.shape[0]))
        r_P = np.random.randint(int(P.shape[0]/N_P))
        r_Q = np.random.randint(int(Q.shape[0]/N_Q))
        R = tf.concat([P[r_P*N_P:(r_P+1)*N_P], Q[r_Q*N_Q:(r_Q+1)*N_Q]], axis=0)
    
        if P_label != None:
            R_label = tf.concat([P_label[r_P*N_P:(r_P+1)*N_P], Q_label[r_Q*N_Q:(r_Q+1)*N_Q]], axis=0)
        else:
            R_label = None
        '''
        
        R = P
        R_label = P_label
  
            
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(R)
            phi_R = phi(R,R_label, W,b,NN_par)
        dR = tape.gradient(phi_R, R)
        
        grad_phi = calc_grad_phi(dR)
        
    
        
        return tf.multiply(-lamda, tf.math.reduce_mean(tf.nn.relu(tf.math.square(dR/L)-1.0)))
    else:
        return tf.constant(0.0, dtype=tf.float32)

