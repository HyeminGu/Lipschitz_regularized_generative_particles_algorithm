import tensorflow as tf

class UndefinedError(Exception):
    """Exception raised for errors in the undefined Neural network topology.
    
    Attributes:
        NN_model -- model with undefined network topology
    """
    
    def __init__(self, NN_model):
        self.NN_model = NN_model
    
    def __str__(self):
        if self.NN_model == 'fnn':
            return('N_fnn_layers is undefined for the fnn.')
        elif self.NN_model == 'cnn':
            return('N_cnn_layers is undefined for the cnn.')
        else:
            return('N_fnn_layers is undefined for the fnn. \nN_cnn_layers is undefined for the cnn.')

def check_nn_topology(NN_model, N_fnn_layers, N_cnn_layers, N_dim, activation_ftn):
    undefined = ''
    if 'cnn' in NN_model: 
        if N_cnn_layers == None:
            undefined = undefined + 'cnn'
        else: 
            N_cnn_layers = [N_dim]+N_cnn_layers
    
    if 'fnn' in NN_model:
        if N_fnn_layers == None:
            undefined = undefined + 'fnn'  
        else:
            if 'cnn' in NN_model: # cnn+fnn
                x, y = (N_dim[0], N_dim[1])
                for i in N_cnn_layers[1:]:
                    x, y = (x//2+x%2, y//2+y%2)
                
                N_fnn_layers = [N_cnn_layers[-1]*x*y]+N_fnn_layers
                if len(activation_ftn) == 1:
                    activation_ftn = ['leaky_relu', 'relu']
            else: # fnn
                N_fnn_layers = [N_dim]+N_fnn_layers
            
    if undefined != '':
        raise UndefinedError(undefined)
        
    return N_fnn_layers, N_cnn_layers, activation_ftn
    
# ---------------------------------------------------------
def spectral_normalization(W, norm_=1):
    W.assign(tf.math.scalar_mul(norm_/tf.norm(W, 2), W))
    
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.)
    return tf.random.normal(shape=size, stddev=xavier_stddev) 

def initialize_W(layers, N_conditions):
    W_init, NN_W, NN_b=[], [], []
    
    if layers == None:
        return NN_W, NN_b
        
    num_layers = len(layers)
    if type(layers[0])==int: # fnn
        for l in range(0,num_layers-1): 
            b = tf.Variable(tf.zeros([1,layers[l+1]]), dtype=tf.float32)
            NN_b.append(b)
            W_init.append(xavier_init(size=[layers[l], layers[l+1]]))
        b = tf.Variable(tf.zeros([1,N_conditions]), dtype=tf.float32)
        NN_b.append(b)
        W_init.append(xavier_init(size=[layers[-1], N_conditions]))
    else: # cnn
        # filter shape: [filter_height, filter_width, in_channels, out_channels]
        NN_b = [tf.constant(1, dtype=tf.float32)]*(num_layers-1)
        W_init.append(xavier_init(size=[5,5, layers[0][-1], layers[1]])) 
        for l in range(1,num_layers-1):    
            W_init.append(xavier_init(size=[5,5, layers[l], layers[l+1]]))
      
    for w in W_init:
        W = tf.Variable(w, dtype=tf.float32) 
        NN_W.append(W)
     
    return NN_W, NN_b

def initialize_NN(NN_par):
    [W_cnn,b_cnn]=initialize_W(NN_par['N_cnn_layers'], NN_par['N_conditions'])
    [W_fnn,b_fnn]=initialize_W(NN_par['N_fnn_layers'], NN_par['N_conditions'])
    
    W = W_cnn+W_fnn
    if NN_par['constraint'] == 'hard' and NN_par['L'] != None:
        for w in W:
            w = spectral_normalization(w, NN_par['L']**(1/len(W)))       
    b = b_cnn+b_fnn
    
    return W, b

# ----------------------------------------------------------------
def bounded_relu(x):
    return tf.math.subtract(tf.nn.relu(x), tf.nn.relu(x-1))
    
def bounded_elu(x):
    return tf.math.subtract(tf.nn.elu(x), tf.nn.elu(x-1))
    
def determine_activation_ftn(ftn_name):
    # activation function that is comatible with Spectral normalization to enforce Lipschitz continuity
    
    # unbounded
    if ftn_name == 'relu':
        activation_ftn = tf.nn.relu
    elif ftn_name == 'leaky_relu':
        activation_ftn = tf.nn.leaky_relu
    elif ftn_name == 'softplus':
        activation_ftn = tf.math.softplus
    elif ftn_name == 'elu':
        activation_ftn = tf.nn.elu
    elif ftn_name == 'abs':
        activation_ftn = tf.math.abs
    elif ftn_name == 'linear':
        activation_ftn = tf.identity
        
    # bounded
    elif ftn_name == 'bounded_relu':
        activation_ftn = bounded_relu
    elif ftn_name == 'bounded_elu':
        activation_ftn = bounded_elu
        
    elif ftn_name == 'sigmoid': # not compatible with SN
        activation_ftn = tf.nn.sigmoid
        
    return activation_ftn


# --------------------------------------------------------------- 
# NN structure for phi

def fnn(x, x_label, W, b, NN_par):
    num_layers = len(W)
    activation_ftn = determine_activation_ftn(NN_par['activation_ftn'][0])
        
    h = x
    for l in range(0,num_layers-1):
        h = activation_ftn(tf.add(tf.matmul(h, W[l]), b[l]))
    out=tf.add(tf.matmul(h, W[-1]), b[-1])
    
    if NN_par['N_conditions'] > 1:
        out = tf.math.reduce_sum(tf.math.multiply(out, x_label), axis=1, keepdims=True)
    
    return out


def cnn_fnn(x, x_label, W, b, NN_par):
    num_layers = len(NN_par['N_cnn_layers'])-1
    activation_ftn = determine_activation_ftn(NN_par['activation_ftn'][0])
    N_samples = x.shape[0]
    
    h = x
    #print(W)
    for l in range(num_layers):
        h = tf.nn.dropout(activation_ftn(tf.nn.conv2d(input=h,filters=W[l],strides=(2,2),padding='SAME')), rate=0.3)
        
    # flatten and dense layers
    activation_ftn2 = determine_activation_ftn(NN_par['activation_ftn'][1])
        
    h = tf.transpose(h, perm=[3, 2,1,0])
    h = tf.transpose(tf.reshape(h, [-1,N_samples]))
    num_layers2 = len(NN_par['N_fnn_layers'])
    
    for l in range(num_layers,num_layers+num_layers2-1):
        #print(tf.shape(W[l]))
        #print(tf.shape(b[l]))
        h = activation_ftn2(tf.add(tf.matmul(h, W[l]), b[l]))
    out=tf.add(tf.matmul(h, W[-1]), b[-1])
    if NN_par['N_conditions'] > 1:
        out = tf.math.reduce_sum(tf.math.multiply(out, x_label), axis=1, keepdims=True)
        
    out = tf.expand_dims(tf.expand_dims(out, axis=-1), axis=-1)
    return out
    
# ----------------------------------------------------------------
# NN structure for the mobility mu
def fnn_mobility(x, x_label, W, b, NN_par):
    num_layers = len(W)
    activation_ftn = determine_activation_ftn(NN_par['activation_ftn'][0])
    activation_ftn_last_layer = determine_activation_ftn(NN_par['activation_ftn'][1])
    if NN_par['K'] == None:
        NN_par['K'] = 1.0
        
    h = tf.add(tf.matmul(x, W[0]), b[0])
    for l in range(1,num_layers):
        h = tf.add(tf.matmul(activation_ftn(h), W[l]), b[l])
    out = NN_par['K']*activation_ftn_last_layer(h)   
    
    if NN_par['N_conditions'] > 1:
        out = tf.math.reduce_sum(tf.math.multiply(out, x_label), axis=1, keepdims=True)
    
    return out


def cnn_fnn_mobility(x, x_label, W, b, NN_par):
    num_layers = len(NN_par['N_cnn_layers'])-1
    activation_ftn = determine_activation_ftn(NN_par['activation_ftn'][0])
    N_samples = x.shape[0]
    if NN_par['K'] == None:
        NN_par['K'] = 1.0
    
    h = x
    for l in range(num_layers):
        h = tf.nn.dropout(activation_ftn(tf.nn.conv2d(input=h,filters=W[l],strides=(2,2),padding='SAME')), rate=0.3)
        
    # flatten and dense layers
    activation_ftn2 = determine_activation_ftn(NN_par['activation_ftn'][1])
    activation_ftn_last_layer = determine_activation_ftn(NN_par['activation_ftn'][2])
        
    h = tf.transpose(h, perm=[3, 2,1,0])
    h = tf.transpose(tf.reshape(h, [-1,N_samples]))
    num_layers2 = len(NN_par['N_fnn_layers'])
    
    h = tf.add(tf.matmul(h, W[num_layers]), b[num_layers])
    for l in range(num_layers+1,num_layers+num_layers2):
        h = tf.add(tf.matmul(activation_ftn(h), W[l]), b[l])
    out = NN_par['K']*activation_ftn_last_layer(h)
    
    if NN_par['N_conditions'] > 1:
        out = tf.math.reduce_sum(tf.math.multiply(out, x_label), axis=1, keepdims=True)
        
    out = tf.expand_dims(tf.expand_dims(out, axis=-1), axis=-1)
    
    return out
 
# ---------------------------------------------------------------- 
def model(NN_par):
    if NN_par['NN_model'] == 'fnn':
        return fnn
    elif NN_par['NN_model'] == 'cnn-fnn':
        return cnn_fnn
    elif NN_par['NN_model'] == 'fnn_mobility':
        return fnn_mobility
    elif NN_par['NN_model'] == 'cnn-fnn_mobility':
        return cnn_fnn_mobility
    