import numpy as np
import tensorflow as tf

def generate_gaussian(
    size, #: tuple[int], # (N, d)
    m, #: float,
    std: float,
    random_seed: int,
):
    np.random.seed(random_seed)
    
    if np.isscalar(m) == False:
        m = np.tile(m, reps=(size[0],1))
    X = std * np.random.standard_normal(size) + m

    return X
		
def generate_four_gaussians(
    size, #: tuple[int], # (N, d)
    dist: float,
    std: float,
    random_seed: int
):
    np.random.seed(random_seed)

    ms = np.array([[0, dist], [dist, 0], [dist, dist], [0, 0]]).T
    num_clusters = ms.shape[1]

    idxs = np.random.choice(num_clusters, size[0])
    means = np.array([ms[:, idx] for idx in idxs])
    X = std * np.random.standard_normal(size) + means
    
    return X
    

def unadjusted_langevin_algorithm(
    max_iter, lr, random_seed, N, d, potential, params={}
):
    tf.random.set_seed(random_seed)
    Y = tf.Variable(tf.random.normal(shape=(N, d)), dtype=tf.float32)

    for epoch in range(max_iter):
        with tf.GradientTape() as tape:
            U = potential(Y, params)          
        dU = tape.gradient(U, Y) 
        Y.assign(Y - lr*dU + tf.math.sqrt(2*lr) * tf.random.normal(shape=(N, d)))    
    return Y.numpy()

def generate_stretched_exponential(
    size, #: tuple[int], # (N, d1, d2, ..., dn)
    beta: float,
    random_seed: int
):
    N = size[0]
    d = np.prod(size[1:])
    def potential_func(X, params):
        # upper case: matrix/vectors, lower case: scalar 
        beta = tf.constant(params['beta'], dtype=tf.float32)
        
        C = tf.constant(np.tile(params['center'], (N,1)), dtype=tf.float32)
        S = tf.constant(np.tile(params['scale'], (N,1)), dtype=tf.float32)
        
        potential = tf.reduce_sum(tf.math.pow(tf.math.abs(tf.math.divide(tf.math.subtract(X, C),S)), beta))              
        return potential
    max_iter, lr = 5000, 0.01
    
    X = unadjusted_langevin_algorithm(
    max_iter, lr, random_seed, N, d, potential_func, 
    params={'beta': beta, 'center': np.zeros(d), 'scale': np.ones(d)})    
    return np.reshape(X, size)
    
def generate_student_t(
    size, #: tuple[int], # (N, d)
    m: float,  # center
    nu: float, # degree of freedom
    random_seed: int
):
    np.random.seed(random_seed)
    
    X = np.zeros(np.prod(size))
    for i in range(np.prod(size)):
        U = 2*np.random.uniform(size=(2))-1
        while np.sum(U**2)>1:
            U = 2*np.random.uniform(size=(2))-1
        W = np.sum(U**2)
        C_sq, R_sq = U[0]**2/W, nu*(W**(-2/nu)-1)
        X[i] = np.sqrt(C_sq*R_sq)
    X = np.reshape(X, size)
    sgn = 2*np.random.binomial(n=1, p=1/2, size=size)-1
    
    if np.isscalar(m) == False:
        m = np.tile(m, reps=(size[0],1))
    return sgn*X+m  
    
def generate_four_student_t(
    size, #: tuple[int], # (N, d)
    dist: float,
    nu: float, # degree of freedom
    random_seed: int
):
    np.random.seed(random_seed)

    ms = np.array([[dist, dist], [-dist, dist], [dist, -dist], [-dist, -dist]]).T
    num_clusters = ms.shape[1]

    idxs = np.random.choice(num_clusters, size[0])
    means = np.array([ms[:, idx] for idx in idxs])
    
    X = generate_student_t(size, 0.0, nu, random_seed) + means
    
    return X
    
def generate_uniform(
    size, #: tuple[int], (N, d)
    shift,#: float,
    l,#: float, 
    random_seed: int
):
    np.random.seed(random_seed)
    
    if np.isscalar(shift) == False:
        shift = np.tile(shift, reps=(size[0],1))
    if np.isscalar(l) == False:
        l = np.tile(l, reps=(size[0],1)) # multiplied element-wise
    X = l * np.random.random(size) + shift 

    return X
    
def generate_one_hot_encoding(
    N: int,
    N_classes: int,
    random_seed: int,
    data=[]
):
    if data == []:
        np.random.seed(random_seed)
        data = np.random.randint(N_classes, size=N)
    
    X_label = np.zeros((data.size, data.max()+1))
    X_label[np.arange(data.size),data] = 1
    
    return X_label
    
def import_mnist(
    N: int,
    label, #: int,
    normalized: bool,
    random_seed: int
):
    np.random.seed(random_seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    if normalized == True:
        x_train = x_train/255.0
    
    if label != None: # one designated label
        label_idx = np.ndarray.flatten(y_train == label)
        x_train = x_train[label_idx]
        
    try:
        label_idx = np.random.permutation(x_train.shape[0])[:N]
    except:
         print('WARNING!: Exceeded the maximum number(=%d) of the label %d' % (x_train.shape[0], label))
         label_idx = np.random.permutation(x_train.shape[0])
    
    X = x_train[label_idx]   
    X = np.expand_dims(X, axis=3)
    
    if label != None: # one designated label
        return X
    else: 
        data = y_train[label_idx]
        X_label = generate_one_hot_encoding(N, 10, random_seed, data)
        return X, X_label
    
def import_cifar10(
    N: int,
    label, #: int,
    normalized: bool,
    random_seed: int
):
    np.random.seed(random_seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()#path="cifar10.npz")
    if normalized == True:
        x_train = x_train/255.0
    
    if label != None: # one designated label
        label_idx = np.ndarray.flatten(y_train == label)
        x_train = x_train[label_idx]
        
    try:
        label_idx = np.random.permutation(x_train.shape[0])[:N]
    except:
         print('WARNING!: Exceeded the maximum number(=%d) of the label %d' % (x_train.shape[0], label))
         label_idx = np.random.permutation(x_train.shape[0])
    
    X = x_train[label_idx]   
    
    if label != None: # one designated label
        return X
    else: 
        data = y_train[label_idx]
        X_label = generate_one_hot_encoding(N, 10, random_seed, data)
        return X, X_label
    
        
    
def generate_logistic(
    size, #: tuple[int], # (N, d)
    loc: float,
    scale: float,
    random_seed: int,
):
    np.random.seed(random_seed)
    X = np.random.logistic(loc=loc, scale=scale, size=size)

    return X
    
def embed_data(
    x,
    di: int,
    df: int,
    offset: float
):
    X = np.concatenate((offset*np.ones([x.shape[0],di]),x),axis=1)
    X=np.concatenate((X,offset*np.ones([x.shape[0],df])),axis=1)
    return X	

def generate_embedded_four_student_t(
    N:int,
    di:int,
    df:int,
    offset:float,
    dist:float,
    nu:float,
    random_seed:int
):
    X = generate_four_student_t(size=(N,2), dist=dist, nu=nu, random_seed=random_seed) 
    X = embed_data(X, di=di, df=df, offset=offset)# target
    return X
    
def generate_embedded_four_gaussians(
    N:int,
    di:int,
    df:int,
    offset:float,
    dist:float,
    std:float,
    random_seed:int
):
    X = generate_four_gaussians(size=(N,2), dist=dist, std=std, random_seed=random_seed) 
    X = embed_data(X, di=di, df=df, offset=offset)# target
    return X
    
    
# -------------------------------------------------------------
def generate_data(p):
    # Input
    # p: parameters dictionary
    # Outputs
    # X_, Y_, [X_label, Y_label]
    # p
    if p.N_dim == None:
        p.N_dim = 2
    
    if p.dataset == 'Learning_gaussian':
        from util.generate_data import generate_gaussian
        p.expname = p.expname+'_%.2f' % p.sigma_Q        
        X_ = generate_gaussian(size=(p.N_samples_Q, p.N_dim), m=0.0, std=p.sigma_Q, random_seed=p.random_seed) # target
        Y_ = generate_gaussian(size=(p.N_samples_P, p.N_dim), m=10.0, std=1.0, random_seed=p.random_seed+100) # initial
        
    elif p.dataset == 'Mixture_of_gaussians':
        from util.generate_data import generate_gaussian, generate_four_gaussians
        p.expname = p.expname+'_%.2f' % p.sigma_Q
        X_ = generate_four_gaussians(size=(p.N_samples_Q, p.N_dim), dist=4.0, std=p.sigma_Q, random_seed=p.random_seed) # target
        Y_ = generate_gaussian(size=(p.N_samples_P, p.N_dim), m=4.0, std=0.5, random_seed=p.random_seed+100) # initial
     
    elif p.dataset == 'Stretched_exponential':
        from util.generate_data import generate_stretched_exponential, generate_gaussian
        p.expname = p.expname+'_%.2f' % p.beta    
        X_ = generate_stretched_exponential(size=(p.N_samples_Q, p.N_dim), beta=p.beta, random_seed=p.random_seed) # target
        Y_ = generate_gaussian(size=(p.N_samples_P, p.N_dim), m=10.0, std=1.0, random_seed=p.random_seed+100) # initial

    elif p.dataset == 'Learning_student_t':
        from util.generate_data import generate_student_t, generate_gaussian
        p.expname = p.expname+'_%.2f' % p.nu    
        X_ = generate_student_t(size=(p.N_samples_Q, p.N_dim), m=0.0, nu=p.nu, random_seed=p.random_seed) # target
        Y_ = generate_gaussian(size=(p.N_samples_P, p.N_dim), m=10.0, std=1.0, random_seed=p.random_seed+100) # initial

    elif p.dataset == 'Mixture_of_student_t':
        from util.generate_data import generate_gaussian, generate_four_student_t
        p.expname = p.expname+'_%.2f' % p.nu    
        X_ = generate_four_student_t(size=(p.N_samples_Q, p.N_dim), dist=10.0, nu=p.nu, random_seed=p.random_seed) # target
        Y_ = generate_gaussian(size=(p.N_samples_P, p.N_dim), m=10.0, std=1.0, random_seed=p.random_seed+100) # initial
        
    elif 'Mixture_of_student_t_submnfld' in p.dataset:
        from util.generate_data import generate_gaussian, generate_embedded_four_student_t
        p.expname = p.expname+'_%.2f' % p.nu   
        di=5
        X_ = generate_embedded_four_student_t(N=p.N_samples_Q, di=di, df=p.N_dim-2-di,offset=0.0, dist=10.0, nu=p.nu, random_seed=p.random_seed) # target
        #m = (5,5,5,5,5, 15,15, 5,5,5,5,5)
        N_dim = p.N_dim
        if 'ae' in p.dataset:
            p.expname = p.expname+'_ae%d' % p.N_latent_dim
            N_dim = p.N_latent_dim
        Y_ = generate_gaussian(size=(p.N_samples_P, N_dim), m=15, std=1.0, random_seed=p.random_seed+100) # initial
        
    elif 'Mixture_of_gaussians_submnfld' in p.dataset:
        from util.generate_data import generate_gaussian, generate_embedded_four_gaussians
        p.expname = p.expname+'_%.2f' % p.sigma_Q    
        di=5
        X_ = generate_embedded_four_gaussians(N=p.N_samples_Q, di=5, df=p.N_dim-2-di,offset=0.0, dist=4.0, std=p.sigma_Q, random_seed=p.random_seed) # target
        N_dim = p.N_dim
        if 'ae' in p.dataset:
            p.expname = p.expname+'_ae%d' % p.N_latent_dim
            if p.sample_latent == True:
                N_dim = p.N_latent_dim
        Y_ = generate_gaussian(size=(p.N_samples_P, N_dim), m=8, std=0.5, random_seed=p.random_seed+100) # initial
        
    elif 'MNIST' in p.dataset:
        from util.generate_data import generate_logistic, import_mnist
        p.N_dim = [28,28,1]
        
        if 'switch' in p.dataset: # change one label to another
            p.expname = p.expname+'_%02d_%02d' % (p.label[0], p.label[1])
            X_ = import_mnist(N=p.N_samples_Q, label=p.label[1], normalized=True, random_seed=p.random_seed) # target
            
        elif p.label == None: # put all labels
            p.expname = p.expname+'_all' 
            X_, X_label = import_mnist(N=p.N_samples_Q, label=p.label, normalized=True, random_seed=p.random_seed) # target
            Y_label = X_label
            p.data_label = Y_label
        else: # random distribution to one designated label
            p.expname = p.expname+'_%02d' % p.label[0]
            X_ = import_mnist(N=p.N_samples_Q, label=p.label, normalized=True, random_seed=p.random_seed) # target
        
        N_dim = p.N_dim
        if 'ae' in p.dataset:  # transport in a latent space
            p.expname = p.expname+'_ae%d' % p.N_latent_dim 
            if p.sample_latent == True:
                if type(p.N_latent_dim) != list:
                    N_dim = [p.N_latent_dim]
                else:
                    N_dim = p.N_latent_dim
        if 'switch' in p.dataset: # change one label to another 
            Y_ = import_mnist(N=p.N_samples_Q, label=p.label[0], normalized=True, random_seed=p.random_seed) # initial
        else:
            Y_ = generate_logistic(size=tuple([p.N_samples_P]+N_dim), loc=0.0, scale=1.0, random_seed=p.random_seed+100) # initial
            
    elif 'CIFAR10' in p.dataset:
        from util.generate_data import generate_logistic, import_cifar10
        p.N_dim = [32,32,3]
        
        if 'switch' in p.dataset: # change one label to another
            p.expname = p.expname+'_%02d_%02d' % (p.label[0], p.label[1])
            X_ = import_cifar10(N=p.N_samples_Q, label=p.label[1], normalized=True, random_seed=p.random_seed) # target
        elif p.label == None: # put all labels
            p.expname = p.expname+'_all' 
            X_, X_label = import_cifar10(N=p.N_samples_Q, label=p.label, normalized=True, random_seed=p.random_seed) # target
            Y_label = X_label
            p.data_label = Y_label
        else: # random distribution to one designated label
            p.expname = p.expname+'_%02d' % p.label[0]
            X_ = import_cifar10(N=p.N_samples_Q, label=p.label, normalized=True, random_seed=p.random_seed) # target
        
        N_dim = p.N_dim
        if 'ae' in p.dataset:  # transport in a latent space
            p.expname = p.expname+'_ae%d' % p.N_latent_dim 
            if p.sample_latent == True:
                if type(p.N_latent_dim) != list:
                    N_dim = [p.N_latent_dim]
                else:
                    N_dim = p.N_latent_dim
        if 'switch' in p.dataset: # change one label to another 
            Y_ = import_cifar10(N=p.N_samples_Q, label=p.label[0], normalized=True, random_seed=p.random_seed) # initial
        else:
            Y_ = generate_logistic(size=tuple([p.N_samples_P]+N_dim), loc=0.0, scale=1.0, random_seed=p.random_seed+100) # initial     

    elif p.dataset in ['BreastCancer',] :
        from numpy import genfromtxt
        p.expname = p.expname+'_dim%d' % (p.N_dim ) 
        X_ = genfromtxt('gene_expression_example/GPL570/%s/target_norm_dataset_dim_%d.csv' % (p.dataset, p.N_dim), delimiter=',')
        Y_ = genfromtxt('gene_expression_example/GPL570/%s/source_norm_dataset_dim_%d.csv' % (p.dataset, p.N_dim), delimiter=',')
        
        p.N_samples_Q, p.N_samples_P = len(X_), len(Y_)
    
    elif p.dataset == '1D_pts':
        p.N_dim = 1
        p.expname = p.expname+'_P%s_Q%s' % ( '-'.join([str(x) for x in p.pts_P]), '-'.join([str(x) for x in p.pts_Q]) )
        X_ = np.array(np.reshape(p.pts_Q, (-1,1)))
        Y_ = np.array(np.reshape(p.pts_P, (-1,1)))
        
        p.N_samples_Q = len(p.pts_Q)
        p.N_samples_P = len(p.pts_P)
        
    elif p.dataset == '2D_pts':
        p.N_dim = 2
        X_ = np.empty((len(p.pts_Q),2))
        Y_ = np.empty((len(p.pts_P),2))
        list_pts_Q, list_pts_P = [], []
        for i, (x, y) in enumerate(zip(p.pts_Q, p.pts_Q_2)):
            X_[i, 0] = x
            X_[i, 1] = y
            list_pts_Q.append(str(x)+","+str(y))
        for i, (x, y) in enumerate(zip(p.pts_P, p.pts_P_2)):
            Y_[i, 0] = x
            Y_[i, 1] = y
            list_pts_P.append(str(x)+","+str(y))
        p.expname = p.expname+'_P%s_Q%s' % ( '-'.join([str(x) for x in list_pts_P]), '-'.join([str(x) for x in list_pts_Q]) )
        
        p.N_samples_Q = len(p.pts_Q)
        p.N_samples_P = len(p.pts_P)
        
    elif p.dataset == '1D_dirac2gaussian':
        from util.generate_data import generate_gaussian
        p.N_dim = 1   
        p.expname = p.expname+'_%.2f' % p.sigma_Q        
        X_ = generate_gaussian(size=(p.N_samples_Q, p.N_dim), m=0.0, std=p.sigma_Q, random_seed=p.random_seed) # target
        if p.sigma_P:
            Y_ = generate_gaussian(size=(p.N_samples_P, p.N_dim), m=0.0, std=p.sigma_P, random_seed=p.random_seed+100) # initial
        else:
            Y_ = generate_gaussian(size=(p.N_samples_P, p.N_dim), m=0.0, std=0.01, random_seed=p.random_seed+100) # initial
            
    elif p.dataset == '1D_dirac2uniform':
        from util.generate_data import generate_gaussian, generate_uniform
        p.N_dim = 1
        p.expname = p.expname+'_%.2f' % p.interval_length       
        X_ = generate_uniform(size=(p.N_samples_Q, p.N_dim), shift=0.0, l=p.interval_length, random_seed=p.random_seed) # target
        if p.sigma_P:
            Y_ = generate_gaussian(size=(p.N_samples_P, p.N_dim), m=0.0, std=p.sigma_P, random_seed=p.random_seed+100) # initial
        else:
            Y_ = generate_gaussian(size=(p.N_samples_P, p.N_dim), m=0.0, std=0.01, random_seed=p.random_seed+100) # initial
    
    if 'X_label' not in locals():
        X_label, Y_label = None, None
    return p, X_, Y_, X_label, Y_label
