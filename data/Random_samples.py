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
    
def generate_two_gaussians(
    size, #: tuple[int], # (N, d)
    mus,#: tuple[float],
    std: float,
    p: float,
    random_seed: int
):
    np.random.seed(random_seed)

    ms = np.array(mus).T
    num_clusters = ms.shape[1]

    idxs = np.random.choice(num_clusters, size[0], p=p)
    means = np.array([ms[:, idx] for idx in idxs])
    X = std * np.random.standard_normal(size) + means
    
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
    
def project_by_PCA(
    X,
    proj_dim:int,
):
    from sklearn.decomposition import PCA
    
    input_shape = X.shape
    X = np.reshape(X, (input_shape[0], -1), order='C')
    pca = PCA(n_components=proj_dim, random_state=0)
    coef = pca.fit_transform(X)
    basis = pca.components_
        
    pca2 = PCA(n_components=min(X.shape[0], 784), random_state=0)
    coef2 = pca2.fit_transform(X)
    print(sum(np.cumsum(pca2.explained_variance_ratio_)<0.70))
    
    X = coef @ basis
    X = np.reshape(X, input_shape, order='C')
    
    eps = 0.01
    
    return X+np.random.normal(size=X.shape)
    
def projected_logistic(
    size, #: tuple[int], # (N, d)
    loc: float,
    scale: float,
    random_seed: int,
    proj_dim: int,
):
    X = generate_logistic(size=size, loc=loc, scale=scale, random_seed=random_seed)
    X_proj = project_by_PCA(X, proj_dim)
    
    return X_proj
    
def generate_swiss_roll(
    size, #: tuple[int], # (N, d)
    random_seed: int,
):
    from sklearn import datasets
    X, _ = datasets.make_swiss_roll(n_samples=size[0], random_state=random_seed)
    return X
    
