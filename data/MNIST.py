import numpy as np
import tensorflow as tf
    
def generate_one_hot_encoding(
    N: int,
    N_classes: int,
    random_seed: int,
    data=[]
):
    if data == []:
        np.random.seed(random_seed)
        data = np.random.randint(N_classes, size=N)
        
    data = np.ndarray.flatten(data)
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
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context


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
    
