import numpy as np
import tensorflow as tf
    
def generate_one_hot_encoding(
    N: int,
    N_classes: int,
    data=[]
):
    if data == []:
        return None
        #np.random.seed(random_seed)
        #data = np.random.randint(N_classes, size=N)
        
    data = np.ndarray.flatten(data)
    X_label = np.zeros((data.size, data.max()+1))
    X_label[np.arange(data.size),data] = 1
    
    return X_label
        
def import_gene_expression_data(
    N_dim: int,
    dataset, #: str,
    N_conditions: int,
):
    from numpy import genfromtxt
    
    current_filename = __file__
    data_dir = current_filename.split('gene_expression_example.py')[0]
    print(data_dir)
    
    ## test not normalized data
    X_ = genfromtxt(data_dir + '/gene_expression_example/GPL570/%s/target_norm_dataset_dim_%d.csv' % (dataset, N_dim), delimiter=',')
    Y_ = genfromtxt(data_dir + '/gene_expression_example/GPL570/%s/source_norm_dataset_dim_%d.csv' % (dataset, N_dim), delimiter=',')
    #X_ = genfromtxt(data_dir + '/gene_expression_example/GPL570/%s/target_dataset_dim_%d.csv' % (dataset, N_dim), delimiter=',')
    #Y_ = genfromtxt(data_dir + '/gene_expression_example/GPL570/%s/source_dataset_dim_%d.csv' % (dataset, N_dim), delimiter=',')
    
    if N_conditions == 1:
        X_data, Y_data = [], []
    else:
        X_data = genfromtxt(data_dir + '/gene_expression_example/GPL570/%s/target_label.csv' % dataset, delimiter=',', dtype=np.int32)
        Y_data = genfromtxt(data_dir + '/gene_expression_example/GPL570/%s/source_label.csv' % dataset, delimiter=',', dtype=np.int32)
        
    #print(sum(X_data), len(X_data)-sum(X_data), sum(Y_data), len(Y_data)-sum(Y_data))
        
    X_label = generate_one_hot_encoding(N=len(X_data), N_classes=N_conditions, data=X_data)
    Y_label = generate_one_hot_encoding(N=len(Y_data), N_classes=N_conditions, data=Y_data)
    
    return X_, X_label, Y_, Y_label

