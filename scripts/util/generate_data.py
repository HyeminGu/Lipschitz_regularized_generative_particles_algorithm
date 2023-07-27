import numpy as np
import tensorflow as tf
import sys

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


def generate_data(param):
    # Input
    # param: parameters dictionary
    # Outputs
    # X_, Y_, [X_label, Y_label]
    # param
    if param['N_dim'] == None:
        param['N_dim'] = 2
        
    sys.path.append('../')
    
    if param['dataset'] == 'Learning_gaussian':
        from data.Random_samples import generate_gaussian
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_gaussian(size=(param['N_samples_Q'], param['N_dim']), m=0.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Mixture_of_gaussians':
        from data.Random_samples import generate_gaussian, generate_four_gaussians
        param['expname'] = param['expname']+'_%.2f' % param['sigma_Q']
        X_ = generate_four_gaussians(size=(param['N_samples_Q'], param['N_dim']), dist=4.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
    
    elif param['dataset'] == 'Learning_student_t':
        from data.Random_samples import generate_student_t, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['nu']
        X_ = generate_student_t(size=(param['N_samples_Q'], param['N_dim']), m=0.0, nu=param['nu'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Learning_from_student_t':
        from data.Random_samples import generate_student_t, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['nu']
        Y_ = generate_student_t(size=(param['N_samples_P'], param['N_dim']), m=0.0, nu=param['nu'], random_seed=param['random_seed']+100) # initial
        X_ = generate_gaussian(size=(param['N_samples_Q'], param['N_dim']), m=10.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target

    elif param['dataset'] == 'Stretched_exponential':
        from data.Random_samples import generate_stretched_exponential, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['beta']
        X_ = generate_stretched_exponential(size=(param['N_samples_Q'], param['N_dim']), beta=param['beta'], random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=10.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == 'Learning_from_Stretched_exponential':
        from data.Random_samples import generate_stretched_exponential, generate_gaussian
        param['expname'] = param['expname'] +'_%.2f' % param['beta']
        Y_ = generate_stretched_exponential(size=(param['N_samples_P'], param['N_dim']), beta=param['beta'], random_seed=param['random_seed']+100) # initial
        X_ = generate_gaussian(size=(param['N_samples_Q'], param['N_dim']), m=10.0, std=param['sigma_Q'], random_seed=param['random_seed']) # target
    
        
    elif 'MNIST' in param['dataset']:
        from data.Random_samples import generate_logistic
        from data.MNIST import import_mnist, generate_one_hot_encoding
        param['N_dim'] = [28,28,1]
        
        if 'switch' in param['dataset']: # change one label to another
            param['expname'] = param['expname']+'_%02d_%02d' % (param['label'][0], param['label'][1])
            X_ = import_mnist(N=param['N_samples_Q'], label=param['label'][1], normalized=True, random_seed=param['random_seed']) # target
            
        elif param['label'] == None: # put all labels
            if param['N_conditions'] > 1:
                param['expname'] = param['expname']+'_cond'
            else:
                param['expname'] = param['expname']+'_uncond'
            X_, X_label = import_mnist(N=param['N_samples_Q'], label=param['label'], normalized=True, random_seed=param['random_seed']) # target
            if param['N_samples_P'] == param['N_samples_Q']:
                Y_label = X_label
            else:
                prop = [np.sum(X_label[:, j], keepdims=False)/param['N_samples_Q'] for j in range(np.shape(X_label)[1])]
                data = np.random.choice(np.shape(X_label)[1], param['N_samples_P'], p = prop)
                Y_label = generate_one_hot_encoding(param['N_samples_P'], np.shape(X_label)[1], param['random_seed'], data)
            param['Y_label'] = Y_label
            param['X_label'] = X_label
        else: # random distribution to one designated label
            param['expname'] = param['expname']+'_%02d' % param['label'][0]
            X_ = import_mnist(N=param['N_samples_Q'], label=param['label'], normalized=True, random_seed=param['random_seed']) # target
        N_dim = param['N_dim']
        if 'ae' in param['dataset']:  # transport in a latent space
            param['expname'] = param['expname']+'_ae%d' % param['N_latent_dim']
            if param['sample_latent'] == True:
                if type(param['N_latent_dim']) != list:
                    N_dim = [param['N_latent_dim']]
                else:
                    N_dim = param['N_latent_dim']
        if 'switch' in param['dataset']: # change one label to another
            Y_ = import_mnist(N=param['N_samples_Q'], label=param['label'][0], normalized=True, random_seed=param['random_seed']) # initial
        else:
            if param['N_project_dim'] == None:
                Y_ = generate_logistic(size=tuple([param['N_samples_P']]+N_dim), loc=0.0, scale=1.0, random_seed=param['random_seed']+100) # initial
            else:
                Y_ = projected_logistic(size=tuple([param['N_samples_P']]+N_dim), loc=0.0, scale=1.0, random_seed=param['random_seed']+100, proj_dim=param['N_project_dim']) # initial
                
    elif 'CIFAR10' in param['dataset']:
        from data.Random_samples import generate_logistic
        from data.CIFAR10 import import_cifar10, generate_one_hot_encoding
        param['N_dim'] = [32,32,3]
        
        if 'switch' in param['dataset']: # change one label to another
            param['expname'] = param['expname']+'_%02d_%02d' % (param['label'][0], param['label[1]'])
            X_ = import_cifar10(N=param['N_samples_Q'], label=param['label'][1], normalized=True, random_seed=param['random_seed']) # target
        elif param['label'] == None: # put all labels
            if param['N_conditions'] > 1:
                param['expname'] = param['expname']+'_cond'
            else:
                param['expname'] = param['expname']+'_uncond'
            X_, X_label = import_cifar10(N=param['N_samples_Q'], label=param['label'], normalized=True, random_seed=param['random_seed']) # target
            if param['N_samples_P'] == param['N_samples_Q']:
                Y_label = X_label
            else:
                prop = [np.sum(X_label[:, j], keepdims=False)/param['N_samples_Q'] for j in range(np.shape(X_label)[1])]
                data = np.random.choice(np.shape(X_label)[1], param['N_samples_P'], p = prop)
                Y_label = generate_one_hot_encoding(param['N_samples_P'], np.shape(X_label)[1], param['random_seed'], data)
            param['Y_label'] = Y_label
            param['X_label'] = X_label
        else: # random distribution to one designated label
            param['expname'] = param['expname']+'_%02d' % param['label'][0]
            X_ = import_cifar10(N=param['N_samples_Q'], label=param['label'], normalized=True, random_seed=param['random_seed']) # target
        
        N_dim = param['N_dim']
        if 'ae' in param['dataset']:  # transport in a latent space
            param['expname'] = param['expname']+'_ae%d' % param['N_latent_dim']
            if param['sample_latent'] == True:
                if type(param['N_latent_dim']) != list:
                    N_dim = [param['N_latent_dim']]
                else:
                    N_dim = param['N_latent_dim']
        if 'switch' in param['dataset']: # change one label to another
            Y_ = import_cifar10(N=param['N_samples_Q'], label=param['label'][0], normalized=True, random_seed=param['random_seed']) # initial
        else:
            if param['N_project_dim'] == None:
                Y_ = generate_logistic(size=tuple([param['N_samples_P']]+N_dim), loc=0.0, scale=1.0, random_seed=param['random_seed']+100) # initial
            else:
                Y_ = projected_logistic(size=tuple([param['N_samples_P']]+N_dim), loc=0.0, scale=1.0, random_seed=param['random_seed']+100, proj_dim=param['N_project_dim']) # initial
                
    elif param['dataset'] in ['Labeled_disease'] :
        from data.gene_expression_example import import_gene_expression_data
        
        param['expname'] = param['expname']+'_dim%d' % (param['N_dim'] )
        
        X_, X_label, Y_, Y_label = import_gene_expression_data(N_dim=param['N_dim'], dataset=param['dataset'], N_conditions=param['N_conditions'])
        
        if 'positive' in param['exp_no']:
            X_ = X_[X_label[:,1]==1]
            Y_ = Y_[Y_label[:,1]==1]
            X_label, Y_label = None, None
            param['N_conditions'] = 1
        elif 'negative' in param['exp_no']:
            X_ = X_[X_label[:,0]==1]
            Y_ = Y_[Y_label[:,0]==1]
            X_label, Y_label = None, None
            param['N_conditions'] = 1
        
        if param['N_conditions'] > 1:
            param['expname'] = param['expname']+'_cond'
        else:
            param['expname'] = param['expname']+'_uncond'
            
        param['N_samples_Q'], param['N_samples_P'] = len(X_), len(Y_)
    
    elif param['dataset'] == 'Sierpinski_carpet':
        from data.Sierpinski_carpet import generate_sierpinski
        from data.Random_samples import generate_gaussian
        
        X_ = generate_sierpinski(size=(param['N_samples_Q'], param['N_dim']), steps = 4, scale = 10, random_seed=param['random_seed']) # target
        param['N_samples_Q'] = X_.shape[0]
        param['N_samples_P'] = param['N_samples_Q']
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=3.0, random_seed=param['random_seed']+100) # initial
        
    elif param['dataset'] == '3D_Sierpinski_carpet':
        from data.Sierpinski_carpet import generate_embedded_sierpinski
        from data.Random_samples import generate_gaussian
        
        di = 0
        X_ = generate_embedded_sierpinski(N=param['N_samples_Q'], di=di, df=param['N_dim']-2-di,offset=0.0, random_seed=param['random_seed']) # target
        #X_ = generate_sierpinski(size=(param['N_samples_Q'], param['N_dim']), steps = 4, scale = 10, random_seed=param['random_seed']) # target
        param['N_samples_Q'] = X_.shape[0]
        param['N_samples_P'] = param['N_samples_Q']
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=3.0, random_seed=param['random_seed']+100) # initial
        
        
    elif param['dataset'] == '3D_Swiss_roll':
        from data.Random_samples import generate_swiss_roll, generate_gaussian
        
        X_ = generate_swiss_roll(size=(param['N_samples_Q'], param['N_dim']), random_seed=param['random_seed']) # target
        Y_ = generate_gaussian(size=(param['N_samples_P'], param['N_dim']), m=0.0, std=param['sigma_P'], random_seed=param['random_seed']+100) # initial
        
        
    if param['mb_size_P'] > param['N_samples_P']:
        param['mb_size_P'] = param['N_samples_P']
    if param['mb_size_Q'] > param['N_samples_Q']:
        param['mb_size_Q'] = param['N_samples_Q']
    
    if 'X_label' not in locals():
        X_label = None
    if 'Y_label' not in locals():
        Y_label = None
    return param, X_, Y_, X_label, Y_label
