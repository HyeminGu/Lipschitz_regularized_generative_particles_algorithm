# Metrics to monitor convergence
# 1. kinetic energy
#    calc_ke(velocity_field, samplesize)
# 2. average particle speed
#    calc_grad_phi(velocity_field)
# 3. sinkhorn divergence
#    calc_sinkhorn(P, Q, reg=0.2)
# 4. FID for iamge data
#    calc_fid(pred, real, batch_size=500)

import numpy as np

def calc_ke(dP_dt, N_samples_P):
    return np.linalg.norm(dP_dt)**2/N_samples_P/2
    
def calc_grad_phi(dP_dt):
    return np.mean(np.linalg.norm(dP_dt, axis=1))
    
def calc_sinkhorn(P, Q, reg=0.2):
    from geomloss import SamplesLoss
    import torch
    N_P = P.shape[0]
    N_Q = Q.shape[0]
    
    X = torch.from_numpy(P).type(torch.float32)
    Y = torch.from_numpy(Q).type(torch.float32)
    
    return SamplesLoss(loss='sinkhorn', p=2)(X,Y).numpy()

def calc_fid(pred, real, batch_size=500):
# Frechet Inception Distance calculated on "dataset" using Inception V3
# pred: model prediction image(s) of 3D or 4D numpy array
# real: real image(s) of 3D or 4D numpy array
    from sys import platform
    from scipy.linalg import sqrtm
    
    library = "pytorch"#"tensorflow"
    if library == "pytorch":
        mu, sigma = calc_statistics_torch(pred, real, batch_size, platform)
    else: # tensorflow
        mu, sigma = calc_statistics_tensorflow(pred, real, batch_size, platform)
        
    # calculate frechet distance
    diff = mu[0]-mu[1]
    covmean = sqrtm(sigma[0] @ sigma[1])
    if np.iscomplex(covmean).any():
        #print("Convert imaginary numbers entries to real numbers")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
        
    return np.dot(diff, diff) + np.trace(sigma[0]) + np.trace(sigma[1]) - 2 * tr_covmean

    
# --------------------------------------------------------------------------------
# Frechet Inception Distance calculation for image data
# Code modified from https://github.com/biweidai/SINF/blob/master/sinf/fid_score.py
# --------------------------------------
def calc_statistics_torch(data1, data2, batch_size, platform=None):
    import torch
    from torch.nn.functional import adaptive_avg_pool2d
    try:
        from util.inception import InceptionV3
    except:
        from inception import InceptionV3
    
    # load_inception_model_and_preprocess_data_torch
    if 'darwin' in platform:
        gpu = "mps"
    elif 'linux' in platform:
        gpu = "cuda"
        batch_size = 20
    else:
        gpu = "cpu"
    device = torch.device(gpu)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

    model = InceptionV3([block_idx]).to(device)
    
    data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1)) # values lie in [0,1]
    data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2)) # values lie in [0,1]
    pred = np.transpose(data1, axes=(0,3,1,2))
    real = np.transpose(data2, axes=(0,3,1,2))
        
    # calculate means, covariances
    mu, sigma = [],[]
    for data in (pred, real):
        predicts = []
        n_batches = int(np.ceil(data.shape[0]/batch_size))
        for i in range(n_batches):
            if i < n_batches-1:
                mini_batch = data[i*batch_size: (i+1)*batch_size]
            else:
                mini_batch = data[i*batch_size:len(data)]
                
            if data.shape[-1] < 3: # gray-scale image
                mini_batch = gray2rgb_images(mini_batch)
            if (data.shape[1] < 299) or (data.shape[2] < 299):
                d_shape = (3,299,299)
                mini_batch = resize_images(mini_batch, d_shape)
                
            model.eval()
            mini_batch = torch.FloatTensor(mini_batch)
            mini_batch = mini_batch.to(device)
            with torch.no_grad():
                pred = model(mini_batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            predicts.append( pred.squeeze(3).squeeze(2).cpu().numpy() )
        predict = np.concatenate(predicts, axis=0)
        
        # flatten the array of prediction into the shape (N_Features , N_Samples)
        if predict.ndim > 2:  # cnn prediction
            predict = np.reshape(predict, (-1, data.shape[0]), order='C')
        else:  # fnn prediction
            predict = np.transpose(predict)

        mu.append(np.mean(predict, axis=1))
        sigma.append(np.cov(predict))
        
    return mu, sigma
    
def calc_statistics_tensorflow(data1, data2, batch_size, platform=None):
    import tensorflow as tf
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    
    if 'linux' in platform:
        batch_size = 20
    
    input_shape = (max((299,data1.shape[1])),max((299,data1.shape[2])),3)
    try:
        model = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False, pooling='avg')
    except: # For Mac
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False, pooling='avg')
    
    pred = preprocess_input(data1)
    real = preprocess_input(data2)
    
    device = None
    
    # calculate means, covariances
    mu, sigma = [],[]
    for data in (pred, real):
        predicts = []
        n_batches = int(np.ceil(data.shape[0]/batch_size))
        for i in range(n_batches):
            if i < n_batches-1:
                mini_batch = data[i*batch_size: (i+1)*batch_size]
            else:
                mini_batch = data[i*batch_size:len(data)]
                
            if data.shape[-1] < 3: # gray-scale image
                mini_batch = gray2rgb_images(mini_batch)
            if (data.shape[1] < 299) or (data.shape[2] < 299):
                d_shape = (299,299,3)
                mini_batch = resize_images(mini_batch, d_shape)
            
            predicts.append(model.predict(mini_batch))
        predict = np.concatenate(predicts, axis=0)
        
        # flatten the array of prediction into the shape (N_Features , N_Samples)
        if predict.ndim > 2:  # cnn prediction
            predict = np.reshape(predict, (-1, data.shape[0]), order='C')
        else:  # fnn prediction
            predict = np.transpose(predict)

        mu.append(np.mean(predict, axis=1))
        sigma.append(np.cov(predict))
        
    return mu, sigma
    
    
def gray2rgb_images(samples):
    samples = np.concatenate([samples, samples, samples], axis=-1)
    return samples
    
def resize_images(samples, new_shape):
    from skimage.transform import resize
    
    new_samples = []
    for i in range(samples.shape[0]):
        new_samples.append( resize(samples[i], new_shape, 0) )
    new_samples = np.stack(new_samples, axis=0)
    return new_samples
        
# ------------------------------------------------------------------------------
# test code
if __name__ == "__main__":
    import sys
    # python3 evaluate_metric.py calc_fid 100 200
    if sys.argv[1] == "calc_fid":
        from tensorflow.keras.datasets.mnist import load_data
        
        N_samples_pred = int(sys.argv[2])
        N_samples_real = int(sys.argv[3])
        
        (x_train, y_train), (x_test, y_test) = load_data(path="mnist.npz")
        x_train = x_train/255.0
        
        idx_pred = np.random.permutation(x_train.shape[0])[:N_samples_pred]
        idx_real = np.random.permutation(x_train.shape[0])[:N_samples_real]

        pred = np.expand_dims(x_train[idx_pred], axis=3)
        real = np.expand_dims(x_train[idx_real], axis=3)
        
        print(calc_fid(pred, real))
        
    # python3 evaluate_metric.py calc_ke 100
    elif sys.argv[1] == "calc_ke":
        N_samples_P = int(sys.argv[2])
        
        dP = np.random.random((N_samples_P, 10))
        print(calc_ke(dP, N_samples_P))
        
    # python3 evaluate_metric.py calc_grad_phi 100
    elif sys.argv[1] == "calc_grad_phi":
        N_samples_P = int(sys.argv[2])
        
        dP = np.random.random((N_samples_P, 10))
        print(calc_grad_phi(dP))
    
    # python3 evaluate_metric.py calc_sinkhorn 0.2
    elif sys.argv[1] == "calc_sinkhorn":
        reg = float(sys.argv[2])
        
        P = np.random.random((30,1))
        Q = np.random.random((40,1))
        
        print(calc_sinkhorn(P,Q, reg=reg))
