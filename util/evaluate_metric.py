import numpy as np
import tensorflow as tf

def calc_fid(pred, real, model_name):
# Frechet Inception Distance calculated on the "model_name"
# pred: model prediction image(s) of 3D or 4D numpy array
# real: real image(s) of 3D or 4D numpy array
# model_name: 'inception_v3', 'vgg16', 'autoencoder_mnist'
    
    if model_name == 'inception_v3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        input_shape = tuple(pred.shape[1:])
        model = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False)
    elif model_name == 'vgg16':
        from tensorflow.keras.applications.vgg16 import VGG16
        from tensorflow.keras.applications.vgg16 import preprocess_input
        input_shape = tuple(pred.shape[1:])
        model = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)
    elif 'autoencoder' in model_name:
        try:
            from util.autoencoder import load_autoencoder, preprocess_input
        except:
            from autoencoder import load_autoencoder, preprocess_input
        if 'mnist' in model_name:
            model = load_autoencoder(save_path='util/saved_model/FID_MNIST_deep_64', encoder = True)
        elif 'cifar10' in model_name:
            model = load_autoencoder(save_path='util/saved_model/FID_CIFAR10_deep_256', encoder = True)
        
    mu, sigma = [],[]
    for data in (pred, real):
        # values in the range [0,255] dtype=float32
        data = data.astype('float32')
        data = 255/(np.max(data)-np.min(data))*(data-np.min(data))
   
        input_img = preprocess_input(data)
        predict = model.predict(input_img)
        # flatten the array of prediction into the shape (N_Features , N_Samples)    
        if predict.ndim > 2:  # cnn prediction
            predict = np.reshape(predict, (-1, data.shape[0]), order='C')
        else: # fnn prediction
            predict = np.transpose(predict)

        mu.append(np.mean(predict, axis=1))
        sigma.append(np.cov(predict))
        
    diff = mu[0]-mu[1]
    eps = 1e-6
    sigma_pred_half = np.linalg.cholesky(sigma[0]+eps*np.eye(sigma[0].shape[0]))
    sigma_half = np.linalg.cholesky(sigma_pred_half*sigma[1]*np.transpose(sigma_pred_half)+eps*np.eye(sigma[0].shape[0]))
    tr_covmean = np.trace(sigma_half)
        
    return np.dot(diff, diff) + np.trace(sigma[0]) + np.trace(sigma[1]) - 2 * tr_covmean
    
def calc_ke(dP_dt, mobility, N_samples_P):
    return np.linalg.norm(mobility*dP_dt)**2/N_samples_P/2
    

def calc_ke_gan(dD_dSamples, dG_dDiscParams):
# calculate the expected kinetic energy of generated particles in GAN - Sobolev descent: Mroueh, Youssef et al. “Sobolev Descent.” AISTATS (2019).
# dY_dt = 1/n \sum_n \frac{\partial G(z)^t}{\partial \theta } \frac{\partial G(\tilde{z})}{\partial \theta }|_{theta=theta_N} \nabla D_N (G_{\theta_N} (\tilde{z}_n))
# dD_dSamples: numpy array of size (N_samples, 1, Y_dim) - gradient of discriminator with respect to the generated sample
# dG_dDiscParams: numpy array of size (N_samples+1, theta_dim, Y_dim) - gradient of generator with respect to the discriminator parameter 
    N_samples = len(dD_dSamples)
    if len(dD_dSamples.shape) == 2:
        dD_dSamples = np.expand_dims(dD_dSamples, axis=1)
        
    dG_dDiscParams1 = np.tile(dG_dDiscParams[-1], (N_samples,1,1))
    dG_dDiscParams = dG_dDiscParams[:-1]
    
    dY_dt = np.matmul( np.matmul(dD_dSamples, np.transpose(dG_dDiscParams, (0,2,1))), dG_dDiscParams1)
    
    return calc_ke(dY_dt, N_samples)
    
   
if __name__ == "__main__":
    # test
    import sys
    if sys.argv[1] == "calc_fid":
        pred = np.random.random((60, 28,28, 1))
        real = np.random.random((22, 28,28, 1))
        print(calc_fid(pred, real, 'autoencoder_mnist'))
    elif sys.argv[1] == "calc_ke":
        N_samples_P = 64
        dP = np.random.random((N_samples_P, 10))
        print(calc_ke(dP, N_samples_P))
    elif sys.argv[1] == "calc_ke_gan":
        dD_dSamples = np.random.random((50, 12))
        dG_dDiscParams = np.random.random((51, 1200, 12))
        print(calc_ke_gan(dD_dSamples, dG_dDiscParams))