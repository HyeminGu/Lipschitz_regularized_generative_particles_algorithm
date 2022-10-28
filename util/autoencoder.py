import keras
from keras import layers
import tensorflow as tf
import numpy as np

# Functions to access trained model
def preprocess_input(x):
    x = x.astype('float32')
    if np.ndim(x) >2:
        if np.max(x)>1:
            x = x/255.
        x = x.reshape((x.shape[0], np.prod(x.shape[1:])))
    return x
    
def postprocess_output(x, N_dim, scale=1.0):
    if type(N_dim) != list:
        N_dim = [N_dim]
    x = scale*x.reshape([x.shape[0]]+N_dim)
    return x
    
def load_autoencoder(save_path, encoder=True):
    if encoder == True:
        filename = save_path+"_encoder"
    else:
        filename = save_path+"_decoder"
    model = keras.models.load_model(filename, compile=False)
    return model
# ------------------------------------------------------------------
# Customized loss
def KL_MSE_loss(y_true, y_pred):
    import tensorflow as tf
    squared_difference = tf.square(y_true - y_pred)
    KL = tf.keras.losses.KLDivergence()
    
    return 2*tf.math.reduce_mean(squared_difference, axis=-1)+ KL(y_true, y_pred) # Note the `axis=-1`   
# ------------------------------------------------------------------
def train_autoencoder_submnfld(dataset, epochs=50, encoding_dim = 3, save_path='util/saved_model/deep'):
    if dataset == 'Mixture_of_student_t_submnfld':
        N_dim=12
        N_samples = 10000
        try:
            from util.generate_data import generate_embedded_four_student_t
        except:
            from generate_data import generate_embedded_four_student_t
        x_train = generate_embedded_four_student_t(N=N_samples, di=5, df=N_dim-2-5,offset=0.0, dist=10.0, nu=0.5, random_seed=200)
        x_test = generate_embedded_four_student_t(N=int(N_samples*0.1), di=5, df=N_dim-2-5,offset=0.0, dist=10.0, nu=0.5, random_seed=200)
    elif dataset == 'Mixture_of_gaussians_submnfld_ae':
        N_dim=12
        N_samples = 10000
        try:
            from util.generate_data import generate_embedded_four_gaussians
        except:
            from generate_data import generate_embedded_four_gaussians
        x_train = generate_embedded_four_gaussians(N=N_samples, di=5, df=N_dim-2-5,offset=0.0, dist=4.0, std=0.5, random_seed=200)
        x_test = generate_embedded_four_gaussians(N=int(N_samples*0.1), di=5, df=N_dim-2-5,offset=0.0, dist=4.0, std=0.5, random_seed=200)
        
    #x_train = preprocess_input(x_train)
    #x_test = preprocess_input(x_test)

    input_img = keras.Input(shape=(N_dim,))
    if "swallow" in save_path:
        encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
        
        decoded = layers.Dense(N_dim, activation='selu')(encoded)
    else:
        encoded = layers.Dense(int(N_dim/2), activation='selu')(input_img)
        encoded = layers.Dense(encoding_dim, activation='selu')(encoded)
        
        decoded = layers.Dense(int(N_dim/2), activation='selu')(encoded)
        decoded = layers.Dense(N_dim, activation='selu')(encoded)   

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    
    # This model maps an encoded representation to its reconstruction
    encoded_input = keras.Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss=KL_MSE_loss)#keras.losses.MeanSquaredError())##keras.losses.KLDivergence())#
    autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=min(200, N_samples),
                shuffle=True,
                validation_data=(x_test, x_test))
                
    # plot result
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    
    import matplotlib.pyplot as plt
    plt.scatter(decoded_imgs[:,5], decoded_imgs[:,6], label="decoded samples")
    plt.scatter(x_test[:,5], x_test[:,6], label="input samples")
    plt.legend()
    plt.show()
    
    from scipy.stats import gaussian_kde
    
    orth_axes = np.array((0,1,2,3,4,7,8,9,10,11))
    complement_data = decoded_imgs[:,orth_axes]
    
    complement_data = complement_data.flatten()
    
    x= np.linspace(complement_data.min(), complement_data.max(), 1000)
    z = gaussian_kde(complement_data)(x)
    plt.plot(x, z, linestyle='-')  
    plt.tight_layout()
    plt.show()

    plt.scatter(encoded_imgs[:,0], encoded_imgs[:,1], label="encoded axis 1, 2")
    if np.shape(encoded_imgs)[1] >=3:
        plt.scatter(encoded_imgs[:,0], encoded_imgs[:,2], label="encoded axis 1, 3")
        plt.scatter(encoded_imgs[:,1], encoded_imgs[:,2], label="encoded axis 2, 3")
        if np.shape(encoded_imgs)[1] >=4:
            plt.scatter(encoded_imgs[:,3], encoded_imgs[:,0], label="encoded axis 4, 1")
            plt.scatter(encoded_imgs[:,3], encoded_imgs[:,1], label="encoded axis 4, 2")
            plt.scatter(encoded_imgs[:,3], encoded_imgs[:,2], label="encoded axis 4, 3")
    #plt.scatter(x_test[:,5], x_test[:,6], label="input samples")
    plt.legend()
    plt.show()
                
    encoder.save(save_path+"_encoder")
    decoder.save(save_path+"_decoder")
                  
def train_autoencoder_image(dataset, epochs=50, encoding_dim = 256, save_path='util/saved_model/deep'):
    if 'MNIST' in dataset:
        import keras.datasets.mnist as ds
        data_shape = (28,28,1)
    elif 'CIFAR10' in dataset:
        import keras.datasets.cifar10 as ds
        data_shape = (32,32,3)
    
    (x_train, x_train_label), (x_test, x_test_label) = ds.load_data()
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)
    
    x_train_label = tf.keras.utils.to_categorical(x_train_label)
    x_test_label = tf.keras.utils.to_categorical(x_test_label)
    
    input_dim = np.shape(x_train)[1]
    
    input_img = keras.Input(shape=(input_dim,))
    if "swallow" in save_path:
        encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
        
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    elif 'MNIST' in dataset:
        encoded = layers.Dense(256, activation='relu')(input_img)
        encoded = layers.Dense(128, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        decoded = layers.Dense(128, activation='sigmoid')(encoded)
        decoded = layers.Dense(256, activation='sigmoid')(decoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)   
    elif 'CIFAR10' in dataset:
        encoded = layers.Dense(1024, activation='relu')(input_img)
        encoded = layers.Dense(512, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        decoded = layers.Dense(512, activation='sigmoid')(encoded)
        decoded = layers.Dense(1024, activation='sigmoid')(decoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)   

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    
    # This model maps an encoded representation to its reconstruction
    encoded_input = keras.Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, x_test))
                
    # plot result
    import matplotlib.pyplot as plt
    
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(data_shape))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(data_shape))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
                
    encoder.save(save_path+"_encoder")
    decoder.save(save_path+"_decoder")                  

    
    
    
if __name__ == "__main__":
    import sys
    try:
        name = sys.argv[1]
    except:
        name = "deep_submnfld_4d"
    try:
        encoding_dim = int(sys.argv[2])
    except:
        encoding_dim=3
        
    if 'MNIST' in name:
        train_autoencoder_image(dataset = name, epochs=100, encoding_dim = encoding_dim, save_path='util/saved_model/'+name)
    elif 'CIFAR10' in name:
        train_autoencoder_image(dataset = name, epochs=100, encoding_dim = encoding_dim, save_path='util/saved_model/'+name)
    elif 'submnfld' in name:
        train_autoencoder_submnfld(dataset="Mixture_of_gaussians_submnfld_ae", epochs=100, encoding_dim = encoding_dim, save_path='util/saved_model/'+name)
