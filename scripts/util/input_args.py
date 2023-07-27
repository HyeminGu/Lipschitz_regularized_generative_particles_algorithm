import argparse
    
def input_params():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument(
        '-N_Q', '--N_samples_Q', type=int, help='total number of target samples', # default=200,
    )
    parser.add_argument(
        '-N_P', '--N_samples_P', type=int, help='total number of prior samples', #default=200,
    )
    parser.add_argument(
        '-N_dim', type=int, help='dimension of input data',
    )
    parser.add_argument(
        '-N_project_dim', type=int, help='dimension of PCA projected space on input',
    )
    parser.add_argument(
        '-sample_latent', type=bool, help='True: sample in the latent space, False: sample in the physical space', #default = False,
    )
    # Dataset property
    parser.add_argument(
        '--dataset', type=str, #choices=['Learning_gaussian', 'Mixture_of_gaussians', 'Mixture_of_gaussians2','Mixture_of_gaussians3','Mixture_of_gaussians4', 'Stretched_exponential', 'Learning_student_t', 'Mixture_of_student_t', 'Mixture_of_student_t_submnfld', 'Mixture_of_gaussians_submnfld','MNIST', 'CIFAR10', 'MNIST_switch', 'CIFAR10_switch', 'MNIST_ae', 'MNIST_ae_switch','CIFAR10_ae',  'Mixture_of_gaussians_submnfld_ae','BreastCancer', '1D_pts', '2D_pts','1D_dirac2gaussian', '1D_dirac2uniform','Lorenz63'], #default='Learning_gaussian',
    )
    parser.add_argument(
        '-beta', type=float, help='gibbs distribution of -|x|^\beta',
    )
    parser.add_argument(
        '-sigma_P', type=float, help='std of initial gaussian distribution',
    )
    parser.add_argument(
        '-sigma_Q', type=float, help='std of target gaussian distribution',
    )
    parser.add_argument(
        '-nu', type=float, help='df of target student-t distribution',
    )
    parser.add_argument(
        '-interval_length', type=float, help='interval length of the uniform distribution',
    )
    parser.add_argument(
        '-label', type=int, nargs="+", help='class label of image data',
    )
    parser.add_argument(
        '-pts_P', type=float, nargs="+", #default=[10.0,]
    )
    parser.add_argument(
        '-pts_Q', type=float, nargs="+", #default=[0.0,]
    )
    parser.add_argument(
        '-pts_P_2', type=float, nargs="+", #default=[0.0,]
    )
    parser.add_argument(
        '-pts_Q_2', type=float, nargs="+", #default=[0.0,]
    )
    parser.add_argument(
        '-y0', type=float, nargs="+", #default=[1.0,2.0, 2.0]
    )
    parser.add_argument(
        '--random_seed', type=int, help='random seed for data generator', #default=0,
    )
    
    
    # (f, Gamma)-divergence
    parser.add_argument(
        '--f', type=str, choices=['KL', 'alpha', 'reverse_KL', 'JS'], #default='KL',
    )
    parser.add_argument(
        '-alpha', type=float, help='parameter value for alpha divergence',
    )    
    parser.add_argument(
        '--formulation', type=str, choices=['LT', 'DV'], help='LT or DV in case of f=KL, otherwise, keep LT', #default='LT',
    )
    parser.add_argument(
        '--Gamma', type=str, choices=['Lipshitz'], #default='Lipshitz',
    )
    parser.add_argument(
        '-L', type=float, help='Lipshitz constant: default=inf w/o constraint',
    )
    parser.add_argument(
        '--reverse', type=bool,  help='True -> D(Q|P), False -> D(P|Q)', #default=False,
    )
    parser.add_argument(
        '--constraint', type=str, choices=['hard', 'soft'], #default='hard',
    )
    parser.add_argument(
        '-lamda', type=float,  help='coefficient on soft constraint', #default=100.0,
    )
    parser.add_argument(
        '--generative_model', type=str, default="GPA_NN", # choices=['GPA_NN', 'GAN'], 
    )
    
    
    # Wasserstein-1 metric
    parser.add_argument(
        '-calc_Wasserstein1', type=bool,  help='True -> check Wasserstein-1 metric', #default=False,
    )
      
    
    # Neural Network <phi>
    parser.add_argument(
        '-NN', '--NN_model', type=str,  choices=['fnn', 'cnn', 'cnn-fnn'], #default='fnn',
    )
    parser.add_argument(
        '-N_fnn_layers', type=int, nargs='+', help='list of the number of FNN hidden layer units / the number of CNN feed-forward hidden layer units',
    )
    parser.add_argument(
        '-N_cnn_layers', type=int, nargs='+', help='list of the number of CNN channels',
    )
    parser.add_argument(
        '--activation_ftn', type=str, nargs='+',  choices=['relu', 'mollified_relu_cos3','mollified_relu_poly3','mollified_relu_cos3_shift','softplus', 'leaky_relu','elu', 'bounded_relu', 'bounded_elu'], help='[0]: for the fnn/convolutional layer, [1]: for the cnn feed-forward layer, [2]: for the LAST cnn feed-forward layer', #default=['relu',],
    )
    parser.add_argument(
        '-eps', type=float,  help='Mollifier shape adjusting parameter when using mollified relu3 activations', #default = 0.5,
    )
    parser.add_argument(
        '--N_conditions', type=int, help='number of classes for the conditional setting', #default=1,
    )
    
    
    # RKHS <phi>
    parser.add_argument(
        '--kernel', type=str,  choices=['gaussian'], #default='gaussian',
    )
    parser.add_argument(
        '-bandwidth', type=float, help='gaussian kernel bandwidth', # default = 2.0,
    )
    
    
    # Discriminator training parameters
    parser.add_argument(
        '--lr_phi', type=float, help='lr for phi',# default=0.001,
    )
    parser.add_argument(
        '-ep_phi', '--epochs_phi', type=int, help='# updates for phi to find phi*',# default=3,
    )
    parser.add_argument(
        '--optimizer', type=str, choices=['sgd', 'adam',], help='optimizer for NN',# default='adam',
    )
    
    # Particles transportation parameters
    parser.add_argument(
        '-ep', '--epochs', type=int, help='# updates for P', #default=1000,
    )
    parser.add_argument(
        '--ode_solver', type=str, choices=['forward_euler', 'AB2', 'AB3', 'AB4', 'AB5', 'ABM1', 'Heun', 'ABM2', 'ABM3', 'ABM4', 'ABM5', 'RK4', 'ode45', 'Newton', 'BFGS','Gradient_Ascent'], help='ode solver for particle ode', #default='forward_euler',
    )
    parser.add_argument(
        '-mobility', type=str, help='problem dependent mobility function\nRecommendation: MNIST - bounded',
    )
    parser.add_argument(
        '-lr_P_decay', type=str, choices=['rational', 'step',], help='delta t decay',
    )
    parser.add_argument(
        '--lr_P', type=float, help='lr for P',#default=1.0,
    )
    
    parser.add_argument(
        '--exp_no', type=str, help='short experiment name under the same data', #default=0,
    )
    parser.add_argument(
        '--mb_size_P', type=int, help='mini batch size for the moving distribution P',# default=200,
    )
    parser.add_argument(
        '--mb_size_Q', type=int, help='mini batch size for the target distribution Q',# default=200,
    )
    
    
    # generated data
    parser.add_argument(
        '-generator_status', type=str, # choices=['learn', 'eval', ], #default=None,
    )
    parser.add_argument(
        '-N_generated_samples', type=int, help='number of generated samples from generator', # default=200,
    )
    parser.add_argument(
        '-resample', type=bool, help='True -> resample P0 or R for each generator learning',# default=True,
    )
    parser.add_argument(
        '-resample_latent', type=bool, help='True -> resample at the latent space', #default = False,
    )
    parser.add_argument(
        '-N_gen_latent_dim', type=int, help='dimension of latent space',
    )
    
    
    # Autoencoder generator parameters
    parser.add_argument(
        '-NN_ae_model', type=str, choices=['fnn', 'cnn',], help='Neural network for the autoencoder',
    )
    parser.add_argument(
        '-ae_model', type=str, choices=['VAE', 'AE',], help='Autoencoder model',
    )
    parser.add_argument(
        '-encoder_channel_layers', type=int, nargs='+', help='weight matrices dimensions for layers (both fnn and cnn)', # default=[8,8,8],
    )
    parser.add_argument(
        '-encoder_kernel_shape_layers', type=int, nargs='+', help='kernel height and width for layers (for cnn) dim=dim(encoder_activation_ftn)',
    )
    parser.add_argument(
        '-encoder_activation_ftn', type=str, nargs='+',  choices=['relu', 'mollified_relu_cos3','mollified_relu_poly3','mollified_relu_cos3_shift','softplus', 'leaky_relu','elu', 'bounded_relu', 'bounded_elu'], help='[0]: for the fnn/convolutional layer, [1]: for the cnn feed-forward layer, [2]: for the LAST cnn feed-forward layer', #default=['relu',],
    )
    parser.add_argument(
        '-decoder_channel_layers', type=int, nargs='+', help='weight matrices dimensions for layers (both fnn and cnn)', # default=[8,8,8],
    )
    parser.add_argument(
        '-decoder_kernel_shape_layers', type=int, nargs='+', help='kernel height and width for layers (for cnn) dim=dim(encoder_activation_ftn)',
    )
    parser.add_argument(
        '-decoder_activation_ftn', type=str, nargs='+',  choices=['relu', 'mollified_relu_cos3','mollified_relu_poly3','mollified_relu_cos3_shift','softplus', 'leaky_relu','elu', 'bounded_relu', 'bounded_elu'], help='[0]: for the fnn/convolutional layer, [1]: for the cnn feed-forward layer, [2]: for the LAST cnn feed-forward layer', #default=['relu',],
    )
    parser.add_argument(
        '-encoder_L', type=float, help='Lipshitz constant for the encoder: default=inf w/o constraint',
    )
    parser.add_argument(
        '-decoder_L', type=float, help='Lipshitz constant for the decoder: default=inf w/o constraint',
    )
    
    
    
    # Autoencoder training parameters
    parser.add_argument(
        '-epochs_ae', type=int, help='# updates for AE to find psi_0*', #default=1000,
    )
    parser.add_argument(
        '-lr_ae', type=float, help='lr for ae',#default=1.0,
    )
    parser.add_argument(
        '-optimizer_ae', type=str, choices=['sgd', 'adam',], help='optimizer for AE',# default='adam',
    )
    parser.add_argument(
        '-epochs_dec', type=int, help='# updates for AE to find psi_t*', #default=1000,
    )
    parser.add_argument(
        '-lr_dec', type=float, help='lr for dec',#default=1.0,
    )
    parser.add_argument(
        '-optimizer_dec', type=str, choices=['sgd', 'adam',], help='optimizer for decoder',# default='adam',
    )
    parser.add_argument(
        '-dec_train_iter', type=int, help='train decoder per each dec_train_iter', #default=1000,
    )
    
    
    # save/display 
    parser.add_argument(
        '--save_iter', type=int, help='save results per each save_iter',# default=10,
    )
    parser.add_argument(
        '--plot_result', type=bool, help='True -> show plots',# default=False,
    )
    parser.add_argument(
        '--plot_intermediate_result', type=bool, help='True -> save intermediate plots',# default=False,
    )
    parser.add_argument(
        '-filename', type=str, help='filename for GPA_NN-Generator',# default='adam',
    )
    
    return parser.parse_known_args()
