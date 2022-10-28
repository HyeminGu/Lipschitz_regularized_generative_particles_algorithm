import argparse
    
def input_params():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument(
        '--dataset', type=str, default='Learning_gaussian', choices=['Learning_gaussian', 'Mixture_of_gaussians', 'Stretched_exponential', 'Learning_student_t', 'Mixture_of_student_t', 'Mixture_of_student_t_submnfld', 'Mixture_of_gaussians_submnfld','MNIST', 'CIFAR10', 'MNIST_switch', 'CIFAR10_switch', 'MNIST_ae', 'MNIST_ae_switch','CIFAR10_ae',  'Mixture_of_gaussians_submnfld_ae','BreastCancer', '1D_pts', '2D_pts','1D_dirac2gaussian', '1D_dirac2uniform',]
    )
    parser.add_argument(
        '-N_Q', '--N_samples_Q', type=int, default=200, help='total number of samples for target',
    )
    parser.add_argument(
        '-N_P', '--N_samples_P', type=int, default=200, help='total number of samples for fake data',
    )
    parser.add_argument(
        '-N_dim', type=int, help='dimension of input data',
    )
    parser.add_argument(
        '-N_latent_dim', type=int, help='dimension of latent space',
    )
    parser.add_argument(
        '-sample_latent', type=bool, default = False, help='True: sample in the latent space, False: sample in the physical space',
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
        '-pts_P', type=float, nargs="+", default=[10.0,]
    )
    parser.add_argument(
        '-pts_Q', type=float, nargs="+", default=[0.0,]
    )
    parser.add_argument(
        '-pts_P_2', type=float, nargs="+", default=[0.0,]
    )
    parser.add_argument(
        '-pts_Q_2', type=float, nargs="+", default=[0.0,]
    )
    parser.add_argument(
        '--random_seed', type=int, default=0, help='#random seed for data generator',
    )
    
    
    # (f, Gamma)-divergence
    parser.add_argument(
        '--f', type=str, default='KL', choices=['KL', 'alpha'],
    )
    parser.add_argument(
        '-alpha', type=float, help='value for alpha',
    )    
    parser.add_argument(
        '--formulation', type=str, default='LT', help='LT# or DV',
    )
    parser.add_argument(
        '--Gamma', type=str, default='Lipshitz', choices=['Lipshitz'],
    )
    parser.add_argument(
        '-L', type=float, help='Lipshitz constant: default=inf w/o constraint',
    )
    parser.add_argument(
        '--reverse', type=bool, default=False, help='True -> D(Q|P), False -> D(P|Q)',
    )
    parser.add_argument(
        '--constraint', type=str, default='hard', choices=['hard', 'soft'],
    )
    parser.add_argument(
        '-lamda', type=float, help='soft constraint constant',
    )
      
    
    # Neural Network definition <phi>
    parser.add_argument(
        '-NN', '--NN_model', type=str, default='fnn', choices=['fnn', 'cnn-fnn'],
    )
    parser.add_argument(
        '-N_fnn_layers', type=int, nargs='+', help='list of the number of FNN hidden layer units / the number of CNN feed-forward hidden layer units',
    )
    parser.add_argument(
        '-N_cnn_layers', type=int, nargs='+', help='list of the number of CNN channels',
    )
    parser.add_argument(
        '--activation_ftn', type=str, nargs='+', default=['relu',], choices=['relu', 'softplus', 'leaky_relu','elu', 'bounded_relu', 'bounded_elu'], help='[0]: activation function of the fnn/cnn that imposes Lipschitz continuity through spectral normalization, [1]: activation function of the cnn dense layer',
    )
    parser.add_argument(
        '--N_conditions', type=int, default=1, help='number of classes for the conditional setting',
    )
    
    # Mobility
    parser.add_argument(
        '--mobility', type=str, default='fixed', choices=['fixed', 'ML', 'potential',], help='assign mobility to adjust speed for each particle; ML: learned to minimize the loss (divergence), potential: use learned potential function of the current step',
    )
    parser.add_argument(
        '-mobility_L', type=float, help='Lipshitz constant for the mobility',
    )
    parser.add_argument(
        '--mobility_lamda', type=float, default=0.0, help='coefficient for mobility^2',
    )
    parser.add_argument(
        '-mobility_bound', type=float, help='mobility bound > 0',
    )
    
    # Neural Network definition <ML trained mobility>
    parser.add_argument(
        '--mobility_activation_ftn', type=str, nargs='+', default=['relu','sigmoid'], choices=['relu', 'softplus', 'leaky_relu','elu', 'bounded_relu', 'sigmoid', 'abs', 'linear'], help='',
    )
    
    
    # training parameters
    parser.add_argument(
        '-ep', '--epochs', type=int, default=1000, help='# updates for P',
    )
    parser.add_argument(
        '-ep_nn', '--epochs_nn', type=int, default=5, help='# updates for NN to find g*',
    )
    parser.add_argument(
        '-lr_P_decay', type=str, choices=['rational', 'step',], help='delta t decay',
    )
    parser.add_argument(
        '--lr_P', type=float, default=1.0, help='lr for P',
    )
    parser.add_argument(
        '--lr_NN', type=float, default=0.1, help='lr for NN',
    )
    parser.add_argument(
        '--exp_no', type=str, default=0, help='short experiment name under the same data',
    )
    parser.add_argument(
        '--mb_size_P', type=int, default=200, help='mini batch size for the moving distribution P',
    )
    parser.add_argument(
        '--mb_size_Q', type=int, default=200, help='mini batch size for the target distribution Q',
    )
    
    
    # save/display 
    parser.add_argument(
        '--save_iter', type=int, default=10, help='save results per each save_iter',
    )
    parser.add_argument(
        '--plot_result', type=bool, default=False, help='True -> show plots',
    )
    
    return parser.parse_args()
