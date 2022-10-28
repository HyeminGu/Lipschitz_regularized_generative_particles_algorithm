# Lipschitz regularized gradient flows and generative particles
This repository is the official implementation of *Lipschitz regularized gradient flows and latent generative particles* in Tensorflow 2

## Requirements
To install requirements:

-tensorflow-gpu=2.7.0 or 2.8.0
```pip install tensorflow==2.7.0```

## Dependent python libraries
pip install numpy, tensorflow, matplotlib, scipy, 
pip install umap-learn  # gene_expression_example


## Usage: run the following command to reproduce the results in the submission
--plot_result True/False : controls plotting the result after the calculation

See more parameters in util/input_args.py

## Experiments 

### 2D Student-t
* Compare KL-divergence, alpha=2-divergence, alpha=10-divergence for heavy tailed data from t(0.5) distribution in 2D
* random_seed=0 for trajectory plot
* random_seed=0,1,2,3,4 for Divergence, Lipschitz regularized FI loss plots

```
python f-Gamma-gpa.py --f KL --formulation DV -L 1.0 --dataset Learning_student_t -nu 0.5 -NN fnn -N_fnn_layers 32 32 32 --epochs_nn 5 --epochs 5000 --lr_NN 0.01 --lr_P 0.1 --exp_no 0 --random_seed $random_seed --plot_result True

python f-Gamma-gpa.py --f alpha -alpha 2 -L 1.0 --dataset Learning_student_t -nu 0.5 -NN fnn -N_fnn_layers 32 32 32 --epochs_nn 5 --epochs 5000 --lr_NN 0.01 --lr_P 0.1 --exp_no 0 --random_seed $random_seed --plot_result True

python f-Gamma-gpa.py --f alpha -alpha 10 -L 1.0 --dataset Learning_student_t -nu 0.5 -NN fnn -N_fnn_layers 32 32 32 --epochs_nn 5 --epochs 5000 --lr_NN 0.01 --lr_P 0.1 --exp_no 0 --random_seed $random_seed --plot_result True
```

### Gene expression data merging
* Influence of choosing latent dimensions
* N_dim=2,5,10,20,50,100,200
* N_fnn_layers=32 32 32		# for N_dim=2,5,10
* N_fnn_layers=64 64 64		# for N_dim=20,50,100
* N_fnn_layers=128 128 128	# for N_dim=200
* Step 1: Download geo data using gene_expression_example/create_dataset.R (already have data!)
* Step 2: Store PCA result (source_norm_dataset_dim_##.csv and target_norm_dataset_dim_##.csv) using gene_expression_example/pca_dim_reduction_normalized.py (already have data!)
* Step 3: Run f-Gamma-gpa2.py as below
* Step 4: Decode the output using gene_expression_example/back_mapping_normalized.py

python f-Gamma-gpa2.py --f KL --formulation DV -L 1  --dataset BreastCancer -NN fnn -N_fnn_layers $N_fnn_layers --activation_ftn relu --epochs_nn 10 --epochs 5000 --lr_NN 0.1 --lr_P 5.0 --exp_no 0 --save_iter 100 --plot_result True -N_dim $N_dim


### Conditional MNIST
* Conditional GPA conditioned by MNIST digit labels
* Generate MNIST digits from logistic distribution
* FID calculated with autoencoder_mnist model stored in util/save_model/FID_MNIST_deep_64_encoder and util/save_model/FID_MNIST_deep_64_decoder

python f-Gamma-gpa.py --f KL --formulation DV -L 1 -N_Q 200 --dataset MNIST --N_conditions 10 -NN cnn-fnn -N_cnn_layers 128 128 128 -N_fnn_layers 784 --epochs_nn 10 --epochs 20000 --lr_NN 0.05 --lr_P 1.0 --random_seed 0 --save_iter 100 --plot_result True --exp_no 1

python f-Gamma-gpa-mb.py --f KL --formulation DV -L 1 -N_Q 2000 --mb_size_Q 200 -N_P 200 --dataset MNIST --N_conditions 10 -NN cnn-fnn -N_cnn_layers 128 128 128 -N_fnn_layers 784 --epochs_nn 10 --epochs 20000 --lr_NN 0.05 --lr_P 1.0 --random_seed 0 --save_iter 100 --plot_result True


### Autoencoder MNIST - digit transform
* Compare GPA in the real space and in latent space with dimension 64 and 128
* MNIST dataset transform digit 2 to 0
* Autoencoder trained by ```Python util/autoencoder.py MNIST_deep_64 64``` and ```Python util/autoencoder.py MNIST_deep_128 128```
* FID calculated with autoencoder_mnist model stored in util/save_model/FID_MNIST_deep_64_encoder and util/save_model/FID_MNIST_deep_64_decoder

python f-Gamma-gpa.py --f KL --formulation DV -L 1 -N_Q 200 --dataset MNIST_switch -label 2 0 -NN cnn-fnn -N_cnn_layers 64 64 64 -N_fnn_layers 784 --epochs_nn 10 --epochs 5000 --lr_NN 0.01 --lr_P 0.1 --random_seed 0 --exp_no 0 --save_iter 100 --plot_result True 

python f-Gamma-ae-gpa.py --f KL --formulation DV -L 1 -N_Q 200 --dataset MNIST_ae_switch -label 2 0 -NN fnn -N_latent_dim 64 -N_fnn_layers 256 512 256 --epochs_nn 10 --epochs 5000 --lr_NN 0.01 --lr_P 0.1 --random_seed 0 --exp_no 1 --save_iter 100 --plot_result True

python f-Gamma-ae-gpa.py --f KL --formulation DV -L 1 -N_Q 200 --dataset MNIST_ae_switch -label 2 0 -NN fnn -N_latent_dim 128 -N_fnn_layers 256 512 256 --epochs_nn 10 --epochs 5000 --lr_NN 0.01 --lr_P 0.1 --random_seed 0 --exp_no 2 --save_iter 100 --plot_result True


---

### 2D Gaussian
* Influence of choosing Lipschitz parameter L
* L=1, 10
* random_seed=0 for trajectory plot
* random_seed=0,1,2,3,4 for Divergence, Lipschitz regularized FI loss plots

python f-Gamma-gpa.py --f KL --formulation DV -L $L --dataset Learning_gaussian -sigma_Q 2.0 -NN fnn -N_fnn_layers 32 32 32 --epochs_nn 5 --epochs 500 --lr_NN 0.05 --lr_P 1.0 --exp_no 0 --random_seed 0 --plot_result True


### 2D Stretched exponential
* Another heavy-tailed distribution
* See both KL and alpha=2 work with L=1
* random_seed=0 for trajectory plot
* random_seed=0,1,2,3,4 for Divergence, Lipschitz regularized FI loss plots

python f-Gamma-gpa.py --f KL --formulation DV -L 1.0 --dataset Stretched_exponential -beta 0.4 -NN fnn -N_fnn_layers 32 32 32 --epochs_nn 5 --epochs 5000 --lr_NN 0.05 --lr_P 1.0 --random_seed 0 --exp_no 0 --plot_result True

python f-Gamma-gpa.py --f alpha -alpha 2 -L 1.0 --dataset Stretched_exponential -beta 0.4 -NN fnn -N_fnn_layers 32 32 32 --epochs_nn 5 --epochs 5000 --lr_NN 0.05 --lr_P 1.0 --random_seed 0 --exp_no 0 --plot_result True


### CIFAR10
* high-dimensional image generation with 200 training data without mode collapse
* FID calculated with autoencoder_mnist model stored in util/save_model/FID_CIFAR10_deep_256_encoder and util/save_model/FID_CIFAR10_deep_256_decoder

python f-Gamma-gpa.py --f KL --formulation DV -L 1.0 -N_Q 200 --dataset CIFAR10 -label 0 -NN cnn-fnn -N_cnn_layers 128 128 128 -N_fnn_layers 3072 --epochs_nn 3 --epochs 100000 --lr_NN 0.01 --lr_P 0.1 --random_seed 0 --exp_no 0 --plot_result True


### 2D Mixture of Gaussians
* Fit data to Multi-wells
* Stable with L=1,10,100, but different converging speed
* It diverges without Lipschitz bound 
* random_seed=0 for trajectory plot
* random_seed=0,1,2 for Divergence, Lipschitz regularized FI loss plots

python f-Gamma-gpa.py --f KL --formulation DV -L 1.0 --dataset Mixture_of_gaussians -sigma_Q 0.5 -NN fnn -N_fnn_layers 32 32 32 --epochs_nn 5 --epochs 10000 --lr_NN 0.005 --lr_P 1.0 --random_seed 0 --plot_result True --exp_no 0

python f-Gamma-gpa.py --f KL --formulation DV -L 10.0 --dataset Mixture_of_gaussians -sigma_Q 0.5 -NN fnn -N_fnn_layers 32 32 32 --epochs_nn 5 --epochs 10000 --lr_NN 0.005 --lr_P 1.0 --exp_no 0 --random_seed 0 --plot_result True

python f-Gamma-gpa.py --f KL --formulation DV -L 100.0 --dataset Mixture_of_gaussians -sigma_Q 0.5 -NN fnn -N_fnn_layers 32 32 32 --epochs_nn 5 --epochs 10000 --lr_NN 0.005 --lr_P 1.0 --exp_no 0 --random_seed 0 --plot_result True

python f-Gamma-gpa.py --f KL --formulation DV --dataset Mixture_of_gaussians -sigma_Q 0.5 -NN fnn -N_fnn_layers 32 32 32 --epochs_nn 5 --epochs 10000 --lr_NN 0.005 --lr_P 1.0 --exp_no 0 --random_seed 0 --plot_result True


### 2D Mixture of Gaussians embedded in 12D 
* real space trained using ```f-Gamma-gpa-mb.py```
* latent space trained using ```f-Gamma-ae-gpa.py```
* Autoencoder trained by ```Python util/autoencoder.py Mixture_of_gaussians_submnfld_deep_2 2``` and ```Python util/autoencoder.py Mixture_of_gaussians_submnfld_deep_2_1 2```

python f-Gamma-gpa-mb.py --f KL --formulation DV -L 1.0 --dataset Mixture_of_gaussians_submnfld -sigma_Q 0.5 -N_dim 12 -NN fnn -N_fnn_layers 32 32 32 --epochs_nn 5 --epochs 10000 --lr_NN 0.005 --lr_P 0.5 -N_Q 5000 --mb_size_Q 200 --random_seed 0 --plot_result True --exp_no 0

python f-Gamma-ae-gpa.py --f KL --formulation DV -L 1 -N_Q 200 --dataset Mixture_of_gaussians_submnfld_ae -sigma_Q 0.5 -N_dim 12 -NN fnn -N_fnn_layers 16 32 16 -N_latent_dim 2 --epochs_nn 10 --epochs 5000 --lr_NN 0.005 --lr_P 0.05 --random_seed 0 --exp_no 1 --save_iter 100 --plot_result True
