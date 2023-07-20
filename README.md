# Lipschitz-regularized Generative Particles Algorithm
We propose **a generative model based on solving ODEs**. Unlike Score-based Generative Model(SGM, Song et al., [2020](https://arxiv.org/abs/2011.13456)) which push-forwards a target measure to Gaussian and reverses the process by solving differential equations, Lipschitz-regularized Generative Particles Algorithm (Lipschitz-regularized GPA) push-forwards an arbitrary source measure to a target measure. Our ODE systems are derived from gradient flows likewise to Maximum Mean Discrepancy flow(MMD flow, Arbel et al., [2019](https://arxiv.org/abs/1906.04370)) and Stein Variational Gradient Descent as Gradient Flow(SVGD flow, Liu, [2017](https://arxiv.org/abs/1704.07520)) which gave us inspiration to build our key algorithm. But our loss functions are optimized over the space of Lipschitz continuous Neural Networks instead of RKHS from the two former papers so that our algorithm is ready-to-use for various interesting examples as well as scalable to higher dimensions $\leq 784$ in our observation without assists of other techniques. Also, our Lipschitz-regularized loss functions are well-defined as divergences as discovered by Birrell [2020](https://arxiv.org/abs/2011.05953) and Dupuis [2019](https://arxiv.org/abs/1911.07422).

<img align="center" width="250" alt="KL-Lip1 GPA transporting Gaussian to Sierpinski carpet in 3D" src="figures/kl-lipschitz_1_4096_4096_00_test_sierpinski2movie.gif?raw=true"/><img align="center" width="250" alt="alpha2-Lip1 GPA transporting MNIST digit 2 to MNIST digit 0 in 784D" src="figures/alpha=2-lipshitz_1_02_00_0200_0200_00_0movie.gif?raw=true"/><img align="center" width="250" alt="KL-Lip1 GPA transporting 5000 samples from Gaussian to Swiss roll approximated by 200 samples in 3D" src="figures/kl-lipschitz_1_0200_5000_00_3d_swiss_roll-movie.gif?raw=true"/>

## Lipschitz-regularized $f$-divergences
We use a new divergence from Birrell [2020](https://arxiv.org/abs/2011.05953) which combines two *metrics* on probability measures.

* For a convex, superlinear, lower semi-continuous function $f$ with $f(1)=0$, classical $f$-divergences $D_f(P\|Q)$ are defined as $E_Q [f(\frac{dP}{dQ})]$ for $P \ll Q$ and $\infty$ otherwise. Obviously, it is meaningful when $P$ is absolutely continuous with respect to $Q$. Some examples are $f(x)=x\log(x)$ defines KL divergence, $f(x)=\frac{x^\alpha-1}{\alpha (\alpha-1)}$ with $\alpha > 1$ define alpha-divergences where the alpha-divergence converges to KL-divergence as $\alpha \rightarrow 1$. There are various $f$'s in this class as listed in  $f$-GANs (Nowozin, [2016](https://arxiv.org/abs/1606.00709)). 

* On the other hand, integral probability metrics(IPMs) $W(P,Q)$ are defined as $$W(P,Q) = \sup\_{\phi \in \Gamma} \{ E_P[\phi]-E_Q[\phi] \}$$ where $\Gamma= \\{\phi:\mathbb{R}^d \rightarrow \mathbb{R}\\}$ denotes a function space. Different function spaces define different IPMs, for example, MMD is defined on $\\{\phi: \\|\phi\\|_{RKHS} \leq 1\\}$, and 1-Wasserstein metric $W_1$ is defined on $\\{\phi: \phi \text{ is 1-Lipschitz continuous}\\}$. On its definition of IPMs, one solves an infinite dimensional optimization problem over functions in $\Gamma$. It is called a **variational representation**.

    Likewise to IPMs, $f$-divergences admit variational representation $$D_f(P \| Q) = \sup\_{\phi \in C_b(\mathbb{R}^d)} \left \\{E_P[\phi] - \inf\_{\nu \in \mathbb{R}} \left \\{ E_Q[f^\*(\phi-\nu)+\nu] \right \\} \right \\}$$ where $C_b$ denotes the set of continuous and bounded functions and $f^\*$ denotes the Legendre transform of $f$. A variational representation of KL divergence from Donsker-Varadhan $$D\_{f\_\text{KL}} (P \| Q)=\sup\_{\phi \in C_b(\mathbb{R}^d)} \left\\{E_P[\phi]-\log E_Q[\exp(\phi)]\right\\}$$ reduces to this family by optimizing the $\nu$ inside.

$(f, \Gamma)$-divergences (Birrell, [2020](https://arxiv.org/abs/2011.05953)) are defined as $$D_f^\Gamma(P \| Q) = \sup\_{\phi \in \Gamma} \left\\{E_P[\phi]- \inf\_{\nu \in \mathbb{R}} \\{E_Q[f^\*(\phi-\nu)+\nu]\\}\right\\}$$ and form divergences for function spaces $\Gamma$ which are rich enough to discriminate different measures $P$ and $Q$. As shown from its formula, $(f, \Gamma)$-divergences interpolate between $f$-divergences and IPMs. 

In this document, we fix the function space $\Gamma_L=\\{\phi: \phi \text{ is } L-\text{Lipschitz continuous}\\}$ and hence focus on $$D_f^{\Gamma_L}(P \| Q) = \sup\_{\phi \in \Gamma_L} \left\\{E_P[\phi]- \inf_\{\nu \in \mathbb{R}} \\{E_Q[f^\*(\phi-\nu)+\nu]\\}\right\\}$$ which is called **Lipschitz-regularized $f$-divergence** or **$(f,\Gamma_L)$-divergence**. 

1. It interpolates between $f$-divergence and 1-Wasserstein metric through the parameter $L$ where $$D_f^{\Gamma_L}(P \| Q) \rightarrow D_f(P \| Q) \text{ as } L \rightarrow \infty$$ and $$D_f^{\Gamma_L}(P \| Q) \rightarrow W_1(P, Q) \text{ as } L \rightarrow 0.$$ 
2. Its value remains finite on wider range of $P$ and $Q$ even for $P$ non-absolutely continuous with respect to $Q$ from the property that $$D_f^\Gamma(P \| Q) \leq \inf \left \\{ D_f(P\|Q), L\*W_1(P,Q) \right \\}.$$
3. The optimizer $\phi^{L,\*} \in \Gamma_L$ uniquely exists up to a constant in the support of $P$ and $Q$.

The variational representation enables Lipschitz-regularized $f$-divergences to be computed on finite number of samples from the propability measures $P$ and $Q$. Let $M$ samples $\{Y^{(i)}\}$ from $P$ and $N$ samples $X^{(i)}$ from $Q$ approximate each distribution by $P\^M = \frac{1}{M}  \sum\_{i=1}^M \~ \delta\_{Y^{(i)}}$ and $Q\^N=\frac{1}{N} \sum\_{i=1}^N \~ \delta\_{X^{(i)}}$, respectively. Depending on the data type, a sample could be a record of a numeric table, an image, a or an audio clip. Currently we support *Numpy* arrays in Python for tables or 2D images. *Be careful with categorical data since we assume the Euclidean distance in space.* The mean of each measure is approximated by the sample mean and results in the formula $$D_f^{\Gamma_L}(P\^M \| Q\^N) = \sup\_{\phi \in \Gamma_L} \left\\{ \frac{1}{M}\sum\_{i=1}^M \phi(Y^{(i)})- \inf\_{\nu \in \mathbb{R}} \left \\{\frac{1}{M}\sum\_{i=1}^Mf^\*(\phi(X^{(i)})-\nu)+\nu \right \\}\right\\}.$$ The lines below calculate the $(f, \Gamma_L)$-divergence of $P$ and $Q$. (It is supported from `v0.2.0`.)
```python
#!/usr/bin/env python3
import numpy as np
    
P = np.random.normal(loc=0.0, scale=1.0, size=(500,2)) # 500 2D samples from P
Q = np.random.normal(loc=10.0, scale=2.0, size=(300,2)) # 300 2D samples from Q
    
print(f_Lip_divergence(P, Q, L=1.0))
```

## Gradient flow on probability measures and Lipschitz-regularized GPA
In our paper [2022](https://arxiv.org/abs/2210.17230), it is proven that the optimal $\phi^{L,\*} \in \Gamma_L$ in the variational representation of the Lipschitz-regularized $f$-divergence $D_f^{\Gamma_L}(P\|Q)$ is the **first variation of the Lipschitz-regularized divergence $D_f^{\Gamma_L}(P\|Q)$** with respect to purturbing the first argument $P$: 
$$\frac{\delta  D_f^{\Gamma_L}(P\|Q)}{\delta P}= \phi^{L,\*} =  \underset{\phi\in \Gamma_L}{\rm argmax} \left\\{E_P[\phi]- \inf\_{\nu \in \mathbb{R}}(\nu + E_Q[f^\*(\phi-\nu)])\right\\}.$$ 
Precisely, the optimal $\phi^{L,\*}$ serves as a potential to transport the probability measure $P$ toward $Q$ leading to the *transport/variational* PDE reformulation of:
$$\partial\_t P_t + {\rm div}(P_t v_t\^L) =0, \quad P_0=P \in \mathcal{P}\_1(\mathbb{R}^d)$$
$$v_t\^L= -\nabla \phi_t^{L,\*}, \quad \phi_t^{L,\*} = \underset{\phi \in \Gamma_L}{\rm argmax}  \left\\{E\_{P_t}[\phi]- \inf\_{\nu \in \mathbb{R}}(\nu + E_Q[f^\*(\phi-\nu)])\right \\}.$$ Note that without Lipschitz regularization, the velocity field $v\_t$ will diverge especially when $P_t \not\ll Q$. On the other hand, **Lipschitz-regularization bounds the particle speed by $L$:  $\\|v_t\^L\\|\leq L$.** In addition, the solution curve $\\{P_t\\}\_{t\geq 0}$ of the transport PDE dissipates $D_f^{\Gamma_L}(P\|Q)$: $$\frac{d}{dt} D_f^{\Gamma_L} (P_t\|Q)=-I_f\^{\Gamma_L}(P_t\|Q)=\int |\nabla \phi_t^{L, \*}|^2 P_t(dx) \leq 0.$$ 

<img align="center" width="860" alt="Gradient flow on probability measures which dissipates the divergence or loss" src="figures/gradient_flow_figure.png?raw=true"/> 

Refer to more details on Wasserstein gradient flows in Chapter 7, 8 of the book by [Santambrogio](https://www.semanticscholar.org/paper/Optimal-Transport-for-Applied-Mathematicians%3A-of-Santambrogio/5d3f82856178dca5b48d14a8259b66121145c97c).

From a computational perspective, it becomes feasible to solve the high-dimensional transport PDE by considering the Lagrangian formulation of the transport PDE, i.e.  the **ODE/variational problem** $$\frac{d}{dt} Y_t = v_t\^L(Y_t)=-\nabla \phi_t^{L,\*} (Y_t), \quad Y_0 \sim P,$$
    $$\phi_t^{L,\*}=\underset{\phi\in \Gamma_L}{\rm argmax}\left\\{E\_{P_t}[\phi]-\inf\_{\nu \in \mathbb{R}}\left\\{ \nu + E\_{Q}[f^\*(\phi -\nu)]\right\\}  \right\\}.$$ Upon Euler time discretization we get **Lipschitz-regularized generative particle algorithm (GPA)**: $$Y\_{n+1}^{(i)}=Y_n^{(i)}-\Delta t\nabla {\phi}\_n^{L, \*}(Y_n^{(i)})\,, \quad Y^{(i)}\_0=Y^{(i)}\, , \, Y^{(i)} \sim P\, , \quad i=1,..., M$$ $$\phi_n^{L,\*}=\underset{\phi\in \Gamma_L^{NN}}{\rm argmax} \left\\{\frac{1}{M}\sum\_{i=1}^{M} \phi(Y_n^{(i)})- \inf\_{\nu \in \mathbb{R}}\left\\{ \nu + \frac{1}{N}\sum\_{i=1}^{N} f^\*(\phi(X^{(i)})-\nu)\right\\}\right\\}$$ where $\Gamma_L^{NN}$ denotes the space of $L$-Lipschitz continuous neural networks imposing Lipschitz continuity from spectral normalization by [Miyato, 2018](https://arxiv.org/abs/1802.05957). The **Lipschitz-regularization enables finite-speed propagation of particles and it ensures the numerical algorithm not to diverge.** As mentioned above, similar computational framework of solving ODEs/variational problems from gradient flows is used in MMD flow [Arbel et al., 2019](https://arxiv.org/abs/1906.04370) and SVGD flow [Liu, 2017](https://arxiv.org/abs/1704.07520).
    
On the other hand, Generative Adversarial Network (GAN), precisely $f$-GAN [Nowozin, 2016](https://arxiv.org/abs/1606.00709) also optimizes $\phi$ named *discriminator*. Instead of transporting particles, GAN redistributes particles by training another neural network named *generator* $g\_\theta$ for minimizing the loss, i.e. $\max\_{\phi} H_f[\phi; g\_\theta (Z), X]$: $$\min\_{\theta} \max\_{\phi} H_f[\phi; g\_\theta (Z), X] , \quad \text{where}$$ $$H_f[\phi; g\_\theta (Z), X] = \frac{1}{M}\sum\_{i=1}\^M \phi(g\_{\theta} (Z^{(i)}))- \inf\_{\nu \in \mathbb{R}}\left\\{ \nu + \frac{1}{N}\sum\_{i=1}\^N f^\*(\phi(X^{(i)})-\nu)\right\\}.$$

<img align="center" width="1040" alt="Comparison of computational schemes: Left - GAN, Right - GPA" src="figures/gpa_vs_gan.png?raw=true"/> 

The lines below run the $(f, \Gamma_L)$-GPA on the source $P$ and the target $Q$. Since GPA hyper-parameters and data-dependent parameters should be tuned example-by-example, it is required to 

1. list up such parameters in `configs/"Dataset name"-GPA_NN.YAML`,
2. write down a script which defines a dataloader in `data/"Recognizable dataloader name".py`,
3. add lines in `shared_lib/generate_data.py` to call the dataloader function in the script written in step 2. If it is simple, one can directly write down a dataloader function in `shared_lib/generate_data.py`.

Then it is ready to run GPA in a console
```console
python3 main.py --dataset "Dataset name" --phi_model GPA_NN
```
or in a Jupyter notebook. There are sample Jupyter notebook examples in the `notebooks` folder.

## Features of Lipschitz-regularized GPA
### Flexibility in the choice of Loss
We observed that the choice of $f\_\text{KL}$ for heavy-tailed data $Student-t(\nu)$ with $\nu=0.5$ renders the discriminator optimization step numerically unstable and eventually leads to the collapse of the algorithm.  On the other hand, the choice of $f\_\alpha$ with $\alpha > 1$ makes the algorithm  stable. However, it still takes a long time to transport particles deep into the  heavy tails due to the speed restriction of the Lipschitz regularization.

| $(f\_{\text{KL}}, \Gamma\_1)$-GPA | $(f\_{\alpha}, \Gamma\_1)$-GPA, $\alpha=2.0$ | $(f\_{\alpha}, \Gamma\_1)$-GPA, $\alpha=10.0$ | 
| :------------------------------: | :-----------------------------------------: | :----------------------------: |
| <img align="center" width="210" alt="KL-Lip1 GPA transporting Gaussian to Student-t(0.5) in 2D" src="./Lipschitz_regularized_generative_particles_algorithm/main/figures/kl-lipshitz_1_0p5_0200_0200_00_heavy_tail-movie.gif?raw=true"/> | <img align="center" width="210" alt="alpha=2-Lip1 GPA transporting Gaussian to Student-t(0.5) in 2D" src="figures/alpha=2-lipshitz_1_0p5_0200_0200_00_heavy_tail-movie.gif?raw=true"/> | <img align="center" width="210" alt="alpha=10-Lip1 GPA transporting Gaussian to Student-t(0.5) in 2D" src="figures/alpha=10-lipshitz_1_0p5_0200_0200_00_heavy_tail-movie.gif?raw=true"/> |

Similar behavior is observed in GAN [Birrell, 2020](https://arxiv.org/abs/2011.05953): $f\_\alpha$ was more effective than $f\_\text{KL}$ in learning a heavy-tailed distribution with GAN.


### Learning from scarce data
Instead of learning a generator $g\_\theta$ as in GANs, solving ODEs in GPA makes it available to learn from a small number of target samples while GANs fail in the same setting. In the MNIST example, we used 200 target samples to learn GPA and GANs. Discriminators in GANs and GPA are implemented in similar neural network structures but only GANs fail when the target sample size is small. We demonstrate how to cure this problematic behavior of GANs from a relatively simple example below.


| $(f\_{\text{KL}}, \Gamma\_1)$-GPA | $(f\_{\text{KL}}, \Gamma\_1)$-GAN | Wasserstein GAN | 
| :-------------------------------: | :-------------------------------: | :-------------: |
|  <img align="center" height="220" alt="Generated images from KL-Lip1 GPA using 200 target samples" src="figures/kl-lip1-gpa-600_200samples.png?raw=true"/> |  <img align="center" height="220" alt="Generated images from KL-Lip1 GAN using 200 target samples" src="figures/kl-lip1-gan-200samples.png?raw=true"/> |  <img align="center" height="220" alt="Generated images from Wasserstein GAN using 200 target samples" src="figures/wasserstein-gan-200samples.png?raw=true"/> |

There are a lot of literatures and methods for data augmentation to enrich training samples for GANs. GPA can be used as a data augmentation tool. [Swiss roll example](#lipschitz-regularized-generative-particles-algorithm) in the introduction restricts the setting that only 200 training data are available. We generate 5000 artificial samples from these 200 training data using GPA and then train a GAN with the original 200 + the 5000 GPA-augmented data. It stabilizes the GAN-training as well as results in a better quality for the generated samples from GAN. 

| $(f\_{\text{KL}}, \Gamma\_1)$-divergence while training GAN | Generated samples from GAN trained with 200 original samples | Generated samples from GAN trained with 200 + 5000 GPA-augmented samples | 
| :---------------------------------------------------------: | :----------------------------------: | :-----------------------------------------: |
|  <img align="center" width="600" alt="KL-Lip1 GAN training loss decrease using 200 target samples and 200 target samples + 5000 augmented samples obtained by KL-Lip1 GPA" src="figures/data_augmentation_influence.png?raw=true"/> |  <img align="center" width="550" alt="Generated samples from KL-Lip1 GAN using 200 target samples" src="figures/without_augmentation.png?raw=true"/> |  <img align="center" width="220" alt="Generated samples from KL-Lip1 GAN using 200 target samples + 5000 augmented samples obtained by KL-Lip1 GPA" src="figures/with_augmentation.png?raw=true"/> |


### Sample diversity of the generated samples
Since GPA is designed to transport particles to particles, the source particles should eventually match the target particles when the number of samples are equal $M=N$. Indeed it is not rare to observe that the transported particles exactly match the target particles. On the other hand, when $M > N$, it is less likely to meet this overfitting behavior in shapes and then GPA is entitled to be a generative model. 


| $N=200$ Target samples |
|:----------------------:|
| <img align="center" height="220" alt="MNIST 200 target samples" src="figures/kl-lipschitz_1_cond_0200_0600_00_check_overfitting-tiled_target.png?raw=true"/> |


|$M=200$ Generated samples from $(f\_\text{KL}, \Gamma_1)$-GPA | $M=600$ Generated samples from $(f\_\text{KL}, \Gamma_1)$-GPA |
|:--------------------------------------------------------:|:---------------------------------------------------------------------:|
|<img align="center" height="220" alt="MNIST 200 target samples" src="figures/kl-lip1-gpa-200_200samples.png?raw=true"/> | <img align="center" height="220" alt="MNIST 200 target samples" src="figures/kl-lip1-gpa-600_200samples.png?raw=true"/> |



## Application to high-dimensional gene expression data sets integration
Since GPA admits setting arbitrary source $P$ and target $Q$, phase transition arises as a natural application of GPA. There is also a GAN architecture designed for this problem: Cycle-Consistent Adversarial Network(Cycle-GAN, Zhu, [2017](https://arxiv.org/abs/1703.10593)). We suggest to apply GPA and transport one data set to the other in order to integrate several gene expression data sets aiming at studying a same disease with a similar experiment design but having batch effects from their different measuring conditions. 

 However, gene expression data lie in a significantly high-dimensional space $\mathbb{R}^{54,675}$ where gradient descent on particles is feckless. Therefore, we use a pretrained autoencoder  to project data sets into a lower dimensional latent space and transport the data set in the latent space. Indeed we ran GPA in a 50 dimensional latent space obtained by PCA serving as an autoencoder. Finally, the reconstruction of the transported data set to the original space completes the task.

| Two data sets showing batch effects | Integrated data sets from GPA |
|:--------------------------------------------------------:|:---------------------------------------------------------------------:|
|<img align="center" height="220" alt="Two data sets showing batch effects" src="figures/source_target-dim50-standardize_all_3.png?raw=true"/> | <img align="center" height="220" alt="Integrated data sets from GPA" src="figures/transported_target-dim50-standardize_all_3.png?raw=true"/> |

This example previews how to apply GPA on higher dimensional examples. More details are available in our paper.

