import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import umap

import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt

from mmd_numpy_sklearn import mmd_linear



from sys import argv

d_red = 200 # reduced dimension
#d_red = int(argv[1])

folder_name = "GPL570/BreastCancer/"
dset1_name = folder_name+"GSE47109.csv" # source distribution
dset2_name = folder_name+"GSE10843.csv" # target distribution


d_name = folder_name.split('/')
d_name = d_name[1]
norm = "norm_"
#norm = ""

red_dset1_name = folder_name+"source_"+norm+"dataset_dim_"+str(d_red)+".csv"
trans_red_dset1_name = folder_name+"output_"+norm+"dataset_dim_"+str(d_red)+".csv"
#trans_red_dset1_name = folder_name+"KL-Lipshitz_1.00_dim20_0142_0141_00_dim_"+str(d_red)+".csv"
red_dset2_name = folder_name+"target_"+norm+"dataset_dim_"+str(d_red)+".csv"


# read data
data1 = pd.read_csv(dset1_name)
X1 = pd.DataFrame.to_numpy(data1).T
#print(X1.shape)

# Standardize the features
scaler1 = StandardScaler()# Fit on training set only.
scaler1.fit(X1)
X1 = scaler1.transform(X1)

data2 = pd.read_csv(dset2_name)
X2 = pd.DataFrame.to_numpy(data2).T

# Standardize the features
scaler2 = StandardScaler()# Fit on training set only.
scaler2.fit(X2)
X2 = scaler2.transform(X2)


red_data1 = pd.read_csv(red_dset1_name)
X1_red = pd.DataFrame.to_numpy(red_data1)

red_data2 = pd.read_csv(red_dset2_name)
X2_red = pd.DataFrame.to_numpy(red_data2)

trans_red_data1 = pd.read_csv(trans_red_dset1_name)
X1_red_trpt = pd.DataFrame.to_numpy(trans_red_data1)


# reconstruction
pca = pk.load(open(folder_name+"pca_norm_"+str(d_red)+".pkl", 'rb'))

X1_hat = pca.inverse_transform(X1_red)
X2_hat = pca.inverse_transform(X2_red)
X1_trpt_hat = pca.inverse_transform(X1_red_trpt)


# MMDs with linear kernel

mmd11h = mmd_linear(X1, X1_hat)
mmd22h = mmd_linear(X2, X2_hat)

mmd12 = mmd_linear(X1, X2)
mmd1th = mmd_linear(X1, X1_trpt_hat)
mmd2th = mmd_linear(X2, X1_trpt_hat)

mmd1h2h = mmd_linear(X1_hat, X2_hat)
mmdth1h = mmd_linear(X1_trpt_hat, X1_hat)
mmdth2h = mmd_linear(X1_trpt_hat, X2_hat)


print("MMD in the real space: ", mmd11h, mmd22h, mmd12, mmd1th, mmd2th, mmd1h2h,  mmdth1h, mmdth2h)

# MMDs with linear kernel
mmd12_ = mmd_linear(X1_red, X2_red)
mmd11h_ = mmd_linear(X1_red, X1_red_trpt)
mmd22h_ = mmd_linear(X2_red, X1_red_trpt)
print("MMD in the latenet space: ",mmd12_, mmd11h_, mmd22h_)


# dimensionality reduction for visualization (In the latent space)
reducer = umap.UMAP()

#reducer.fit(np.concatenate((X1_hat, X2_hat)))
reducer.fit(np.concatenate((X1_red, X2_red)))

X1_red_vis = reducer.transform(X1_red)
X2_red_vis = reducer.transform(X2_red)
X1_red_trpt_vis = reducer.transform(X1_red_trpt)

plt.plot(X1_red_vis[:, 0], X1_red_vis[:, 1], 'bo', label='source', alpha=0.3)
plt.plot(X1_red_trpt_vis[:, 0], X1_red_trpt_vis[:, 1], 'k*', label='output',alpha=0.4)
plt.plot(X2_red_vis[:, 0], X2_red_vis[:, 1], 'ro', label='target' , alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("%s%s_%sdim%d-latent_data.png" % (folder_name, d_name, norm, d_red))
plt.show()


# dimensionality reduction for visualization (In the real space)
reducer.fit(np.concatenate((X1, X2)))

X1_vis = reducer.transform(X1_hat)
X2_vis = reducer.transform(X2_hat)
X1_trpt_vis = reducer.transform(X1_trpt_hat)

plt.plot(X1_vis[:, 0], X1_vis[:, 1], 'bo', label='source', alpha=0.3)
plt.plot(X1_trpt_vis[:, 0], X1_trpt_vis[:, 1], 'k*', label='output',alpha=0.4)
plt.plot(X2_vis[:, 0], X2_vis[:, 1], 'ro', label='target' , alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("%s%s_%sdim%d-reconstructed_data.png" % (folder_name, d_name, norm, d_red))
plt.show()



#plot
#plt.plot(X1[:,0], X1[:,1], 'bo')
#plt.plot(X2[:,0], X2[:,1], 'ro')
#plt.plot(X1_hat[:,0], X1_hat[:,1], 'g*')
#plt.plot(X2_hat[:,0], X2_hat[:,1], 'y*')
#plt.plot(X1_trpt_hat[:,0], X1_trpt_hat[:,1], 'k*')
#plt.show()

