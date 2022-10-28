import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt

from sys import argv

#d_red = 200 # reduced dimension
d_red = int(argv[1])

folder_name = "GPL570/BreastCancer/"
dset1_name = folder_name+"GSE47109.csv" # source distribution
dset2_name = folder_name+"GSE10843.csv" # target distribution


# read datasets
data1 = pd.read_csv(dset1_name)
X1 = pd.DataFrame.to_numpy(data1).T
(n1, d) = X1.shape
print(n1, d)

data2 = pd.read_csv(dset2_name)
X2 = pd.DataFrame.to_numpy(data2).T
n2 = X2.shape[0]
#print(n2)

# Standardize the features
scaler1 = StandardScaler()# Fit on training set only.
scaler1.fit(X1)
X1 = scaler1.transform(X1)

scaler2 = StandardScaler()# Fit on training set only.
scaler2.fit(X2)
X2 = scaler2.transform(X2)

# concatenate
X = np.concatenate((X1, X2))
#print(X.shape)


# perform PCA
pca = decomposition.PCA(n_components=d_red)
pca.fit(X)


#perform dimensionality reduction
X_red = pca.transform(X)
#print(X_red.shape)
X1_red = X_red[:n1-1,:]
X2_red = X_red[n1:,:]

##plot in the latent space
#plt.plot(X1_red[:,0], X1_red[:,1], 'bo')
#plt.plot(X2_red[:,0], X2_red[:,1], 'ro')
#plt.show()

# save
pk.dump(pca, open(folder_name+"pca_norm_"+str(d_red)+".pkl","wb"))

np.savetxt(folder_name+"source_norm_dataset_dim_"+str(d_red)+".csv", X1_red, delimiter=",")
np.savetxt(folder_name+"target_norm_dataset_dim_"+str(d_red)+".csv", X2_red, delimiter=",")

# reconstruction
X1_hat = pca.inverse_transform(X1_red)
X2_hat = pca.inverse_transform(X2_red)

#plot original and reconstructed
idx1, idx2 = 1, 2
plt.plot(X[:n1-1,idx1], X[:n1-1,idx2], 'bo')
plt.plot(X[n1:,idx1], X[n1:,idx2], 'ro')
plt.plot(X1_hat[:,idx1], X1_hat[:,idx2], 'b*')
plt.plot(X2_hat[:,idx1], X2_hat[:,idx2], 'r*')
plt.tight_layout()
plt.show()
