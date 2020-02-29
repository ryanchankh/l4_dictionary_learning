import os

import array
import numpy as np
import matplotlib.pyplot as plt

import utils


DATASET_PATH = "./dataset/van_hateren/"
mode = "l4"
p = int(5e5) # number of samples
n = 256  # data dimension
num_steps = 1000

# process data
print("processing data")
img_arr = utils.read_raw_imgs(DATASET_PATH)
img_arr = utils.normalize_imgs(img_arr)
img_arr = utils.extract_patches(img_arr, p)
img_arr, wtn_mat = utils.whiten_imgs(img_arr)

# initialize model
Y = img_arr.reshape(p, n).T
A = np.linalg.qr(np.random.rand(n, n))[0]

# train model 
if mode == "l4":
    for t in range(num_steps):
        dA = 4 * np.power(A @ Y, 3) @ Y.T
        U, _, V = np.linalg.svd(dA, compute_uv=True)
        A = U @ V
#
        loss = np.sum(np.power(A@Y, 4))
        print(f'step: {t}\tloss: {loss}')


    # plot basis
    utils.plot_neurons(A.T)

if mode == "fastica":

    from sklearn.decomposition import FastICA
    transformer = FastICA(n_components=n, whiten=False, w_init=A)
    X_transformed = transformer.fit(Y.T)

    utils.plot_neurons(transformer.components_)
