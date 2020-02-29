import os

import array
import numpy as np
import matplotlib.pyplot as plt


def read_raw_imgs(dataset_path):
    raw_img_names = os.listdir(dataset_path)
    raw_imgs = []
    for im_path in raw_img_names:
        if im_path[-4:] != ".iml":
            continue # not .iml image file
        with open(dataset_path+im_path, 'rb') as handle:
           s = handle.read()
        arr = array.array('H', s)
        arr.byteswap()
        img = np.array(arr, dtype='uint16').reshape(1024, 1536)
        raw_imgs.append(img)
    return np.stack(raw_imgs)


def normalize_imgs(img_arr):
    norm_imgs = img_arr - np.min(img_arr, axis=0)
    return norm_imgs / np.max(norm_imgs, axis=0)


def whiten_imgs(X, eps=1e-8):
    num_imgs, img_h, img_w = X.shape
    X = X.reshape(num_imgs, -1)
    X = X - np.mean(X, axis=0)
    Xcov = np.dot(X.T, X) / num_imgs
    d, V = np.linalg.eigh(Xcov)
    D = np.diag(1. / np.sqrt(d+eps))
    W = np.dot(np.dot(V, D), V.T)
    wtn_X = np.dot(X, W)
    wtn_X = wtn_X.reshape(num_imgs, img_h, img_w)
    return (wtn_X, W)  # whiten img arry, whiten matrix


def extract_patches(img_arr, num_patches, patch_size=(16, 16)):
    if len(img_arr.shape) == 3:
        num_imgs, img_h, img_w = img_arr.shape
    else:
        num_imgs, img_h, img_w, _ = img_arr.shape

    h_range = img_h - patch_size[0]
    w_range = img_w - patch_size[1]
    img_patches = []
    for _ in range(num_patches):
        k_pick = np.random.randint(low=0, high=num_imgs, size=None)
        h_pick = np.random.randint(low=0, high=h_range, size=None)
        w_pick = np.random.randint(low=0, high=w_range, size=None)
        img_patches.append(img_arr[k_pick, h_pick:h_pick+patch_size[0],
                                   w_pick:w_pick+patch_size[1]])
    return np.stack(img_patches)


def plot_neurons(mat):
    fig, ax = plt.subplots(ncols=16, nrows=16, figsize=(10, 10), dpi=200)
    k = 0
    for i in range(16):
        for j in range(16):
            ax[i, j].imshow(mat[:, k].reshape(16, 16), cmap="gray")
            ax[i, j].set_axis_off()
            k += 1
    plt.show()
