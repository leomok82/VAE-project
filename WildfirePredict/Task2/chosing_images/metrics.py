import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import skimage.metrics

def mse(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):
    return skimage.metrics.peak_signal_noise_ratio(imageA, imageB)

def cosine_sim(imageA, imageB):
    imageA_flat = imageA.flatten().reshape(1, -1)
    imageB_flat = imageB.flatten().reshape(1, -1)
    return cosine_similarity(imageA_flat, imageB_flat)[0][0]