import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import skimage.metrics

__all__ = ['mse', 'psnr', 'cosine_sim', 'combined_similarity_score', 'compare_images']

def mse(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return np.finfo(np.float64).max  # Return a very high PSNR value for identical images
    psnr_value = skimage.metrics.peak_signal_noise_ratio(imageA, imageB)
    return psnr_value

def cosine_sim(imageA, imageB):
    imageA_flat = imageA.flatten().reshape(1, -1)
    imageB_flat = imageB.flatten().reshape(1, -1)
    return cosine_similarity(imageA_flat, imageB_flat)[0][0]

def combined_similarity_score(gen_image, obs_image, w1=0.48, w2=0.04, w3=0.48):
    mse_score = mse(gen_image, obs_image)
    cos_sim_score = cosine_sim(gen_image, obs_image)
    psnr_score = psnr(gen_image, obs_image)

    # Set a lower bound for PSNR to avoid extremely high values
    min_psnr = 1e-10
    psnr_score = max(psnr_score, min_psnr)

    combined_score = (w1 * mse_score) + (w2 * (1 - cos_sim_score)) + (w3 * (1 / psnr_score))
    return combined_score

def compare_images(generated_images, observed_images, w1=0.48, w2=0.04, w3=0.48):
    best_matches = []
    for obs_image in observed_images:
        scores = [combined_similarity_score(gen_image, obs_image, w1, w2, w3) for gen_image in generated_images]
        best_match_idx = np.argmin(scores)
        best_matches.append((best_match_idx, scores[best_match_idx]))
    return best_matches
