import numpy as np
from WildfirePredict.Task2.chosing_images.metrics import mse, psnr, cosine_sim

def combined_similarity_score(gen_image, obs_image, w1=0.48, w2=0.04, w3=0.48):
    mse_score = mse(gen_image, obs_image)
    cos_sim_score = cosine_sim(gen_image, obs_image)
    psnr_score = psnr(gen_image, obs_image)

    combined_score = (w1 * mse_score) + (w2 * (1 - cos_sim_score)) + (w3 * (1 / psnr_score))
    return combined_score

def compare_images(generated_images, observed_images, w1=0.48, w2=0.04, w3=0.48):
    best_matches = []
    for obs_image in observed_images:
        scores = [combined_similarity_score(gen_image, obs_image, w1, w2, w3) for gen_image in generated_images]
        best_match_idx = np.argmin(scores)
        best_matches.append((best_match_idx, scores[best_match_idx]))
    return best_matches
