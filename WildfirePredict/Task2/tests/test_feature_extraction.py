import unittest
import numpy as np
from feature_extraction import extract_features, compute_weights
from metrics import mse, psnr, cosine_sim

class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        # Create some mock images
        self.image1 = np.array([[1, 2], [3, 4]])
        self.image2 = np.array([[4, 3], [2, 1]])
        self.images1 = [self.image1]
        self.images2 = [self.image2]
    
    def test_extract_features(self):
        features = extract_features(self.images1, self.images2, mse, cosine_sim, psnr)
        self.assertEqual(len(features), 1)
        self.assertEqual(len(features[0]), 3)
    
    def test_compute_weights(self):
        features = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
        w1, w2, w3 = compute_weights(features)
        self.assertAlmostEqual(w1 + w2 + w3, 1.0, places=5)

if __name__ == '__main__':
    unittest.main()