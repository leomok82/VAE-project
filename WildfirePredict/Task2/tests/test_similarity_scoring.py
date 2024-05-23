import unittest
import numpy as np
from similarity_scoring import combined_similarity_score, compare_images
from metrics import mse, psnr, cosine_sim

class TestSimilarityScoring(unittest.TestCase):

    def setUp(self):
        # Create some mock images
        self.image1 = np.array([[1, 2], [3, 4]])
        self.image2 = np.array([[4, 3], [2, 1]])
        self.images = [self.image1, self.image2]
    
    def test_combined_similarity_score(self):
        score = combined_similarity_score(self.image1, self.image2, 0.4, 0.3, 0.3)
        self.assertTrue(score >= 0)
    
    def test_compare_images(self):
        best_matches = compare_images(self.images, self.images, 0.4, 0.3, 0.3)
        self.assertEqual(len(best_matches), len(self.images))

if __name__ == '__main__':
    unittest.main()