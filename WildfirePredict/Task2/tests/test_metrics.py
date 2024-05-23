import unittest
import numpy as np
from metrics import mse, psnr, cosine_sim

class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Create some mock images
        self.image1 = np.array([[1, 2], [3, 4]])
        self.image2 = np.array([[4, 3], [2, 1]])
    
    def test_mse(self):
        result = mse(self.image1, self.image2)
        expected = np.mean((self.image1 - self.image2) ** 2)
        self.assertEqual(result, expected)
    
    def test_psnr(self):
        result = psnr(self.image1, self.image2)
        self.assertTrue(result > 0)
    
    def test_cosine_sim(self):
        result = cosine_sim(self.image1, self.image2)
        self.assertTrue(0 <= result <= 1)

if __name__ == '__main__':
    unittest.main()