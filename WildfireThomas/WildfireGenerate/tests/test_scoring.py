import pytest
import numpy as np

from WildfireThomas.WildfireGenerate.task2functions import scoring

@pytest.fixture
# def images():
#     # Generate two noisy images
#     image_shape = (256, 256)  # Define image shape

#     # Generate noisy images
#     image1 = skimage.img_as_float(np.random.normal(0, 10, image_shape))
#     image2 = skimage.img_as_float(np.random.normal(0, 10, image_shape))
#     images = [image1, image2]
#     return images

def images():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[4, 3], [2, 1]])
    return image1, image2
 
def test_mse(images):
    image1, image2 = images
    result = scoring.mse(image1, image2)
    expected = np.mean((image1 - image2) ** 2)
    assert result == expected
 
def test_psnr(images):
    image1, image2 = images
    result = scoring.psnr(image1, image2)
    assert result > 0
 
def test_cosine_sim(images):
    image1, image2 = images
    result = scoring.cosine_sim(image1, image2)
    assert 0 <= result <= 1

def test_combined_similarity_score(images):
    image1, image2 = images
    score = scoring.combined_similarity_score(image1, image2, 0.4, 0.3, 0.3)
    assert score >= 0
 
# def test_compare_images(images):
#     best_matches = scoring.compare_images(images, images, 0.4, 0.3, 0.3)
#     assert len(best_matches) == len(images)
 
if __name__ == '__main__':
    pytest.main()
 