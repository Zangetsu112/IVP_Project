import matplotlib.pyplot as plt
import numpy as np
import skimage


class PreprocessingFunctions:
    def __init__(self, image_channel):
        self.image = image_channel
        self.relativeFilter = np.zeros((7, 7), int)

        # Point Spread Function assuming noise is random
        self.psf = np.ones((5, 5)) / 25

        # Initiating the filter required for Multi-Resolution Regression
        for i in range(7):
            for j in range(7):
                self.relativeFilter[i][j] = max(abs(3 - i), abs(3 - j))

    def wiener_difference(self):
        filtered_image = skimage.restoration.wiener(self.image, self.psf, 1100)
        return filtered_image

    def compute_fingerprint(self):
        smoothened_image = self.wiener_difference()
        return abs(smoothened_image - self.image)

    def multi_resolution_regression_filter(self):
        row, col = self.image.shape
        padded_image = np.zeros((row + 6, col + 6))
        padded_image[3:-3, 3:-3] = self.compute_fingerprint()
        for i in range(3, row + 3):
            for j in range(3, col + 3):
                res = np.multiply(self.relativeFilter, padded_image[i - 3: i + 4, j - 3:j + 4])
                res = np.sum(res)
                self.image[i - 3][j - 3] = res

        # Normalizing Image
        maxi, mini = 0, float('inf')
        for i in range(self.image.shape[0]):
            maxi = max(maxi, max(self.image[i]))
            mini = min(mini, min(self.image[i]))
        norm = maxi - mini
        self.image -= mini
        self.image /= norm


if __name__ == '__main__':
    image = skimage.color.rgb2gray(skimage.data.astronaut())
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    image_processing_object = PreprocessingFunctions(image)

    plt.subplot(2, 2, 2)
    plt.imshow(image_processing_object.wiener_difference(), cmap='gray')
    plt.title('Wiener Smoothened')

    plt.subplot(2, 2, 3)
    plt.imshow(image_processing_object.compute_fingerprint(), cmap='gray')
    plt.title('Image Fingerprint')

    plt.subplot(2, 2, 4)
    image_processing_object.multi_resolution_regression_filter()
    plt.imshow(image_processing_object.image, cmap='gray')
    plt.title('Relative Intensity Amplified')
    plt.show()
