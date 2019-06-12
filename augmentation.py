from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def interpolate(img, target_size):
    if img.shape == target_size:
        return img
    else:
        return resize(img, target_size, mode='reflect')

def normalize(img):
    img = (img - np.mean(img)) / (np.std(img) + 0.000000000001)
    return img

def plot_image(img):
    plt.imshow(img)
    plt.show()

def augment():
    augmentator = iaa.Sequential([
        iaa.Sometimes(0.35, iaa.GaussianBlur(sigma=(0,0.2))),
        iaa.Sometimes(0.45, iaa.ContrastNormalization(alpha=(0.9,1.1))),
        iaa.Sometimes(0.25, iaa.Multiply(mul=(0.95,1.05))),
        iaa.Sometimes(0.3, iaa.Affine(scale=(0.9,1.1))),

        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),

        iaa.OneOf([
            iaa.Affine(rotate=(90)),
            iaa.Affine(rotate=(-90)),
            iaa.Affine(rotate=(0))
        ])
    ], random_order=True)
    
    return augmentator

def pipeline(all_images, target_size):
    aug = augment()
    all_new_images = []
    for each_image in all_images:
        each_image_transformed = each_image
        each_image_transformed = interpolate(each_image_transformed, target_size)
        augmented_image = aug.augment_image(each_image_transformed)
        normalized_image = normalize(augmented_image)
        all_new_images.append(normalized_image)
    return all_new_images
