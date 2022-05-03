from forgery_functions import extract_splice, add_splice
import numpy as np
import pickle
import csv
from scipy.sparse import csr_matrix
import pandas as pd
import cv2
import random
import matplotlib.pyplot as plt


#  def plot(subplots, r, c):
    #  plt.figure(figsize=(8,8))
    #  for i in range(1, len(subplots) + 1):
    #      plt.subplot(r, c, i)
    #      plt.imshow(subplots[i-1], cmap='gray')
    #  plt.show()

with open('./mias-dataset.csv') as file:
    image_paths = []
    reader = csv.reader(file)
    for row in reader:
        image_paths.append(row)
image_paths = [path[0] for path in image_paths]

"""
    Dataset: 322 Images from MIAS Dataset
        - 122 Pristine Images
        - 200 Forged Images
            - 50 Splicing Forgery
            - 50 Copy-move Forgery
            - 50 Multiple splices in one image
            - 50 Multiple copy moves in one image
    Splices Extracted from images to be Copy Move forged are selected
    randomly and used to do splicing forgery in spliced images set.
"""

generated_splices = []
clipped_images = []
counter, index = 100, 0
while counter > 0 and  index < 300:
    image = cv2.imread(image_paths[index])
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, splice = extract_splice(gray_image)
    if _:
        generated_splices.append(splice)
        clipped_images.append(image_paths[index])
        counter -= 1
    index +=1 

left_over = list(set(image_paths)- set(clipped_images))
image_and_splice = list(zip(clipped_images, generated_splices))

to_copy_move = clipped_images[0:50]
multiple_cm = clipped_images[50:100]
to_splice = left_over[0:50]
multiple_splice = left_over[50:100]
pristine_images = left_over[100:]

# Initiate DataFrame
training_data = pd.DataFrame(columns= ['Original', 'Forged', 'Difference','Type'])

def get_forged(image, splice, forge_type='copy_move', rng=100):
    if forge_type == 'splicing':
        splice = random.choice(generated_splices)
    _, forged_image = add_splice(image, splice, rng)
    while not _:
        rng += 20
        _, forged_image = add_splice(image, splice, rng)
    return forged_image

# Pristine Images
for image in pristine_images:
    gray_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2GRAY)
    dict_ = {
                'Original': image, 
                'Forged': gray_image, 
                'Difference': csr_matrix(np.zeros((gray_image.shape), dtype=int)), 
                'Type': 'Pristine'
            }
    training_data = training_data.append(dict_, ignore_index=True)
print('Finished with Pristine images')


# Copy Move forgeries
for path, splice in image_and_splice[0:50]:
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    forged_image = get_forged(gray_image, splice)
    difference = csr_matrix(gray_image - forged_image)
    dict_ = {
                'Original': path, 
                'Forged': forged_image, 
                'Difference': difference, 
                'Type': 'Copy-Move'
            }
    training_data = training_data.append(dict_, ignore_index=True)
print("Finished with copy move forgery...")

# Multiple copy move forgeries in one image
for path, splice in image_and_splice[50:100]:
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    forged_image = get_forged(gray_image, splice)
    more_splices = random.randint(1, 3)
    for next_splice in range(1, 3):
        forged_image = get_forged(forged_image, splice)
    difference = csr_matrix(gray_image - forged_image)
    dict_ = {
                'Original': path, 
                'Forged': forged_image, 
                'Difference': difference, 
                'Type': 'Multiple-Copy-Move'
            }
    training_data = training_data.append(dict_, ignore_index=True)
print("Finished with Multiple Copy Move Forgery...")

# Splicing Forgery
for path in to_splice:
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    forged_image = get_forged(gray_image, None, forge_type='splicing')
    difference = csr_matrix(gray_image - forged_image)
    dict_ = {
                'Original': path, 
                'Forged': forged_image, 
                'Difference': difference, 
                'Type': 'Splicing'
            }
    training_data = training_data.append(dict_, ignore_index=True)
print("Finished with Spicing Forgery...")

# Multiple Splicing Forgery 
for path in multiple_splice:
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    forged_image = get_forged(gray_image, None, forge_type='splicing')
    more_splices = random.randint(1, 3)
    for next_splice in range(1, 3):
        forged_image = get_forged(forged_image, None, forge_type='splicing')
    difference = csr_matrix(gray_image - forged_image)
    dict_ = {
                'Original': path, 
                'Forged': forged_image, 
                'Difference': difference, 
                'Type': 'Multiple-Splicing'
            }
    training_data = training_data.append(dict_, ignore_index=True)
print('Finished with multiple splicing...')

training_data.to_pickle('./training_data.pkl')
