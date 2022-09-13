import csv
import cv2
import random
import numpy as np
from pathlib import Path
from forgery_functions import extract_splice, add_splice


image_paths = []
with open('./mias-dataset.csv') as file:
    reader = csv.reader(file)
    for row in reader:
        image_paths.append(row[0])

"""
    Dataset: 322 Images from MIAS Dataset
        - 322 Pristine Images
        - 870 Forged Images
            - 322 Splicing Forgery
            - 113 Copy-move Forgery
            - 322 Multiple splices in one image
            - 113 Multiple copy moves in one image
    Splices Extracted from images to be Copy Move forged are selected
    randomly and used to do splicing forgery in spliced images set.
"""

generated_splices = []
clipped_images = []
for i in range(322):
    image = cv2.imread(image_paths[index])
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, splice = extract_splice(gray_image)
    if _:
        generated_splices.append(splice)
        clipped_images.append(image_paths[index])

image_and_splice = list(zip(clipped_images, generated_splices))
print("Completed splice generation...")

def get_forged(image, splice, forge_type='copy_move', rng=100):
    if forge_type == 'splicing':
        splice = random.choice(generated_splices)
    _, forged_image = add_splice(image, splice, rng)
    while not _:
        rng += random.randint(-50, 50)
        if forge_type == 'splicing':
            splice = random.choice(generated_splices)
        _, forged_image = add_splice(image, splice, rng)
    return forged_image


# Copy Move forgeries
forged_path = Path('./Forged/CopyMove')
for path, splice in image_and_splice:
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    forged_image = get_forged(gray_image, splice)
    cv2.imwrite(str(forged_path/path[11:]), forged_image)
print("Finished with copy move forgery...")

# Multiple copy move forgeries in one image
forged_path = Path('./Forged/MultipleCopyMove')
for path, splice in image_and_splice:
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    forged_image = get_forged(gray_image, splice)
    more_splices = random.randint(1, 3)
    for next_splice in range(more_splices):
        forged_image = get_forged(forged_image, splice)
    cv2.imwrite(str(forged_path/path[11:]), forged_image)
print("Finished with Multiple Copy Move Forgery...")

# Splicing Forgery
forged_path = Path('./Forged/Splicing')
for path in to_splice:
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    forged_image = get_forged(gray_image, None, forge_type='splicing')
    cv2.imwrite(str(forged_path/path[11:]), forged_image)
print("Finished with Spicing Forgery...")

# Multiple Splicing Forgery 
forged_path = Path('./Forged/MultipleSplicing')
for path in multiple_splice:
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    forged_image = get_forged(gray_image, None, forge_type='splicing')
    more_splices = random.randint(1, 3)
    for next_splice in range(1, 3):
        forged_image = get_forged(forged_image, None, forge_type='splicing')
    cv2.imwrite(str(forged_path/path[11:]), forged_image)
print('Finished with multiple splicing...')
