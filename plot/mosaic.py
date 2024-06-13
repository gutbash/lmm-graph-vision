from PIL import Image
import numpy as np
import os
import random

# Modified to load images from multiple folders
def load_images_from_multiple_folders(folders):
    images = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
                img_path = os.path.join(folder, filename)
                images.append(Image.open(img_path))
    return images

# Load all images from a folder
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
            img_path = os.path.join(folder, filename)
            images.append(Image.open(img_path))
    return images

# Create a scattered mosaic
def create_mosaic(images, canvas_size):
    canvas = Image.new('RGB', canvas_size, (255, 255, 255))  # Create a white canvas
    for img in images:
        # Random range size of the image
        random_size = random.randint(100, 300)
        img = img.resize((random_size, random_size))
        
        # Random position for each image
        max_x = canvas_size[0] - img.width
        max_y = canvas_size[1] - img.height
        if max_x > 0 and max_y > 0:  # Ensure the image can fit on the canvas
            rand_x = random.randint(0, max_x)
            rand_y = random.randint(0, max_y)
            canvas.paste(img, (rand_x, rand_y))
    return canvas

# Define your folder path and canvas size
path_1 = 'images/binary_search_tree'
path_2 = 'images/binary_tree'
path_3 = 'images/directed_graph'
path_4 = 'images/undirected_graph'

paths = [path_1, path_2, path_3, path_4]

canvas_size = (8000, 1000)  # Example canvas size

images = load_images_from_multiple_folders(paths)
mosaic = create_mosaic(images, canvas_size)
mosaic.save('mosaic.png')