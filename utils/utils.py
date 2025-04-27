import os
import cv2
import numpy as np


def load_data(data_dir, target_size=(250, 250)):
    # Initialize empty lists to store images and labels
    data = []
    labels = []

    # Check if the directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist")

    categories = os.listdir(data_dir)
    if not categories:
        raise ValueError(f"No categories found in {data_dir}")

    for category in categories:
        # Path to the category directory
        category_dir = os.path.join(data_dir, category)
        if not os.path.isdir(category_dir):
            continue

        # Get the label (0 for 'nofire' and 1 for 'fire')
        label = 1 if category == 'fire' else 0

        for img_file in os.listdir(category_dir):
            # Read the image
            img_path = os.path.join(category_dir, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue

                # Resize the image to the target size
                img = cv2.resize(img, target_size)
                # Normalize the image pixels to be in the range [0, 1]
                img = img.astype('float32') / 255.0
                # Append the image and label to the data lists
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue

    if not data:
        raise ValueError("No valid images found in the dataset")

    return np.array(data), np.array(labels)