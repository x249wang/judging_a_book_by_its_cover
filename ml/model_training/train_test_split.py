import re
import os
import random
from shutil import copyfile
import config


def train_val_test_split(image_paths, val_prop=0.15, test_prop=0.15):

    for image_path in image_paths:
        random_num = random.random()

        if random_num < val_prop:
            new_image_path = re.sub("raw_images", "validation", image_path)
        elif random_num < val_prop + test_prop:
            new_image_path = re.sub("raw_images", "test", image_path)
        else:
            new_image_path = re.sub("raw_images", "train", image_path)

        copyfile(image_path, new_image_path)


def get_file_paths(directory):

    file_paths = []

    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))

    return file_paths


if __name__ == "__main__":

    image_paths = get_file_paths(config.data_folder)
    train_val_test_split(
        image_paths, val_prop=config.val_prop, test_prop=config.test_prop
    )
