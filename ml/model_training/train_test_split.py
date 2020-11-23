import re
import os
import random
from shutil import copyfile
import ml.model_training.config as config
from ml.logging import logger


def train_val_test_split(image_paths, val_prop=0.15, test_prop=0.15):

    for image_path in image_paths:
        random_num = random.random()

        if random_num < val_prop:
            new_image_path = re.sub(
                config.data_folder, config.validation_data_path, image_path
            )
        elif random_num < val_prop + test_prop:
            new_image_path = re.sub(
                config.data_folder, config.test_data_path, image_path
            )
        else:
            new_image_path = re.sub(
                config.data_folder, config.training_data_path, image_path
            )

        copyfile(image_path, new_image_path)


def get_file_paths(directory):

    file_paths = []

    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))

    return file_paths


if __name__ == "__main__":

    if not os.path.exists(config.training_data_path):
        os.makedir(config.training_data_path)

    if not os.path.exists(config.validation_data_path):
        os.makedir(config.validation_data_path)

    if not os.path.exists(config.test_data_path):
        os.makedir(config.test_data_path)

    image_paths = get_file_paths(config.data_folder)
    logger.info(f"Number of images in {config.data_folder}: {len(image_paths)}")

    train_val_test_split(
        image_paths, val_prop=config.val_prop, test_prop=config.test_prop
    )

    training_prop = 1 - config.val_prop - config.test_prop
    training_folder = re.sub(
        config.data_folder, config.training_data_path, config.data_folder
    )
    validation_folder = re.sub(
        config.data_folder, config.validation_data_path, config.data_folder
    )
    test_folder = re.sub(config.data_folder, config.test_data_path, config.data_folder)

    logger.info(
        f"Divided images according to "
        f"{training_prop*100}%/{config.val_prop*100}%/{config.test_prop*100}% "
        f"train/val/test split"
    )
    logger.info(f"Training data saved to {training_folder}")
    logger.info(f"Validation data saved to {validation_folder}")
    logger.info(f"Test data saved to {test_folder}")
