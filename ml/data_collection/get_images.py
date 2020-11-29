import os
import urllib.request
import pandas as pd
from PIL import Image
import ml.data_collection.config as config
from ml.logging import logger


def create_filename(base_filename, file_id, target, source_url):
    file_ext = os.path.splitext(source_url)[-1]
    return f"{base_filename}_{file_id}_{target}{file_ext}"


def download_file(source_url, dest_filename):
    urllib.request.urlretrieve(source_url, dest_filename)


if __name__ == "__main__":

    if not os.path.exists(config.images_path):
        os.mkdir(config.images_path)

    # Load book metadata
    book_data = pd.read_csv(config.raw_table_data_path)
    book_data["image_mode"] = ""

    # Download book cover images, keeping only RGB format ones
    logger.info("Started downloading book cover images")

    for index, row in book_data.iterrows():

        # if index % 1000 == 0: print(index)

        book_id = row["book_id"]
        image_url = row["image_url"]
        book_rating = row["average_rating"]

        image_filename = create_filename(
            f"{config.images_path}/cover", book_id, book_rating, image_url
        )

        download_file(image_url, image_filename)

        image = Image.open(image_filename)
        image_mode = image.mode

        book_data.loc[index, "image_mode"] = image_mode

        if image_mode not in ["RGB", "RGBA"]:
            os.remove(image_filename)

        book_data.to_csv(config.final_table_data_path, index=False)

    num_images = len(os.listdir(config.images_path))
    logger.info(f"Downloaded {num_images} book cover images")
