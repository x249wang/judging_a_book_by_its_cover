import os
import urllib.request
import pandas as pd
from PIL import Image


def create_filename(base_filename, file_id, target, source_url):
    file_ext = os.path.splitext(source_url)[-1]
    return f"{base_filename}_{file_id}_{target}{file_ext}"


def download_file(source_url, dest_filename):
    urllib.request.urlretrieve(source_url, dest_filename)


if __name__ == "__main__":

    book_data = pd.read_csv("data/book_data.csv")
    book_data["image_mode"] = ""

    for index, row in book_data.iterrows():

        # if index % 1000 == 0: print(index)

        book_id = row["book_id"]
        image_url = row["image_url"]
        book_rating = row["average_rating"]

        image_filename = create_filename(
            "data/raw_images/cover", book_id, book_rating, image_url
        )

        download_file(image_url, image_filename)

        image = Image.open(image_filename)
        image_mode = image.mode

        book_data.loc[index, "image_mode"] = image_mode

        if len(image_mode) < 3:
            os.remove(image_filename)

        book_data.to_csv("data/book_data.csv", index=False)
