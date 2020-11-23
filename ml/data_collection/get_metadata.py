import os
import pickle
import time
import json
import pandas as pd
from goodreads import client
from goodreads.request import GoodreadsRequestException
import ml.data_collection.config as config
from ml.logging import logger


def retrieve_book_details(client, book_id):

    book_result = client.book(book_id)
    return {
        "title": book_result.title,
        "image_url": book_result.image_url,
        "average_rating": book_result.average_rating,
        "ratings_count": book_result.ratings_count,
    }


if __name__ == "__main__":

    if not os.path.exists(config.data_folder):
        os.mkdir(config.data_folder)

    with open(config.credentials_path) as f:
        credentials = json.load(f)

    goodreads_client = client.GoodreadsClient(credentials["key"], credentials["secret"])

    book_data = {}
    book_id = 0

    logger.info("Started collecting book metadata")
    while len(book_data) < config.n_books:

        book_id += 1
        time.sleep(1)

        print(book_id)

        if book_id % config.save_every == 0:
            # Save results once every while in case the program crashes
            with open(config.raw_data_path, "wb") as output:
                pickle.dump(book_data, output)

        try:
            book_data[book_id] = retrieve_book_details(goodreads_client, book_id)

        except GoodreadsRequestException:
            pass

        except Exception as e:
            print(e)
            pass

    with open(config.raw_data_path, "wb") as output:
        pickle.dump(book_data, output)
    logger.info(
        f"Saved {len(book_data)} raw results (in dict format) to {config.raw_data_path}"
    )

    book_data_table = (
        pd.DataFrame.from_dict(book_data, orient="index")
        .reset_index(drop=False)
        .rename(columns={"index": "book_id"})
    )

    book_data_table[
        (book_data_table.ratings_count.astype(int) > config.min_likes_threshold)
        & (~book_data_table.image_url.str.contains("nophoto"))
    ].to_csv(config.raw_table_data_path, index=False)
    logger.info(f"Saved results (in df format) to {config.raw_table_data_path}")
