import pickle
import time
import json
import pandas as pd
from goodreads import client
from goodreads.request import GoodreadsRequestException


N_BOOKS = 100000


def retrieve_book_details(client, book_id):

    book_result = client.book(book_id)
    return {
        "title": book_result.title,
        "image_url": book_result.image_url,
        "average_rating": book_result.average_rating,
        "ratings_count": book_result.ratings_count,
    }


if __name__ == "__main__":

    with open("credentials.json") as f:
        credentials = json.load(f)

    goodreads_client = client.GoodreadsClient(credentials["key"], credentials["secret"])

    book_data = {}
    book_id = 0

    while len(book_data) < N_BOOKS:

        book_id += 1
        time.sleep(1)

        # if book_id % 1000 == 0:
        #     print(book_id)

        #     with open('data/book_data.pkl', 'wb') as output:
        #         pickle.dump(book_data, output)

        try:
            book_data[book_id] = retrieve_book_details(goodreads_client, book_id)

        except GoodreadsRequestException:
            pass

        except Exception as e:
            print(e)
            pass

    with open("data/book_data.pkl", "wb") as output:
        pickle.dump(book_data, output)

    book_data_table = (
        pd.DataFrame.from_dict(book_data, orient="index")
        .reset_index(drop=False)
        .rename(columns={"index": "book_id"})
    )

    book_data_table[
        (book_data_table.ratings_count.astype(int) > 10)
        & (~book_data_table.image_url.str.contains("nophoto"))
    ].to_csv("data/book_data.csv", index=False)
