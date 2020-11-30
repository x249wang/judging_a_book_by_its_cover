from time import perf_counter
import numpy as np
from fastapi.testclient import TestClient

from .backend import app


def test_welcome_page():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the app!"}


def test_predict_on_invalid_image():
    filename = "fixtures/black_white_image.jpg"

    with TestClient(app) as client:

        response = client.post("/judge", files={"file": open(filename, "rb")})

        assert response.status_code == 400
        assert response.json() == {"detail": "Images must be in RGB mode"}


def test_predict_on_valid_image():
    filename = "fixtures/rgb_book_cover.jpg"

    with TestClient(app) as client:

        response = client.post("/judge", files={"file": open(filename, "rb")})

        assert response.status_code == 200
        assert response.json()["predicted_book_rating"] >= 1
        assert response.json()["predicted_book_rating"] <= 5


def test_prediction_latency():
    n_repeats = 200
    filename = "fixtures/rgb_book_cover.jpg"
    latency_array = []

    with TestClient(app) as client:

        for ii in range(n_repeats):
            start = perf_counter()
            _ = client.post("/judge", files={"file": open(filename, "rb")})
            end = perf_counter()
            latency_array.append(end - start)

    latency_p99 = np.quantile(latency_array, 0.99)

    assert latency_p99 <= 1, "Serving latency at 99th percentile should be <= 1 sec"
