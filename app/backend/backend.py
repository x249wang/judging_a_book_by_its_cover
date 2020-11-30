import asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from PIL import Image

import config
from data_processing import is_valid_image, process_image
from model import CNNModel


app = FastAPI(
    title="Judging a Book by Its Cover",
    description="CNN model for predicting a book's rating based solely on its cover image",
    version="1.0",
)


@app.on_event("startup")
async def load_model():

    global model
    model = CNNModel()

    checkpoint = torch.load(config.CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()


@app.get("/")
async def root():
    return {"message": "Welcome to the app!"}


@app.post("/judge")
async def predict_book_rating(file: UploadFile = File(...)):

    image = Image.open(file.file)

    if not is_valid_image(image):
        raise HTTPException(status_code=400, detail="Images must be in RGB mode")

    image_processed = process_image(image)
    model_output = model(image_processed)
    model_output = torch.clamp(model_output, min=1.0, max=5.0)

    predicted_rating = round(float(model_output.detach()), 1)
    response_data = {"predicted_book_rating": predicted_rating}

    return JSONResponse(content=response_data, status_code=200)


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8080)
