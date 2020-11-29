# Collect data
python -m ml.data_collection.get_metadata
python -m ml.data_collection.get_images

# Train model
python -m ml.model_training.train_test_split
python -m ml.model_training.train

# Copy model artifact to web app
cp -p "`ls -dtr ml/model_training/checkpoints/*.pth | tail -1`" ./app/backend/model.pth
cp -p ml/model_training/model.py ./app/backend/model.py
