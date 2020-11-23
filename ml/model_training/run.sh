python train_test_split.py
python train.py

cp -p "`ls -dtr checkpoints/*.pth | tail -1`" ../backend/model.pth
