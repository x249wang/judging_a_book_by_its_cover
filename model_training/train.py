# Reference: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#finetuning-the-convnet

import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from dataset import BookcoverDataset
from split_data import get_file_paths
from transfer_learning import image_transforms
from model import CNNModel


# Load data
train_image_paths = get_file_paths("../data_collection/data/train")
validation_image_paths = get_file_paths("../data_collection/data/validation")
test_image_paths = get_file_paths("../data_collection/data/test")

train_dataset = BookcoverDataset(train_image_paths, transforms=image_transforms)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

validation_dataset = BookcoverDataset(
    validation_image_paths, transforms=image_transforms
)
validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)

test_dataset = BookcoverDataset(test_image_paths, transforms=image_transforms)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size)


# Set up model
torch.manual_seed(config.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = CNNModel()

criterion = nn.MSELoss()
optimizer = optim.Adam(model_ft.parameters())


# Model training
start = time.time()

for epoch in range(config.num_epochs):

    print("Epoch {}/{}:".format(epoch + 1, config.num_epochs))
    train_running_loss = 0.0

    for idx, batch in enumerate(train_loader):

        if idx % 100 == 0:
            print(f"Batch {idx}")

        model_ft.train()
        optimizer.zero_grad()

        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        outputs = model_ft(inputs)
        train_loss = criterion(outputs, labels)

        train_loss.backward()
        optimizer.step()

        train_running_loss += train_loss.item() * inputs.size(0)

        if idx > 0 and idx % 500 == 0:

            model_ft.eval()
            val_running_loss = 0.0

            with torch.no_grad():
                for inputs, labels in validation_loader:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model_ft(inputs)
                    val_loss = criterion(outputs, labels)

                    val_running_loss += val_loss.item() * inputs.size(0)

            val_epoch_loss = val_running_loss / len(validation_loader.dataset)  # 0.1868
            print(f"Epoch {epoch} batch {idx} validation loss: {val_epoch_loss}")

    train_epoch_loss = train_running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch} training loss: {train_epoch_loss}")

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_ft.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_epoch_loss": train_epoch_loss,
            "val_epoch_loss": val_epoch_loss,
        },
        config.model_path.format(epoch, current_datetime),
    )

end = time.time()

time_elapsed = end - start
print(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")


# Model evaluation
model_ft.eval()
test_running_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_ft(inputs)
        test_loss = criterion(outputs, labels)

        test_running_loss += test_loss.item() * inputs.size(0)

test_running_loss /= len(test_loader.dataset)
print(test_running_loss)
# 0.1885
