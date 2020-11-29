# Reference: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#finetuning-the-convnet
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import ml.model_training.config as config
from ml.model_training.dataset import BookcoverDataset
from ml.model_training.train_test_split import get_file_paths
from ml.model_training.transformations import image_transforms
from ml.model_training.model import CNNModel
from ml.logging import logger


if __name__ == "__main__":

    if not os.path.exists(config.checkpoint_path):
        os.makedir(config.checkpoint_path)

    if not os.path.exists(config.tensorboard_path):
        os.makedir(config.tensorboard_path)

    # Load data
    train_image_paths = get_file_paths(config.training_data_path)
    validation_image_paths = get_file_paths(config.validation_data_path)
    test_image_paths = get_file_paths(config.test_data_path)

    train_dataset = BookcoverDataset(train_image_paths, transforms=image_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    logger.info(
        f"Loaded {len(train_loader.dataset)} images as training data "
        f"from {config.training_data_path}, in batches of {config.batch_size}"
    )

    validation_dataset = BookcoverDataset(
        validation_image_paths, transforms=image_transforms
    )
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
    logger.info(
        f"Loaded {len(validation_loader.dataset)} images as validation data "
        f"from {config.validation_data_path}, in batches of {config.batch_size}"
    )

    test_dataset = BookcoverDataset(test_image_paths, transforms=image_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    logger.info(
        f"Loaded {len(test_loader.dataset)} images as test data "
        f"from {config.test_data_path}, in batches of {config.batch_size}"
    )

    # Set up model
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = CNNModel(
        fc_hidden_dim=config.fc_hidden_dim,
        dropout=config.dropout,
        freeze_layers=config.freeze_layers,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_ft.parameters())

    writer = SummaryWriter(config.summary_writer_path)
    # tensorboard --logdir=runs

    logger.info(f"Device: {device}; epochs: {config.num_epochs}")
    logger.info(f"Tensorboard log written to: {config.summary_writer_path}")

    # Model training with periodic validation
    start = time.time()
    logger.info(
        "Training started. For training/val loss records, refer to tensorboard logs"
    )

    for epoch in range(config.num_epochs):

        print("Epoch {}/{}:".format(epoch + 1, config.num_epochs))
        train_running_loss, samples_seen = 0.0, 0

        for idx, batch in enumerate(train_loader):

            batch_num = epoch * len(train_loader) + idx

            model_ft.train()
            optimizer.zero_grad()

            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            outputs = model_ft(inputs)
            train_loss = criterion(outputs, labels)

            train_loss.backward()
            optimizer.step()

            train_running_loss += train_loss.item() * inputs.size(0)
            samples_seen += inputs.size(0)

            if idx > 0 and idx % config.eval_every == 0:

                print(
                    f"Epoch {epoch} batch {idx} training loss: {train_running_loss / samples_seen}"
                )
                writer.add_scalar(
                    "training loss", train_running_loss / samples_seen, batch_num
                )

                model_ft.eval()
                val_running_loss = 0.0

                with torch.no_grad():
                    for inputs, labels in validation_loader:

                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model_ft(inputs)
                        val_loss = criterion(outputs, labels)

                        val_running_loss += val_loss.item() * inputs.size(0)

                val_epoch_loss = val_running_loss / len(validation_loader.dataset)
                print(f"Epoch {epoch} batch {idx} validation loss: {val_epoch_loss}")
                writer.add_scalar("validation loss", val_epoch_loss, batch_num)

        train_epoch_loss = train_running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} training loss: {train_epoch_loss}")
        writer.add_scalar("training loss", train_epoch_loss, batch_num)

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
        checkpoint_filename = config.model_path_pattern.format(epoch, current_datetime)
        checkpoint_path = f"{config.checkpoint_path}/{checkpoint_filename}"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_ft.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_epoch_loss": train_epoch_loss,
                "val_epoch_loss": val_epoch_loss,
            },
            checkpoint_path,
        )
        logger.info(f"Model checkpoint saved at {checkpoint_path}")

    end = time.time()

    time_elapsed = end - start
    training_time_str = f"{time_elapsed // 60}m {time_elapsed % 60}s"
    print(f"Training completed in {training_time_str}")
    logger.info(f"Training complete, which took {training_time_str}")

    writer.close()

    # Model evaluation
    model_ft.eval()
    test_running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            outputs = torch.clamp(outputs, min=1.0, max=5.0)
            test_loss = criterion(outputs, labels)

            test_running_loss += test_loss.item() * inputs.size(0)

    test_loss = test_running_loss / len(test_loader.dataset)
    print(f"Test set loss: {test_loss}")
    logger.info(f"Test set loss: {test_loss}")
