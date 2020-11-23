data_folder = "ml/data_collection/data/raw_images"

val_prop = 0.15
test_prop = 0.15

num_epochs = 2
batch_size = 4
checkpoint_path = "ml/model_training/checkpoints"
model_path_pattern = "model_ft_epoch_{}_{}.pth"
seed = 1

tensorboard_path = "ml/model_training/runs"
summary_writer_path = f"{tensorboard_path}/resnet18_finetune_experiment"

training_data_path = "ml/model_training/data/train"
validation_data_path = "ml/model_training//data/validation"
test_data_path = "ml/model_training/data/test"
