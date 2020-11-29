data_folder = "ml/data_collection/data/raw_images"

val_prop = 0.15
test_prop = 0.15

fc_hidden_dim = 256
dropout = 0.1
freeze_layers = True

num_epochs = 2
batch_size = 2 ** 3
seed = 1

eval_every = 100

model_training_path = "ml/model_training"

checkpoint_path = f"{model_training_path}/checkpoints"
model_path_pattern = "model_ft_epoch_{}_{}.pth"

tensorboard_path = f"{model_training_path}/runs"
summary_writer_path = f"{tensorboard_path}/resnet18_finetune_experiment"

training_data_path = f"{model_training_path}/data/train"
validation_data_path = f"{model_training_path}//data/validation"
test_data_path = f"{model_training_path}/data/test"
