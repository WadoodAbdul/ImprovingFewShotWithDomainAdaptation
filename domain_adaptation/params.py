"""Params for ADDA."""

# params for dataset and data loader
data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50
image_size = 64


# params for training network
num_gpu = 1
num_epochs_pre = 3
log_step_pre = 30
eval_step_pre = 5
save_step_pre = 5
num_epochs = 10
log_step = 30
save_step = 1
eval_step = 150
manual_seed = None
num_epochs_classifier = 20
num_epochs_encoder = 1

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
