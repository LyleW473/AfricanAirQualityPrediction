import torch
import matplotlib.pyplot as plt


model_num = 17
epoch_num = 1000
checkpoint = torch.load(f"model_checkpoints/{model_num}/{epoch_num}.pt")

train_losses = torch.tensor(checkpoint["stats"]["train_losses"])
val_losses = torch.tensor(checkpoint["stats"]["val_losses"])

# Convert from MSE to RMSE
train_losses = train_losses ** 0.5
val_losses = val_losses ** 0.5

# ------------------------------
# Visualise losses per epoch

size_per_epoch_train = len(train_losses) // epoch_num
size_per_epoch_val = len(val_losses) // epoch_num

# Get losses per epoch
train_losses_per_epoch = train_losses.reshape(-1, size_per_epoch_train).mean(dim=1)
val_losses_per_epoch = val_losses.reshape(-1, size_per_epoch_val).mean(dim=1)


print(size_per_epoch_train, size_per_epoch_val)
print(train_losses_per_epoch.shape, val_losses.shape)
print(train_losses_per_epoch[:100])

fig, ax = plt.subplots()
ax.set_title("Losses per epoch (RMSE)")
ax.plot(train_losses_per_epoch, label="Train")
ax.plot(val_losses_per_epoch, label="Validation")
ax.legend()
plt.show()

# ------------------------------
# Visualise running losses

train_indices = torch.arange(1, len(train_losses) + 1)
running_train_losses = torch.cumsum(train_losses, dim=0) / train_indices
val_indices = torch.arange(1, len(val_losses) + 1)
running_val_losses = torch.cumsum(val_losses, dim=0) / val_indices
max_size = min(len(running_train_losses), len(running_val_losses))

running_train_losses = running_train_losses[:max_size]
running_val_losses = running_val_losses[:max_size]

print(running_train_losses[:100])

fig, ax = plt.subplots()
ax.set_title("Running losses (RMSE)")
ax.plot(running_train_losses, label="Train")
ax.plot(running_val_losses, label="Validation")
ax.legend()
plt.show()