import torch
import os
from .model import Model
from .config import SAVE_INTERVAL, STATS_TRACK_INTERVAL

class Trainer():

    def __init__(self, learning_rate, device):
        self.device = device
        self.initialise_model(learning_rate=learning_rate, device=device)

    def initialise_model(self, learning_rate, device):
        self.model = Model().to(device)
        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train(self, all_inputs, all_targets, losses_list):
        self.model.train()

        running_mse_loss = 0.0

        num_batches = all_inputs.shape[0]
        for i in range(0, num_batches):

            self.optimiser.zero_grad()

            batch = all_inputs[i]
            targets = all_targets[i]

            preds = self.model(batch)
            loss = torch.nn.functional.mse_loss(preds, targets)
            loss.backward()
            self.optimiser.step()

            loss_item = loss.item()
            running_mse_loss += loss_item
            losses_list.append(loss_item)

        return running_mse_loss


    def evaluate(self, all_inputs, all_targets, losses_list, verbose=False):
        self.model.eval()

        running_mse_loss = 0.0
        total_steps = 1
        num_batches = all_inputs.shape[0]

        for i in range(0, num_batches):
            batch = all_inputs[i]
            targets = all_targets[i]

            preds = self.model(batch)
            loss = torch.nn.functional.mse_loss(preds, targets)

            loss_item = loss.item()
            running_mse_loss += loss_item
            losses_list.append(loss_item)
            total_steps += 1
        
        if verbose:
            rmse_score = (running_mse_loss / total_steps) ** 0.5
            print(f"Local RMSE (NN): {rmse_score}")

        return running_mse_loss

    def get_predictions_for_dataset(self, test_df, batch_size):

        self.model.eval()

        all_preds = []
        batches = torch.tensor(test_df.values, dtype=torch.float32).to(self.device)

        for i in range(0, test_df.shape[0] - batch_size, batch_size):
            batch = batches[i:i+batch_size]
            preds = self.model(batch)
            all_preds.append(preds)
        
        # Get the last batch, adding padding
        last_batch = batches[-(batches.shape[0] % batch_size):]
        padding = torch.zeros((batch_size - last_batch.shape[0], last_batch.shape[1])).to(self.device)
        last_batch = torch.cat([last_batch, padding], dim=0)
        preds = self.model(last_batch)[:batch_size - padding.shape[0]]
        all_preds.append(preds)

        # Concatenate all the predictions
        all_preds = torch.cat(all_preds, dim=0).reshape(-1)
        return all_preds.cpu().detach().numpy() # Convert to numpy array for submission
    
    def execute(self, train_inputs, train_targets, val_inputs, val_targets, total_epochs):

        train_running_mse_loss = 0.0
        val_running_mse_loss = 0.0

        train_losses = []
        val_losses = []

        for epoch in range(total_epochs):
            train_running_mse_loss += self.train(
                                                all_inputs=train_inputs,
                                                all_targets=train_targets,
                                                losses_list=train_losses
                                                )
            val_running_mse_loss += self.evaluate(
                                        all_inputs=val_inputs,
                                        all_targets=val_targets,
                                        losses_list=val_losses
                                        )
            
            if epoch % STATS_TRACK_INTERVAL == 0:
                # Display stats
                train_mse_loss_running = train_running_mse_loss / len(train_losses)
                val_mse_loss_running = val_running_mse_loss / len(val_losses)
                train_rmse_loss_running = train_mse_loss_running ** 0.5
                val_rmse_loss_running = val_mse_loss_running ** 0.5

                print(f"Epoch: {epoch + 1}/{total_epochs} | T_MSE: {train_mse_loss_running} | T_RMSE: {train_rmse_loss_running} | V_MSE: {val_mse_loss_running} | V_RMSE: {val_rmse_loss_running}")
            
            if epoch % SAVE_INTERVAL == 0:
                self.save_model(train_losses, val_losses)

    def save_model(self, train_losses, val_losses):

        checkpoint = {
                    "model": {
                            "model_state_dict": self.model.state_dict(),
                            "optimiser_state_dict": self.optimiser.state_dict()
                            },
                    "stats":
                            {
                            "train_losses": train_losses,
                            "val_losses": val_losses
                            }
                    }
        if not os.path.exists("model_checkpoints"):
            os.makedirs("model_checkpoints")
        model_number = len(os.listdir("model_checkpoints"))
        torch.save(checkpoint, f"model_checkpoints/model_{model_number}.pt")