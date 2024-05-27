import torch
import os
from .config import SAVE_INTERVAL, STATS_TRACK_INTERVAL
from .model_manager import ModelManager

class Trainer():

    def __init__(self, device, generator, model_num=None, epoch_num=None):
        self.generator = generator
        self.device = device
        
        # Initialise model
        self.model_manager = ModelManager(device=device)
        if model_num is not None and epoch_num is not None:
            self.config, self.model, self.optimiser, self.hyperparams, self.checkpoint_directory = self.model_manager.load_model(
                                                                                                                                model_num=model_num, 
                                                                                                                                epoch_num=epoch_num
                                                                                                                                )
        else:
            self.config, self.model, self.optimiser, self.hyperparams, self.checkpoint_directory = self.model_manager.initialise_model()

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
    
        for _ in range(total_epochs):
            train_running_mse_loss += self.train(
                                                all_inputs=train_inputs,
                                                all_targets=train_targets,
                                                losses_list=self.config["stats"]["train_losses"]
                                                )
            val_running_mse_loss += self.evaluate(
                                        all_inputs=val_inputs,
                                        all_targets=val_targets,
                                        losses_list=self.config["stats"]["val_losses"]
                                        )
            
            # Shuffle batches
            random_indexes = torch.randperm(train_inputs.shape[0], device=self.device, generator=self.generator)
            train_inputs = train_inputs[random_indexes]
            train_targets = train_targets[random_indexes]

            # Display stats
            if self.config["misc"]["current_epoch"] % STATS_TRACK_INTERVAL == 0:
                train_mse_loss_running = train_running_mse_loss / len(self.config["stats"]["train_losses"])
                val_mse_loss_running = val_running_mse_loss / len(self.config["stats"]["val_losses"])
                train_rmse_loss_running = train_mse_loss_running ** 0.5
                val_rmse_loss_running = val_mse_loss_running ** 0.5

                print(f"Epoch: {self.config["misc"]["current_epoch"]}/{total_epochs} | T_MSE: {train_mse_loss_running} | T_RMSE: {train_rmse_loss_running} | V_MSE: {val_mse_loss_running} | V_RMSE: {val_rmse_loss_running}")
            
            # Save model
            if self.config["misc"]["current_epoch"] % SAVE_INTERVAL == 0 and self.config["misc"]["current_epoch"] != 0:
                self.save_model()

            # Update epoch
            self.config["misc"]["current_epoch"] += 1


    def save_model(self):
        # Update config
        self.config["model"]["model_state_dict"] = self.model.state_dict()
        self.config["model"]["optimiser_state_dict"] = self.optimiser.state_dict()

        torch.save(self.config, f"{self.checkpoint_directory}/{self.config["misc"]["current_epoch"]}.pt") 