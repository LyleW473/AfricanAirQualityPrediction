import torch
from .model import Model

class Trainer():

    def __init__(self, device):
        self.device = device
        self.initialise_model(device=device)

    def initialise_model(self, device):
        self.model = Model().to(device)
        self.optimiser = torch.optim.AdamW(self.model.parameters())

    def train(self, X_train, Y_train, batch_size, total_epochs):

        running_mse_loss = 0.0
        batches = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        target_batches = torch.tensor(Y_train.values, dtype=torch.float32).reshape(-1, 1).to(self.device)
        print(target_batches.shape)

        # Iterate over the epochs
        for epoch in range(total_epochs):
            # Iterate over the batches
            for i in range(0, batches.shape[0] - batch_size, batch_size):

                self.optimiser.zero_grad()

                batch = batches[i:i+batch_size]
                targets = target_batches[i:i+batch_size]

                preds = self.model(batch)
                loss = torch.nn.functional.mse_loss(preds, targets)
                loss.backward()
                self.optimiser.step()
                running_mse_loss += loss.item()
            
            running_rmse_loss = (running_mse_loss / (batches.shape[0] * (epoch + 1))) ** 0.5
            print(f"Epoch: {epoch + 1}/{total_epochs} | MSELoss: {running_mse_loss / (batches.shape[0] * (epoch + 1))} | RMSELoss: {running_rmse_loss}")


    def evaluate(self, X_val, Y_val, batch_size):

        running_mse_loss = 0.0
        val_batches = torch.tensor(X_val.values, dtype=torch.float32).to(self.device)
        val_targets = torch.tensor(Y_val.values, dtype=torch.float32).reshape(-1, 1).to(self.device)

        for i in range(0, val_batches.shape[0] - batch_size, batch_size):
            batch = val_batches[i:i+batch_size]
            targets = val_targets[i:i+batch_size]
            preds = self.model(batch)
            loss = torch.nn.functional.mse_loss(preds, targets)
            running_mse_loss += loss.item()

        rmse_score = (running_mse_loss / val_batches.shape[0]) ** 0.5
        print(f"Local RMSE (NN): {rmse_score}")

    def get_predictions_for_dataset(self, test_df, batch_size):
        
        all_preds = []
        batches = torch.tensor(test_df.values, dtype=torch.float32).to(self.device)
        print(batches.shape)

        for i in range(0, test_df.shape[0] - batch_size, batch_size):
            batch = batches[i:i+batch_size]
            preds = self.model(batch)
            all_preds.append(preds)
        
        # Get the last batch, adding padding
        last_batch = batches[-(batches.shape[0] % batch_size):]
        padding = torch.zeros((batch_size - last_batch.shape[0], last_batch.shape[1])).to(self.device)
        print(last_batch.shape)
        last_batch = torch.cat([last_batch, padding], dim=0)
        print(last_batch.shape)
        preds = self.model(last_batch)[:batch_size - padding.shape[0]] 
        all_preds.append(preds)

        # Concatenate all the predictions
        all_preds = torch.cat(all_preds, dim=0).reshape(-1)
        return all_preds.cpu().detach().numpy() # Convert to numpy array for submission