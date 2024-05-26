import torch
from .model import Model

class Trainer():

    def __init__(self, learning_rate, device):
        self.device = device
        self.initialise_model(learning_rate=learning_rate, device=device)

    def initialise_model(self, learning_rate, device):
        self.model = Model().to(device)
        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train(self, X_train, Y_train, batch_size, total_epochs):
        self.model.train()

        running_mse_loss = 0.0
        batches = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        target_batches = torch.tensor(Y_train.values, dtype=torch.float32).reshape(-1, 1).to(self.device)
        total_steps = 1

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
                
                total_steps += 1
            
            mse_loss_running = (running_mse_loss / total_steps)
            rmse_loss_running = mse_loss_running ** 0.5
            print(f"Epoch: {epoch + 1}/{total_epochs} | MSELoss: {mse_loss_running} | RMSELoss: {rmse_loss_running}")


    def evaluate(self, X_val, Y_val, batch_size):
        self.model.eval()

        running_mse_loss = 0.0
        val_batches = torch.tensor(X_val.values, dtype=torch.float32).to(self.device)
        val_targets = torch.tensor(Y_val.values, dtype=torch.float32).reshape(-1, 1).to(self.device)

        total_steps = 1
        for i in range(0, val_batches.shape[0] - batch_size, batch_size):
            batch = val_batches[i:i+batch_size]
            targets = val_targets[i:i+batch_size]

            preds = self.model(batch)
            loss = torch.nn.functional.mse_loss(preds, targets)
            running_mse_loss += loss.item()

            total_steps += 1
        
        rmse_score = (running_mse_loss / total_steps) ** 0.5
        print(f"Local RMSE (NN): {rmse_score}")

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