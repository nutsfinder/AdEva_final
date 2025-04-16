import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from AdEva_final.config.configuration import Configuration

class SimpleTransformer(nn.Module):
    # define your encoder/decoder layers...
    pass

class ModelTrainer:
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def train(self, X_train, y_train):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleTransformer(...).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.params["learning_rate"])

        ds = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        loader = DataLoader(ds, batch_size=self.params["batch_size"], shuffle=True)

        for epoch in range(self.params["epochs"]):
            model.train()
            epoch_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred.squeeze(), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}: loss={epoch_loss/len(loader):.4f}")

        os.makedirs(self.config.model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.config.model_dir, self.config.model_name))
        return model
