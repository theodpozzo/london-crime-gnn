import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import numpy as np

from utils.logs import log_results
from utils.results import display_metrics, plot_predicted_vs_actual, plot_residuals

def train_specific_node(model, optimizer, data, node_index, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out[node_index], data.y.squeeze(0))
        loss.backward()
        optimizer.step()
        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

@torch.no_grad()
def test_specific_node(model, data, node_index):
    model.eval()
    pred = model(data.x, data.edge_index)
    y_true = data.y.squeeze(0).cpu().numpy()
    y_pred = pred[node_index].cpu().numpy()
        
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    ev_score = explained_variance_score(y_true, y_pred)
    
    # You might want to keep the original PyTorch MSE calculation for consistency
    torch_mse = F.mse_loss(pred, data.y).item()

    return {
        'mse': mse,
        'torch_mse': torch_mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'ev_score': ev_score,
        'y_true': y_true,
        'y_pred': y_pred
    }


def train_and_test_specific_node(data, model_class, out_channels, node_index, 
                                 epochs=100, 
                                 lr=0.01, 
                                 weight_decay=5e-4, 
                                 hidden_channels=64, 
                                 log_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_class(data.num_features, hidden_channels, out_channels).to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    # print(f"Training the {model_class.__name__} model...")
    train_specific_node(model, optimizer, data, node_index, epochs=epochs)
    
    # print(f"Testing the {model_class.__name__} model...")
    test_results = test_specific_node(model, data, node_index)
    # print(test_results)
    
    if log_file:
        log_results(log_file, model_class.__name__, lr, epochs, weight_decay, hidden_channels, test_results)

    # display_metrics(test_results, model_class.__name__)
    
    return test_results

def train(model, optimizer, train_data, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = F.mse_loss(out, train_data.y)
        loss.backward()
        optimizer.step()
        # if epoch % (epochs // 10) == 0 or epoch == epochs-1:
        #     print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

@torch.no_grad()
def test(model, test_data):
    model.eval()
    pred = model(test_data.x, test_data.edge_index)
    
    # Convert tensors to numpy arrays for sklearn metrics
    y_true = test_data.y.cpu().numpy()
    y_pred = pred.cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    ev_score = explained_variance_score(y_true, y_pred)
    
    # You might want to keep the original PyTorch MSE calculation for consistency
    torch_mse = F.mse_loss(pred, test_data.y).item()

    return {
        'mse': mse,
        'torch_mse': torch_mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'ev_score': ev_score,
        'y_true': y_true,
        'y_pred': y_pred
    }

def train_and_test(data, model_class, out_channels,
                   epochs = 100, 
                   lr = 0.01, 
                   weight_decay = 5e-4, 
                   hidden_channels = 64, 
                   log_file=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model_class(in_channels = data.x.shape[1], 
                        out_channels = out_channels,
                        hidden_channels = hidden_channels
                        ).to(device)
    data = data.to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # print(f"Training the {model_class.__name__} model...")
    train(model, optimizer, data, epochs=epochs)
    
    # print(f"Testing the {model_class.__name__} model...")
    test_results = test(model, data)
    
    if log_file:
        log_results(log_file, model_class.__name__, lr, epochs, weight_decay, hidden_channels, test_results)

    # display_metrics(test_results, model_class.__name__)
    
    return test_results