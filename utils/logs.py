import csv
import os

def initialize_log(file_path):
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Learning Rate', 'Epochs', 'Weight Decay', 'Hidden Channels', 'Test Results']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def log_results(file_path, model_name, lr, epochs, weight_decay, hidden_channels, test_results):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Learning Rate', 'Epochs', 'Weight Decay', 'Hidden Channels', 'Test Results']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'Model': model_name,
            'Learning Rate': lr,
            'Epochs': epochs,
            'Weight Decay': weight_decay,
            'Hidden Channels': hidden_channels,
            'Test Results': test_results
        })
