import os  # For directory operations
import torch  # For managing the device configuration

# Configuration class
class Config:
    def __init__(self):
        self.data_dir = "./Training_Input"
        self.label_dir = "./Ground_Truth"
        self.exp_name = "CristaeSegmentation"
        self.base_lr = 1e-4
        self.batch_size = 16
        self.epochs = 5000
        self.num_workers = 10
        self.device = torch.device('cuda')  # Use GPU if available
        self.log_dir = "./logs"
        self.output_dir = "./visual_outputs"  # Directory to save visualizations

# Create an instance of the config class
opt = Config()

# Ensure the output directory exists
os.makedirs(opt.output_dir, exist_ok=True)
