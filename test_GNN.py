import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import utils
from utils import Config
import os
from train import *
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings(
    "ignore",
    message="Deterministic behavior was enabled with either `torch.use_deterministic_algorithms",
)

cfg = Config()
cfg.optimizer.lr = 0.003754
cfg.optimizer.weight_decay = 1.7698e-6
cfg.training.batch_size = 16
cfg.transformer.d_model = 64
cfg.transformer.num_encoder_layers = 4
cfg.transformer.num_decoder_layers = 2
cfg.transformer.dropout = 0.18155

utils.set_seed(cfg.training.seed)
    
train_loader, val_loader, test_loader, scaler = get_dataloaders(cfg)
gnn_transformer = build_model(cfg)

print("Model:", gnn_transformer)
num_params = sum(p.numel() for p in gnn_transformer.parameters())
print(f'Number of Parameters: {num_params}')

trainer = Trainer(
    model=gnn_transformer,
    loss_fn = masked_mse_loss,
    train_loader=train_loader,
    val_loader=val_loader,
    cfg=cfg,

)
trained_gnn_transformer, transformer_history = trainer.fit()

model_folder = os.path.join(cfg.dataset.saves_dir, 'Models/')
model_path = os.path.join(model_folder, 'ss_gnn_opt_smooth.pth')
scaler_folder = os.path.join(cfg.dataset.saves_dir, 'Scalers/')
scaler_path = os.path.join(scaler_folder, 'ss_gnn_opt_smooth.pkl')

torch.save(trained_gnn_transformer, model_path)
scaler.save(scaler_path)

loss_fns = [masked_mse_loss, masked_FDE_loss]
utils.evaluate_model(
    trained_gnn_transformer,
    test_loader,
    loss_fns=loss_fns
)