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
cfg.optimizer.lr = 2.575e-4
cfg.optimizer.weight_decay = 1.917e-5
cfg.training.batch_size = 32
cfg.training.delta = 0.74549
cfg.training.time_decay = 0.03772
cfg.training.lambda_vel = 0.1# 35.2048
cfg.training.lambda_acc = 0.05# 0.0196743
cfg.transformer.d_model = 128
cfg.transformer.num_encoder_layers = 3
cfg.transformer.num_decoder_layers = 3
cfg.transformer.dropout = 0.090305

utils.set_seed(cfg.training.seed)
    
train_loader, val_loader, test_loader, scaler = get_dataloaders(cfg)
gnn_transformer = build_model(cfg)

print("Model:", gnn_transformer)
num_params = sum(p.numel() for p in gnn_transformer.parameters())
print(f'Number of Parameters: {num_params}')

criterion = maskedCriterion(cfg)

trainer = Trainer(
    model=gnn_transformer,
    train_loader=train_loader,
    val_loader=val_loader,
    cfg=cfg,

)
trained_gnn_transformer, transformer_history = trainer.fit()

model_folder = os.path.join(cfg.dataset.saves_dir, 'Models/')
model_path = os.path.join(model_folder, '1203_opt_noGNN.pth')
scaler_folder = os.path.join(cfg.dataset.saves_dir, 'Scalers/')
scaler_path = os.path.join(scaler_folder, '1203_opt_noGNN.pkl')

torch.save(trained_gnn_transformer, model_path)
scaler.save(scaler_path)

loss_fns = [masked_mse_loss, masked_FDE_loss]
utils.evaluate_model(
    trained_gnn_transformer,
    test_loader,
    loss_fns=loss_fns
)