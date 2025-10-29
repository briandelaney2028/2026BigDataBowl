import numpy as np
import os
import time
import shutil
from copy import deepcopy
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from dataclasses import asdict
import utils
from utils import Config
from feature_engineering import engineer_features
from data_sequencing import generate_sequences_4D
from FeatureScaler import FeatureScaler
from Transformers import GNNTransformer
from train import Trainer, prepare_targets_as_deltas, collate_default, masked_mse_loss
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings(
    "ignore",
    message="Deterministic behavior was enabled with either `torch.use_deterministic_algorithms",
)

def build_model(cfg: Config) -> torch.nn.Module:
    """
    Generates an untrained model based on cfg

    Parameters:
        cfg (Config): sampled configuration

    Returns:
        (torch.nn.Module): untrained model
    """
    
    return GNNTransformer(
        cfg.dataset.input_size,
        cfg.dataset.output_size,
        cfg=cfg.transformer
    )

def get_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load data into DataLoaders

    Parameters:
        cfg (Config): sampled configuration

    Returns:
        (tuple[DataLoader, DataLoader, DataLoader]): train, val, test DataLoaders
    """

    # load data, and engineer features
    df_input, df_output, _, _ = utils.load_prediction_data(cfg.dataset)
    df_input['player_height'] = df_input['player_height'].apply(utils.height_to_inches)
    df_input = engineer_features(utils.invert_direction(df_input), cfg.dataset)
    df_output = utils.invert_direction(utils.map_play_direction(df_input, df_output))

    # get sequence data
    save_path = os.path.join(cfg.dataset.data_dir, 'Saves', 'GNNtransformer_data.npz')
    required_keys = ['X', 'y', 'player_mask', 'target_mask', 'y_mask', 'ids']
    data = utils.load_saved_data(save_path, required_keys)

    # If loading failed, generate and save new data
    if data is None:
        print('[INFO] Generating new data sequences...')
        X, y, player_mask, target_mask, y_mask, ids = generate_sequences_4D(df_input, cfg.dataset, df_output=df_output)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, X=X, y=y, player_mask=player_mask, target_mask=target_mask, y_mask=y_mask, ids=ids)
        print('[INFO] Data saved successfully to {save_path}')
    else:
        X, y, player_mask, target_mask, y_mask, ids = data['X'], data['y'], data['player_mask'], data['target_mask'], data['y_mask'], data['ids']
    print('[INFO] Data ready for use.')

    # Make Train-Test split
    X_train, X_temp, y_train, y_temp, player_mask_train, player_mask_temp, target_mask_train, target_mask_temp, y_mask_train, y_mask_temp = train_test_split(
        X, y, player_mask, target_mask, y_mask, test_size=0.3, random_state=cfg.training.seed
    )

    # Make Validation-Test split
    X_val, X_test, y_val, y_test, player_mask_val, player_mask_test, target_mask_val, target_mask_test, y_mask_val, y_mask_test = train_test_split(
        X_temp, y_temp, player_mask_temp, target_mask_temp, y_mask_temp, test_size=0.3, random_state=cfg.training.seed
    )
    # Scale Features
    scaler = FeatureScaler(feature_names=cfg.dataset.features, method='standard', angle_features=cfg.dataset.angle_features)
    X_train_scaled = scaler.fit_transform(X_train, cfg.dataset.scaled_features)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    if not cfg.dataset.target_features:
        y_train_scaled = scaler.transform(y_train, ['x', 'y'])
        y_val_scaled = scaler.transform(y_val, ['x', 'y'])
        y_test_scaled = scaler.transform(y_test, ['x', 'y'])
    else:
        y_train_scaled = scaler.transform(y_train)
        y_val_scaled = scaler.transform(y_val)
        y_test_scaled = scaler.transform(y_test)

    # Obtain last positions (B, N, 2)
    last_train_scaled = X_train_scaled[:, :, -1, :2]
    last_val_scaled = X_val_scaled[:, :, -1, :2]
    last_test_scaled = X_test_scaled[:, :, -1, :2]

    # convert y to deltas from last known position
    y_train_deltas = prepare_targets_as_deltas(y_train_scaled, last_train_scaled, player_mask_train)
    y_val_deltas = prepare_targets_as_deltas(y_val_scaled, last_val_scaled, player_mask_val)
    y_test_deltas = prepare_targets_as_deltas(y_test_scaled, last_test_scaled, player_mask_test)

    # wrap data
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_deltas, dtype=torch.float32),
        torch.tensor(player_mask_train, dtype=torch.bool),
        torch.tensor(target_mask_train, dtype=torch.bool),
        torch.tensor(y_mask_train, dtype=torch.bool)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val_deltas, dtype=torch.float32),
        torch.tensor(player_mask_val, dtype=torch.bool),
        torch.tensor(target_mask_val, dtype=torch.bool),
        torch.tensor(y_mask_val, dtype=torch.bool)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test_deltas, dtype=torch.float32),
        torch.tensor(player_mask_test, dtype=torch.bool),
        torch.tensor(target_mask_test, dtype=torch.bool),
        torch.tensor(y_mask_test, dtype=torch.bool)
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,  collate_fn=collate_default)
    val_loader   = DataLoader(  val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_default)
    test_loader  = DataLoader( test_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_default)

    return train_loader, val_loader, test_loader


class OptunaTrainer(Trainer):
    def __init__(self, *args, trial: optuna.trial.Trial = None, trial_logdir: str = None, **kwargs):
        """
        Inhereit Trainer and add Optuna reporting funcitonality

        Parameters:
            trial: optuna.trial.Trial object for reporting & pruning
            trial_logdir: custom log directory for this trial's TensorBoard
        """
        super().__init__(*args, **kwargs)
        self.trial = trial
        self.trial_logdir = trial_logdir

        # if a custom logdir is provided, reinitialize the writer so logs go to trail folder
        if trial_logdir is not None:
            try:
                self.writer.close()
            except Exception:
                pass
            self.writer = SummaryWriter(log_dir=trial_logdir)
            self.log_dir = self.writer.log_dir

    def fit(self, report_intermediate: bool = True, prune_threshold: int = 1):
        """
        Fit function that reports validation loss to optuna trial each epoch and prunes
        
        Parameters:
            report_intermediate (bool): whether to report out
            prune_threshold (int): minimum epoch to wait before the first prune
        """
        cfg = self.cfg.training
        print(f'[Training Configuration]\n{asdict(self.cfg)}')

        for epoch in range(cfg.epochs):
            epoch_start = time.time()
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            # warmup and scheduling (same as parent)
            if epoch < self.cfg.scheduler.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.plateau_scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            self.writer.add_scalar('LR', current_lr, epoch)

            # Early stopping bookkeeping (same as parent)
            improved = val_loss < self.best_val_loss - cfg.min_delta
            if improved:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.best_state = deepcopy(self.model.state_dict())
                if self.trial is not None:
                    ckpt_dir = os.path.join('Saves', 'Models', 'checkpoints', f'trail_{self.trial.number}')
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(self.best_state, os.path.join(ckpt_dir, 'best.pth'))
            else:
                self.epochs_no_improve += 1

            # Logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            print(
                f'[Epoch {epoch+1}/{cfg.epochs}] '
                f'Train={train_loss:.4e} Val={val_loss:.4e} '
                f'LR={current_lr:.2e} Time={time.time()-epoch_start:.1f}s'
            )

            # Optuna reporting & pruning
            if self.trial is not None and report_intermediate:
                # report validation loss to Optuna
                self.trial.report(val_loss, epoch)
                # only aprune after threshold
                if epoch >= prune_threshold and self.trial.should_prune():
                    # save checkpoint
                    ckpt_dir = os.path.join('Saves', 'Models', 'checkpoints', f'trial_{self.trial.number}')
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(ckpt_dir, f'pruned_epoch_{epoch+1}.pth'))
                    raise optuna.exceptions.TrialPruned(f'Trial {self.trial.number} was prruned at epoch {epoch+1}')
                
            # Early Stopping
            if self.epochs_no_improve >= cfg.early_stopping_patience:
                print(f'\n[Early Stopping] No improvement for {cfg.early_stopping_patience} epochs.')
                break

        # load best model (if it exists)
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
        
        self.writer.close()

        print(f'\nBest Val Loss: {self.best_val_loss:.4e}')
        print(f'Total Training Time: {time.time() - self.start_time:.1f}s')
        print(f'TensorBoard logs saved at: {self.log_dir}')

        return self.model, self.history
    

def obj(trial: optuna.trial.Trial, n_epochs_warmup_prune: int = 3):
    """
    Optuna objective:
        - suggest hyperparameters
        - build config, model, dataloaders
        - run OptunaTrainer.fit() and report minimal validation loss returned
    """

    # Hyperparameters to tune
    lr = trial.suggest_float('optimizer.lr', 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float('optimizer.weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('training.batch_size', [16, 32, 64])
    d_model = trial.suggest_categorical('transformer.d_model', [64, 128, 256])
    num_enc = trial.suggest_categorical('transformer.num_encoder_layers', [2, 3, 4])
    num_dec = trial.suggest_categorical('transformer.num_decoder_layers', [2, 3])
    dropout = trial.suggest_float('transformer.dropout', 0.05, 0.3)

    # build Config instance
    cfg = Config()
    # optimizer
    cfg.optimizer.lr = lr
    cfg.optimizer.weight_decay = weight_decay
    # training
    cfg.training.batch_size = batch_size
    cfg.training.seed = 42  
    # transformer
    cfg.transformer.d_model = d_model
    cfg.transformer.num_encoder_layers = num_enc
    cfg.transformer.num_decoder_layers = num_dec
    cfg.transformer.dropout = dropout

    utils.set_seed(cfg.training.seed)

    # prepare data loaders and model
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg)

    # setup trial-specific logging dir
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    trial_logdir = os.path.join('runs', 'optuna', f'trial_{trial.number}_{timestamp}')
    os.makedirs(trial_logdir, exist_ok=True)

    # trainer
    trainer = OptunaTrainer(
        model=model,
        loss_fn=masked_mse_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        log_dir=trial_logdir,
        logger=None,
        trial=trial
    )

    try:
        # fit model, prune
        trained_model, _ = trainer.fit(report_intermediate=True, prune_threshold=n_epochs_warmup_prune)
    except optuna.exceptions.TrialPruned as e:
        # Reraise to let Optuna handle pruning
        raise

    # evaluate the final test loss
    results = utils.evaluate_model(
        trained_model,
        test_loader,
        loss_fns=masked_mse_loss
    )
    test_loss = results['masked_mse_loss']
    return test_loss

def run_optuna_study(
        n_trials: int = 40,
        n_jobs: int = 1,
        timeout: int = None,
        study_name: str = 'gnn_transformer_opt',
        storage = None
):
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=4, n_warmup_steps=3)
    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        direction='minimize',
        storage=storage,
        load_if_exists=True
    )

    # optimize
    study.optimize(lambda t: obj(t, n_epochs_warmup_prune=3), n_trials=n_trials, n_jobs=n_jobs, timeout=timeout)
    
    print('[INFO] Study Finished')
    print('Best Trial:')
    trial = study.best_trial
    print(f'\Test Loss: {trial.value}')
    print('\tParams:')
    for k, v in trial.params.items():
        print(f'\t\t{k}: {v}')
    
    # copy best checkpoint to central location if exists
    best_ckpt_src = os.path.join('Saves', 'Models', 'checkpoints', f'trial_{trial.number}', 'best.pth')
    if os.path.exists(best_ckpt_src):
        os.makedirs('Saves', 'Models', 'checkpoints', exist_ok=True)
        shutil.copy(best_ckpt_src, os.path.join('Saves', 'Models', 'checkpoints', 'best_model.pth'))
        print('Copied best model checkpoint to "Saves/Models/checkpoints/best_model.pth')
    else:
        print('No checkpoint found for best trial. (Could have been pruned or checkpointing not triggered)')
    
    return study

if __name__ == '__main__':
    # Trial params
    n_trials = 40
    n_jobs = 1
    timeout_seconds = None

    study = run_optuna_study(n_trials=n_trials, n_jobs=n_jobs, timeout=timeout_seconds)