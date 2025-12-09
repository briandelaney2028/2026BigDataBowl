import numpy as np
import os
import time
import shutil
from copy import deepcopy
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict
import utils
from utils import Config
from train import *
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings(
    "ignore",
    message="Deterministic behavior was enabled with either `torch.use_deterministic_algorithms",
)


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
                    raise optuna.exceptions.TrialPruned(f'Trial {self.trial.number} was pruned at epoch {epoch+1}')
                
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
    delta = trial.suggest_float('training.delta', 0.3, 1.0)
    time_decay = trial.suggest_float('training.time_decay', 0, 0.12)
    # lambda_vel = trial.suggest_float('training.lambda_vel', 1e-3, 1e2, log=True)
    # lambda_acc = trial.suggest_float('training.lambda_acc', 1e-3, 1e2, log=True)
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
    cfg.training.delta = delta
    cfg.training.time_decay = time_decay
    cfg.training.lambda_vel = 0 #lambda_vel
    cfg.training.lambda_acc = 0 #lambda_acc
    cfg.training.seed = 42  
    # transformer
    cfg.transformer.d_model = d_model
    cfg.transformer.num_encoder_layers = num_enc
    cfg.transformer.num_decoder_layers = num_dec
    cfg.transformer.dropout = dropout

    utils.set_seed(cfg.training.seed)

    # prepare data loaders and model
    train_loader, val_loader, test_loader, _ = get_dataloaders(cfg)
    model = build_model(cfg)

    # setup trial-specific logging dir
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    trial_logdir = os.path.join('runs', 'optuna', f'trial_{trial.number}_{timestamp}')
    os.makedirs(trial_logdir, exist_ok=True)

    # trainer
    trainer = OptunaTrainer(
        model=model,
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
        loss_fns=masked_FDE_loss
    )
    test_loss = results['masked_FDE_loss']
    return test_loss

def run_optuna_study(
        n_trials: int = 40,
        n_jobs: int = 1,
        timeout: int = None,
        study_name: str = 'no_GNN',
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
    trial = study.best_trial
    print(f'Best Trial: {trial.number}')
    print(f'\tTest Loss: {trial.value}')
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
    n_trials = 30
    n_jobs = 1
    timeout_seconds = None

    study = run_optuna_study(n_trials=n_trials, n_jobs=n_jobs, timeout=timeout_seconds)