import os
import wandb
import numpy as np
import hydra
import pickle
import torch
import lightning as pl
import matplotlib.pyplot as plt
import mdtraj as md
import pyemma
import nglview as nv
import plotly.graph_objects as go
from tqdm import tqdm
from matplotlib.colors import LogNorm
from itertools import combinations
import mdtraj as md
import pyemma
import nglview as nv
import plotly.graph_objects as go
from tqdm import tqdm
from matplotlib.colors import LogNorm
from itertools import combinations

from pathlib import Path
from torch.optim import Adam
from typing import Optional, Dict
from omegaconf import DictConfig, OmegaConf

# MLColvar imports
from mlcolvar.data import DictDataset, DictModule
from mlcolvar.cvs import AutoEncoderCV, DeepTDA, DeepTICA, VariationalAutoEncoderCV
from mlcolvar.core.transform import Statistics, Transform
from mlcolvar.core.loss.elbo import elbo_gaussians_loss
from mlcolvar.utils.trainer import MetricsCallback

# PyTorch Lightning imports
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# Ensure numpy compatibility
np.bool = np.bool_

# Color constants for visualization
blue = (70 / 255, 110 / 255, 250 / 255)
green = (100 / 255, 170 / 255, 120 / 255)


def sanitize_range(range_tensor: torch.Tensor) -> torch.Tensor:
    if (range_tensor < 1e-6).nonzero().sum() > 0:
        print(
            "[Warning] Normalization: the following features have a range of values < 1e-6:",
            (range_tensor < 1e-6).nonzero(),
        )
    range_tensor[range_tensor < 1e-6] = 1.0
    return range_tensor


class PostProcess(Transform):
    def __init__(
        self,
        stats=None,
        reference_frame_cv=None,
        feature_dim=1,
    ):
        super().__init__(in_features=feature_dim, out_features=feature_dim)
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("range", torch.ones(feature_dim))
        
        if stats is not None:
            min_val = stats["min"]
            max_val = stats["max"]
            self.mean = (max_val + min_val) / 2.0
            range_val = (max_val - min_val) / 2.0
            self.range = sanitize_range(range_val)
        
        if reference_frame_cv is not None:
            self.register_buffer(
                "flip_sign",
                torch.ones(1) * -1 if reference_frame_cv < 0 else torch.ones(1)
            )
        else:
            self.register_buffer("flip_sign", torch.ones(1))
        
    def forward(self, x):
        x = x.sub(self.mean).div(self.range)
        x = x * self.flip_sign
        return x


class TAE(AutoEncoderCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backward(self, loss):
        loss.backward(retain_graph=True)


class VDELoss(torch.nn.Module):
    def forward(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        z_t: torch.Tensor,
        z_t_tau: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> tuple:
        elbo_loss = elbo_gaussians_loss(target, output, mean, log_variance, weights)
        
        z_t_mean = z_t.mean(dim=0)
        z_t_tau_mean = z_t_tau.mean(dim=0)
        z_t_centered = z_t - z_t_mean.repeat(z_t.shape[0], 1)
        z_t_tau_centered = z_t_tau - z_t_tau_mean.repeat(z_t_tau.shape[0], 1)
        
        ac_num = z_t_centered.reshape(1, -1) @ z_t_tau_centered.reshape(-1, 1)
        ac_den = z_t_centered.norm(2) * z_t_tau_centered.norm(2)
        auto_correlation_loss = - ac_num / ac_den
        
        return elbo_loss, auto_correlation_loss


class VariationalDynamicsEncoder(VariationalAutoEncoderCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = VDELoss()
        self.optimizer = Adam(self.parameters(), lr=1e-4)
    
    def backward(self, loss):
        loss.backward(retain_graph=True)
    
    def training_step(self, train_batch, batch_idx):
        x = train_batch["data"]
        input_tensor = x
        loss_kwargs = {}
        if "weights" in train_batch:
            loss_kwargs["weights"] = train_batch["weights"]

        # Encode/decode.
        mean, log_variance, x_hat = self.encode_decode(x)

        # Reference output
        if "target" in train_batch:
            x_ref = train_batch["target"]
        else:
            x_ref = x
        
        # Values for autocorrelation loss
        if self.norm_in is not None:
            input_normalized = self.norm_in(input_tensor)
            x_ref_normalized = self.norm_in(x_ref)
        z_t = self.encoder(input_normalized)
        z_t_tau = self.encoder(x_ref_normalized)
        
        # Loss function.
        elbo_loss, auto_correlation_loss = self.loss_fn(
            x_ref, x_hat, mean, log_variance,
            z_t, z_t_tau,
            **loss_kwargs
        )

        # Log.
        name = "train" if self.training else "valid"
        self.log(f"{name}_elbo_loss", elbo_loss, on_epoch=True)
        self.log(f"{name}_auto_correlation_loss", auto_correlation_loss, on_epoch=True)
        self.log(f"{name}_loss", elbo_loss + auto_correlation_loss, on_epoch=True)

        return elbo_loss + auto_correlation_loss


def load_dataset(cfg: DictConfig) -> tuple:
    molecule = cfg.molecule
    dataset_size = cfg.dataset_size
    timelag = cfg.get('timelag', 10)
    method = cfg.method
    
    if method == 'tda':
        current_cad_path = f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-{dataset_size}/current-cad.pt"
        current_label_path = f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-{dataset_size}/current-label.pt"
        current_cad = torch.load(current_cad_path)
        current_label = torch.load(current_label_path)
        
        dataset = DictDataset({
            "data": current_cad,
            "labels": current_label.to(torch.float32)
        })
        
    elif method == 'tica':
        current_cad_path = f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-{dataset_size}/current-cad.pt"
        timelagged_cad_path = f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-{dataset_size}/lag{timelag}-cad.pt"
        current_cad = torch.load(current_cad_path)
        timelagged_cad = torch.load(timelagged_cad_path)

        dataset = DictDataset({
            "data": current_cad,
            "data_lag": timelagged_cad,
            "weights": torch.ones(current_cad.shape[0], dtype=torch.float32, device=current_cad.device),
            "weights_lag": torch.ones(timelagged_cad.shape[0], dtype=torch.float32, device=timelagged_cad.device)
        })
        
    elif method in ['tae', 'vde']:
        current_cad_path = f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-{dataset_size}/current-cad.pt"
        timelagged_cad_path = f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-{dataset_size}/lag{timelag}-cad.pt"
        current_cad = torch.load(current_cad_path)
        timelagged_cad = torch.load(timelagged_cad_path)
        
        dataset = DictDataset({
            "data": current_cad,
            "target": timelagged_cad
        })
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    datamodule = DictModule(dataset, lengths=[0.8, 0.2])
    return datamodule, current_cad.shape[1]


def create_model(cfg: DictConfig, input_dim: int):
    method = cfg.method
    mlcv_dim = cfg.get('mlcv_dim', 2)
    
    if method == 'tda':
        model = DeepTDA(
            n_states=cfg.n_states,
            n_cvs=cfg.get('n_cvs', 1),
            target_centers=cfg.target_centers,
            target_sigmas=cfg.target_sigmas,
            layers=cfg.layers
        )
        
    elif method == 'tica':
        model = DeepTICA(
            n_cvs=cfg.get('n_cvs', 1),
            layers=cfg.layers
        )
        
    elif method == 'tae':
        options = {
            "encoder": {
                "activation": cfg.get('activation', 'tanh'),
                "dropout": cfg.get('dropout', [0.5, 0.5, 0.5]),
            }
        }
        model = TAE(
            encoder_layers=[input_dim] + cfg.hidden_layers + [mlcv_dim],
            options=options
        )
        
    elif method == 'vde':
        options = {
            "encoder": {
                "activation": cfg.get('activation', 'relu'),
                "dropout": cfg.get('dropout', [0.3, 0.3, 0.3]),
                "last_layer_activation": True
            }
        }
        model = VariationalDynamicsEncoder(
            n_cvs=cfg.get('n_cvs', 1),
            encoder_layers=[input_dim] + cfg.hidden_layers + [cfg.get('n_cvs', 1)],
            options=options
        )
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return model


def train_model(cfg: DictConfig, model, datamodule):
    """Train the model"""
    # Define callbacks
    metrics = MetricsCallback()
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        min_delta=cfg.get('early_stopping_min_delta', 0.1),
        patience=cfg.get('early_stopping_patience', 50)
    )

    # Create WandB logger
    wandb_logger = WandbLogger(
        project="bioemu-baselines",
        name=f"{cfg.method}_{cfg.molecule}_{cfg.dataset_size}",
        log_model=False,
        tags=["baseline"]
    )

    # Define trainer
    trainer = pl.Trainer(
        callbacks=[metrics, early_stopping],
        max_epochs=cfg.get('max_epochs', 1000),
        logger=wandb_logger,
        enable_checkpointing=True,
        default_root_dir="./lightning_logs"
    )

    # Fit
    trainer.fit(model, datamodule)
    model.eval()
    
    return trainer


def evaluate_model(cfg: DictConfig, model, input_dim: int):
    """Evaluate the model and create visualizations"""
    molecule = cfg.molecule
    method = cfg.method
    timelag = cfg.get('timelag', 10)
    
    # Load projection data
    projection_data_path = f"/home/shpark/prj-mlcv/lib/DESRES/DESRES-Trajectory_{molecule}-0-protein/{molecule}-0-cad.pt"
    projection_data = torch.load(projection_data_path)

    model.eval()
    cv = model(projection_data)
    cv = cv.detach().numpy()
    
    print(f"CV shape: {cv.shape}")
    
    # Post-process for TAE and VDE
    if method in ['tda', 'tae', 'vde']:
        stats = Statistics(torch.from_numpy(cv).cpu()).to_dict()
        model.postprocessing = PostProcess(stats, feature_dim=cv.shape[1]).to(model.device)
        postprocessed_cv = model(projection_data)
        postprocessed_cv_numpy = postprocessed_cv.detach().cpu().numpy()
        
        print(f"Original CV range: [{cv.min():.6f}, {cv.max():.6f}]")
        print(f"Postprocessed CV range: [{postprocessed_cv.min():.6f}, {postprocessed_cv.max():.6f}]")
    else:
        postprocessed_cv_numpy = cv
    
    # Load and compare with TICA
    switch = cfg.get('switch', True)
    projection_data_np = projection_data.numpy()
    if switch:
        tica_file = f'../data/{molecule}/{molecule}_tica_model_switch_lag{timelag}.pkl'
        projection_data_np = (1 - np.power(projection_data_np / 0.8, 6)) / (1 - np.power(projection_data_np / 0.8, 12))
    else:
        tica_file = f'../data/{molecule}/{molecule}_tica_model_lag{timelag}.pkl'
        
    with open(tica_file, 'rb') as f:
        tica = pickle.load(f)
    tica_coord = tica.transform(projection_data_np)
    
    # Create visualization
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    
    if cv.shape[1] >= 2:
        # 2D CV visualization
        hb = ax.hexbin(
            tica_coord[:, 0], tica_coord[:, 1], C=-postprocessed_cv_numpy[:, 0],
            gridsize=200,
            reduce_C_function=np.mean,
            cmap='viridis',
        )
    else:
        # 1D CV visualization
        hb = ax.hexbin(
            tica_coord[:, 0], tica_coord[:, 1], C=-postprocessed_cv_numpy[:, 0],
            gridsize=200,
            reduce_C_function=np.mean,
            cmap='viridis',
        )
    
    plt.colorbar(hb)
    plt.xlabel("TIC 1")
    plt.ylabel("TIC 2")
    plt.title(f"{method.upper()} - {molecule}")
    
    # Save plot
    plot_path = f"./img/{method}_{molecule}_tica_lag{timelag}.png"
    os.makedirs("./img", exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    # Log to WandB
    wandb.log({
        "tica_comparison": wandb.Image(plot_path),
        "cv_min": float(cv.min()),
        "cv_max": float(cv.max()),
        "cv_mean": float(cv.mean()),
        "cv_std": float(cv.std()),
    })
    
    return cv, postprocessed_cv_numpy if method in ['tae', 'vde'] else cv


def save_model(cfg: DictConfig, model, datamodule, input_dim: int):
    # Save state dict
    molecule = cfg.molecule
    method = cfg.method
    save_dir = Path(f"/home/shpark/prj-mlcv/lib/bioemu/model/_baseline_")
    save_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = save_dir / f"{method}-{molecule}.pt"
    torch.save(model.state_dict(), state_dict_path)
    
    # Save JIT traced model
    model.trainer = Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False)
    random_input = torch.rand(1, input_dim).to(model.device)
    traced_script_module = torch.jit.trace(model, random_input)
    jit_path = save_dir / f"{method}-{molecule}-jit.pt"
    traced_script_module.save(str(jit_path))
    
    print(f"Models saved to {save_dir}")


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="baseline_tda",
)
def main(cfg: DictConfig) -> None:
    print("=" * 50)
    print("DESRES Baseline Training")
    print("=" * 50)
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize WandB
    wandb.init(
        project="bioemu-baselines",
        name=f"{cfg.method}_{cfg.molecule}_{cfg.dataset_size}",
        config=OmegaConf.to_container(cfg, resolve=True),
        dir="./wandb_logs"
    )

    try:
        # Load dataset
        print("Loading dataset...")
        datamodule, input_dim = load_dataset(cfg)
        print(f"Input dimension: {input_dim}")
        
        # Create model
        print(f"Creating {cfg.method.upper()} model...")
        model = create_model(cfg, input_dim)
        print(f"Model: {model}")
        
        # Train model
        print("Training model...")
        train_model(cfg, model, datamodule)
        
        # Evaluate model
        print("Evaluating model...")
        cv, postprocessed_cv = evaluate_model(cfg, model, input_dim)
        
        # Save model
        print("Saving model...")
        save_model(cfg, model, datamodule, input_dim)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        wandb.log({"error": str(e)})
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
