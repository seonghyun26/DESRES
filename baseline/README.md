# DESRES Baseline Models

This directory contains a unified implementation of three baseline methods for molecular dynamics analysis:

1. **TDA (Time-lagged Discriminant Analysis)** - Supervised learning using labeled states
2. **TAE (Time-lagged Auto-encoder)** - Unsupervised learning using time-lagged reconstruction  
3. **VDE (Variational Dynamics Encoder)** - Variational approach with dynamics constraints

## Features

- **Hydra Configuration Management**: Easy parameter configuration through YAML files
- **Weights & Biases Integration**: Automatic experiment tracking and visualization
- **Unified Interface**: Single script handles all three methods
- **Model Saving**: Saves both PyTorch state dicts and JIT traced models
- **TICA Comparison**: Automatically compares results with traditional TICA

## Usage

### Basic Usage

```bash
# Run TDA on CLN025
python main.py --config-name=baseline_tda

# Run TAE on 2JOF
python main.py --config-name=baseline_tae

# Run VDE on CLN025
python main.py --config-name=baseline_vde
```

### Available Configurations

- `baseline_tda.yaml` - TDA on CLN025 (50k dataset)
- `baseline_tae.yaml` - TAE on 2JOF (5k dataset)
- `baseline_tae_cln025.yaml` - TAE on CLN025 (5k dataset)  
- `baseline_vde.yaml` - VDE on CLN025 (5k dataset)
- `baseline_vde_2jof.yaml` - VDE on 2JOF (5k dataset)

### Configuration Override

You can override any configuration parameter from the command line:

```bash
# Change molecule and dataset size
python main.py --config-name=baseline_tae molecule=CLN025 dataset_size=50k

# Modify training parameters
python main.py --config-name=baseline_vde max_epochs=500 early_stopping_patience=100

# Change model architecture
python main.py --config-name=baseline_tae hidden_layers=[200,200,100] mlcv_dim=3
```

## Method Details

### TDA (Time-lagged Discriminant Analysis)
- Uses labeled data with discrete states
- Learns discriminative collective variables
- Good for systems with known metastable states

### TAE (Time-lagged Auto-encoder)
- Uses time-lagged pairs for reconstruction
- Learns latent representations through auto-encoding
- Captures slow dynamics without explicit labeling

### VDE (Variational Dynamics Encoder)
- Combines variational auto-encoder with dynamics constraints
- Uses ELBO loss + autocorrelation loss
- Provides uncertainty quantification through variational framework

## Output

The script produces:

1. **Trained Models**: Saved in `{model_save_dir}/_baseline_/`
   - `{method}-{molecule}.pt` - PyTorch state dict
   - `{method}-{molecule}-jit.pt` - JIT traced model

2. **Visualizations**: Saved in `./plots/`
   - TICA comparison plots showing learned CVs vs traditional TICA

3. **Weights & Biases Logs**: Experiment tracking in project "bioemu-baselines"
   - Training metrics, model parameters, visualizations

## Requirements

- PyTorch Lightning
- MLColvar
- Weights & Biases
- Hydra
- MDTraj
- PyEMMA (for TICA comparison)

## Dataset Structure

The script expects datasets in the following structure:
```
dataset/
├── {molecule}-{size}/
│   ├── current-cad.pt          # Current frame features
│   ├── current-label.pt        # Labels (for TDA only)
│   └── lag{timelag}-cad.pt     # Time-lagged features (for TAE/VDE)
└── {molecule}-all/
    └── cad.pt                  # Full dataset for evaluation
```
