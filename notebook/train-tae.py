from mlcolvar.data import DictDataset, DictModule
from mlcolvar.cvs import AutoEncoderCV
from mlcolvar.core.transform import Statistics
from mlcolvar.core.transform import Transform
from mlcolvar.utils.trainer import MetricsCallback

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import numpy as np
import mdtraj as md
import nglview as nv
import torch
import lightning
import pickle

np.bool = np.bool_

from tqdm import tqdm
from pytorch_lightning import Trainer

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


timelag = 10
molecule = "2JOF"
MLCV_DIM = 1

class TAE(AutoEncoderCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def backward(self, loss):
        loss.backward(retain_graph=True)
    
def sanitize_range(range: torch.Tensor):
    """Sanitize

    Parameters
    ----------
    range : torch.Tensor
        range to be used for standardization

    """

    if (range < 1e-6).nonzero().sum() > 0:
        print(
            "[Warning] Normalization: the following features have a range of values < 1e-6:",
            (range < 1e-6).nonzero(),
        )
    range[range < 1e-6] = 1.0

    return range

class PostProcess(Transform):
    def __init__(
        self,
        stats = None,
        reference_frame_cv = None,
        feature_dim = 1,
    ):
        super().__init__(in_features=feature_dim, out_features=feature_dim)
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("range", torch.ones(feature_dim))
        
        if stats is not None:
            min = stats["min"]
            max = stats["max"]
            self.mean = (max + min) / 2.0
            range = (max - min) / 2.0
            self.range = sanitize_range(range)
        
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



# Load data
pdb_path = f"../data/{molecule}/{molecule}_from_mae.pdb"
current_cad_path = f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-5k/current-cad.pt"
timelagged_cad_path = f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-5k/lag{timelag}-cad.pt"
current_cad = torch.load(current_cad_path)
timelagged_cad = torch.load(timelagged_cad_path)
new_dataset = DictDataset({
	"data": current_cad,
	"target": timelagged_cad
})
datamodule = DictModule(new_dataset,lengths=[0.8,0.2])


# Load model
options = {
	"encoder": {
		"activation": "tanh",
		"dropout": [0.5, 0.5, 0.5],
	}
}
model = TAE(
    encoder_layers=[datamodule.dataset["data"].shape[1], 100, 100, MLCV_DIM],
    options=options
)
print(model)


# Training
metrics = MetricsCallback()
early_stopping = EarlyStopping(monitor="valid_loss", min_delta=0.1, patience=50)
trainer = lightning.Trainer(
    callbacks=[metrics, early_stopping],
	max_epochs=None,
 	logger=None,
  	enable_checkpointing=False
)
trainer.fit( model, datamodule )
model.eval()


# Evalauation
projection_data_path = f"/home/shpark/prj-mlcv/lib/DESRES/dataset/{molecule}-all/cad.pt"
projection_data = torch.load(projection_data_path)
cv = model(projection_data)
cv = cv.detach().numpy()
stats = Statistics(torch.from_numpy(cv).cpu()).to_dict()
model.postprocessing = PostProcess(stats).to(model.device)
postprocessed_cv = model(projection_data)
print(f"cv.shape: {cv.shape}")
print(f"cv.max(): {cv.max()}")
print(f"cv.min(): {cv.min()}")
print(f"postprocessed_cv.max(): {postprocessed_cv.max()}")
print(f"postprocessed_cv.min(): {postprocessed_cv.min()}")

# Save model
model_save_dir = "/home/shpark/prj-mlcv/lib/bioemu/model"
model_name = "tae"
torch.save(model.state_dict(), f"{model_save_dir}/_baseline_/{model_name}-{molecule}.pt")
model.trainer = Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False)
input_dim = datamodule.dataset["data"].shape[1]
random_input = torch.rand(1, input_dim).to(model.device)
traced_script_module = torch.jit.trace(model, random_input)
traced_script_module.save(f"{model_save_dir}/_baseline_/{model_name}-{molecule}-jit.pt")
print(f"Model saved to {model_save_dir}/_baseline_/{model_name}-{molecule}-jit.pt")


# Evaluation -TICA
with open(f'../data/{molecule}/{molecule}_tica_model_lag{timelag}.pkl', 'rb') as f:
    tica = pickle.load(f)
projection_data_np = projection_data.numpy()
tica_coord = tica.transform(projection_data_np)
postprocessed_cv_numpy = postprocessed_cv.detach().cpu().numpy()
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
hb = ax.hexbin(
	tica_coord[:, 0], tica_coord[:, 1], C=-postprocessed_cv_numpy[:, 0],  # data
	gridsize=200,                     # controls resolution
	reduce_C_function=np.mean,       # compute average per hexagon
	cmap='viridis',                  # colormap
)
plt.colorbar(hb)
plt.xlabel("TIC 1")
plt.ylabel("TIC 2")
plt.show()
plt.savefig(f"./tae_{molecule}_tica_lag{timelag}.png")