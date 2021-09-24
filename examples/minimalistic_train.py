from iti.data.dataset import SDODataset, SOHODataset
from iti.train.model import DiscriminatorMode
from iti.train.trainer import Trainer

base_dir = "/gss/r.jarolim/iti/soho_to_sdo"
sdo_data_path = "/gss/r.jarolim/data/ch_detection"
soho_data_path = "/gss/r.jarolim/data/soho_iti2021_prep"
# Init model
trainer = Trainer(input_dim_a=5, input_dim_b=5, upsampling=1, discriminator_mode=DiscriminatorMode.CHANNELS, lambda_diversity=0, norm='in_rs_aff')

# Init training datasets
sdo_train = SDODataset(sdo_data_path,
                       resolution=2048, patch_shape=(256, 256),
                       months=list(range(11)))
soho_train = SOHODataset(soho_data_path,
                         resolution=1024, patch_shape=(128, 128),
                         months=list(range(11)))
# Init validation/plotting datasets
sdo_valid = SDODataset(sdo_data_path,
                       resolution=2048, patch_shape=(256, 256),
                       months=[11, 12], limit=100)
soho_valid = SOHODataset(soho_data_path,
                         resolution=1024, patch_shape=(128, 128),
                         months=[11, 12], limit=100)

# Start training
trainer.startBasicTraining(base_dir,
                           ds_A=soho_train, ds_B=sdo_train,
                           ds_valid_A=soho_valid, ds_valid_B=sdo_valid)
