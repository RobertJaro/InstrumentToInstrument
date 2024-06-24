from itipy.data.dataset import SDODataset, SOHODataset
from itipy.train.model import DiscriminatorMode
from itipy.trainer import Trainer

base_dir = "<<path for training results>>"
sdo_data_path = "<<data set B>>"
soho_data_path = "<<data set A>>"
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
