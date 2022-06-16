import os
import torch

from iti.train.model import DiscriminatorMode
from iti.trainer import Trainer

path = "/gpfs/gpfs0/robert.jarolim/iti/hmi_hinode_v2"
trainer = Trainer(1, 1, upsampling=2, norm='in_rs_aff', lambda_diversity=0, use_batch_statistic=True)
trainer.cuda()
start_it = trainer.resume(path, 200000)
#stereo mag 360000
# hmiv12 500000
# kso 140000 --> BA; 200000 --> AB

torch.save(trainer.gen_ab, os.path.join(path, 'generator_AB.pt'))
torch.save(trainer.gen_ba, os.path.join(path, 'generator_BA.pt'))