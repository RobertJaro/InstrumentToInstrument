import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch

from iti.trainer import Trainer

path = "/gss/r.jarolim/iti/hmi_hinode_v13"
trainer = Trainer(1, 1, upsampling=2, norm='in_rs_aff', lambda_diversity=0)
trainer.cuda()
start_it = trainer.resume(path, 400000)
#stereo mag 360000
# hmiv12 500000
# kso 140000 --> BA; 200000 --> AB

torch.save(trainer.gen_ab, os.path.join(path, 'generator_AB.pt'))
torch.save(trainer.gen_ba, os.path.join(path, 'generator_BA.pt'))