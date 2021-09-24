import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from skimage.io import imsave

import torch
from torch.utils.data import DataLoader

from iti.data.dataset import KSOFlatDataset
from iti.train.trainer import Trainer

resolution = 1024
base_path = "/gss/r.jarolim/iti/kso_quality_1024_v7"
epoch = 140000
evaluation_path = os.path.join(base_path, 'evaluation')
os.makedirs(evaluation_path, exist_ok=True)

q1_dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_synoptic", resolution, months=[11, 12])
q1_loader = DataLoader(q1_dataset, batch_size=1, shuffle=True)
q1_iter = q1_loader.__iter__()

q2_dataset = KSOFlatDataset("/gss/r.jarolim/data/kso_general/quality2", resolution, months=[11, 12])
q2_loader = DataLoader(q2_dataset, batch_size=1, shuffle=True)
q2_iter = q2_loader.__iter__()

trainer = Trainer(1, 1)
trainer.resume(checkpoint_dir=base_path, epoch=epoch)
trainer.cuda()

with torch.no_grad():
    for i in range(20):
        A_img = next(q2_iter).float().cuda()
        AB_img, ABA_img = trainer.forwardABA(A_img)
        #
        B_img = next(q1_iter).float().cuda()
        BA_img, BAB_img = trainer.forwardBAB(B_img)
        #
        AA_img = trainer.forwardBA(A_img)
        BB_img = trainer.forwardAB(B_img)
        #
        imsave(os.path.join(evaluation_path, '%03d_sample_B.jpg' % i), B_img.detach().cpu().numpy()[0, 0])
        imsave(os.path.join(evaluation_path, '%03d_sample_BA.jpg' % i), BA_img.detach().cpu().numpy()[0, 0])
        imsave(os.path.join(evaluation_path, '%03d_sample_BAB.jpg' % i), BAB_img.detach().cpu().numpy()[0, 0])
        #
        imsave(os.path.join(evaluation_path, '%03d_sample_A.jpg' % i), A_img.detach().cpu().numpy()[0, 0])
        imsave(os.path.join(evaluation_path, '%03d_sample_AB.jpg' % i), AB_img.detach().cpu().numpy()[0, 0])
        imsave(os.path.join(evaluation_path, '%03d_sample_ABA.jpg' % i), ABA_img.detach().cpu().numpy()[0, 0])
        #
        imsave(os.path.join(evaluation_path, '%03d_sample_AA.jpg' % i), AA_img.detach().cpu().numpy()[0, 0])
        imsave(os.path.join(evaluation_path, '%03d_sample_BB.jpg' % i), BB_img.detach().cpu().numpy()[0, 0])
