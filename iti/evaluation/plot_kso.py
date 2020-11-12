import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from matplotlib import pyplot as plt
from skimage.io import imsave
from torch.utils.data import DataLoader

from iti.data.dataset import KSODataset
from iti.train.trainer import Trainer

resolution = 256
base_path = "/gss/r.jarolim/iti/kso_quality_256_v11"
os.makedirs(os.path.join(base_path, 'evaluation'), exist_ok=True)

q2_dataset = KSODataset("/gss/r.jarolim/data/kso_general/quality2", resolution)
q2_loader = DataLoader(q2_dataset, batch_size=4, shuffle=True)
q2_iter = q2_loader.__iter__()

q1_dataset = KSODataset("/gss/r.jarolim/data/kso_general/quality1", resolution)
q1_loader = DataLoader(q1_dataset, batch_size=4, shuffle=True)
q1_iter = q1_loader.__iter__()

trainer = Trainer(1, 1, norm='in_aff')
trainer.cuda()
iteration = trainer.resume(base_path, epoch=40000)

with torch.no_grad():
    A_img = next(q2_iter).float().cuda()
    AB_img, ABA_img = trainer.forwardABA(A_img)
    AB_img = AB_img.detach().cpu().numpy()
    ABA_img = ABA_img.detach().cpu().numpy()
    #
    B_img = next(q1_iter).float().cuda()
    BA_img, BAB_img = trainer.forwardBAB(B_img)
    BA_img = BA_img.detach().cpu().numpy()
    BAB_img = BAB_img.detach().cpu().numpy()
    #
    AA_img = trainer.forwardBA(A_img)
    AA_img = AA_img.detach().cpu().numpy()
    #
    BB_img = trainer.forwardAB(B_img)
    BB_img = BB_img.detach().cpu().numpy()
    #
    A_img = A_img.detach().cpu().numpy()
    B_img = B_img.detach().cpu().numpy()

for i in range(4):
    imsave(os.path.join(base_path, 'evaluation/%d_%d_A.jpg' % (iteration, i)),
           A_img[i, 0])
    imsave(os.path.join(base_path, 'evaluation/%d_%d_AB.jpg' % (iteration, i)),
           AB_img[i, 0])
    imsave(os.path.join(base_path, 'evaluation/%d_%d_ABA.jpg' % (iteration, i)),
           ABA_img[i, 0])

    imsave(os.path.join(base_path, 'evaluation/%d_%d_B.jpg' % (iteration, i)),
           B_img[i, 0])
    imsave(os.path.join(base_path, 'evaluation/%d_%d_BA.jpg' % (iteration, i)),
           BA_img[i, 0])
    imsave(os.path.join(base_path, 'evaluation/%d_%d_BAB.jpg' % (iteration, i)),
           BAB_img[i, 0])

    imsave(os.path.join(base_path, 'evaluation/%d_%d_AA.jpg' % (iteration, i)),
           AA_img[i, 0])
    imsave(os.path.join(base_path, 'evaluation/%d_%d_BB.jpg' % (iteration, i)),
           BB_img[i, 0])