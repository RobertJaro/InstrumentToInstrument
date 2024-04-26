from lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler


class ITIDataModule(LightningDataModule):

    def __init__(self, A_train_ds, B_train_ds, A_valid_ds, B_valid_ds, iterations_per_epoch=10000, num_workers=4, batch_size=1, **kwargs):
        super().__init__()
        self.A_train_ds = A_train_ds
        self.B_train_ds = B_train_ds
        self.A_valid_ds = A_valid_ds
        self.B_valid_ds = B_valid_ds
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.iterations_per_epoch = iterations_per_epoch

    def train_dataloader(self):
        gen_A = DataLoader(self.A_train_ds, batch_size=self.batch_size, num_workers=self.num_workers,
                           sampler=RandomSampler(self.A_train_ds, replacement=True, num_samples=self.iterations_per_epoch))
        dis_A = DataLoader(self.A_train_ds, batch_size=self.batch_size, num_workers=self.num_workers,
                            sampler=RandomSampler(self.A_train_ds, replacement=True, num_samples=self.iterations_per_epoch))
        gen_B = DataLoader(self.B_train_ds, batch_size=self.batch_size, num_workers=self.num_workers,
                            sampler=RandomSampler(self.B_train_ds, replacement=True, num_samples=self.iterations_per_epoch))
        dis_B = DataLoader(self.B_train_ds, batch_size=self.batch_size, num_workers=self.num_workers,
                            sampler=RandomSampler(self.B_train_ds, replacement=True, num_samples=self.iterations_per_epoch))
        return {"gen_A": gen_A, "dis_A": dis_A, "gen_B": gen_B, "dis_B": dis_B}

    def val_dataloader(self):
        A = DataLoader(self.A_valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        B = DataLoader(self.B_valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        return [A, B]
