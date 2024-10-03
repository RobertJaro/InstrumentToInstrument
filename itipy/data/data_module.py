from lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler


class ITIDataModule(LightningDataModule):
    """
        DataModule for the ITI project based on the LightningDataModule. It provides the DataLoader for the training and validation data.


        Args:
            A_train_ds: Dataset for domain A training data
            B_train_ds: Dataset for domain B training data
            A_valid_ds: Dataset for domain A validation data
            B_valid_ds: Dataset for domain B validation data
            iterations_per_epoch: Number of iterations per epoch
            num_workers: Number of workers for DataLoader
            batch_size: Batch size for DataLoader
            **kwargs: Additional arguments

    """

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
        """
                DataLoader for the training data

                Returns:
                    dict: DataLoader for the training data. This includes the generators and discriminators for both domains.
        """
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
        """
                DataLoader for the validation data

                Returns:
                    list: DataLoader for the validation data. This includes the generators and discriminators for both domains.
        """
        A = DataLoader(self.A_valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        B = DataLoader(self.B_valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        return [A, B]
