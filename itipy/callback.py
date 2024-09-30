import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from itipy.trainer import Trainer

from itipy.iti import ITIModule

class BasicPlot(pl.Callback):
    """
        Basic plot callback for visualization of the data and the model predictions.

        Args:
            data (Dataset): Data to visualize.
            model (ITIModule): Model to use.
            plot_id (str): Plot id.
            plot_settings (list): List of plot settings.
            dpi (int): Dots per inch.
            batch_size (int): Batch size.
        """
    def __init__(self, data, model: Trainer, plot_id, plot_settings, dpi=100, batch_size=None, **kwargs):
        self.data = data
        self.model = model
        self.plot_settings = plot_settings
        self.dpi = dpi
        self.plot_id = plot_id
        self.batch_size = batch_size if batch_size is not None else len(data)

        super().__init__(**kwargs)

    def on_validation_epoch_end(self, *args, **kwargs):
        data = self.loadData()

        rows = len(data)
        columns = len(data[0])

        f, axarr = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
        axarr = np.reshape(axarr, (rows, columns))
        for i in range(rows):
            for j in range(columns):
                plot_settings = self.plot_settings[j].copy()
                ax = axarr[i, j]
                ax.axis("off")
                ax.set_title(plot_settings.pop("title", None))
                ax.imshow(data[i][j], **plot_settings)
        plt.tight_layout()
        wandb.log({f"{self.plot_id}": f})
        plt.close()
        del f, axarr, data

    def loadData(self):
        with torch.no_grad():
            loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=False)
            data, predictions = [], []
            for data_batch in loader:
                data_batch = data_batch.float().cuda()
                predictions_batch = self.predict(data_batch)
                data += [data_batch.detach().cpu().numpy()]
                predictions += [[pred.detach().cpu().numpy() for pred in predictions_batch]]
            data = np.concatenate(data)
            predictions = map(list, zip(*predictions)) # transpose
            predictions = [np.concatenate(p) for p in predictions]
            samples = [data, ] + [*predictions]
            # separate into rows and columns
            return [[d[j, i] for d in samples for i in range(d.shape[1])] for j in
                    range(len(self.data))]

    def predict(self, input_data):
        raise NotImplementedError()


class PlotABA(BasicPlot):
    """
    Plot callback for visualization of the data and the model predictions for the translation Instrument A -> Instrument B -> Instrument A.

    Args:
        data (Dataset): Data to visualize.
        model (ITIModule): Model to use.
        plot_settings_A (dict or list): Plot settings for Instrument A.
        plot_settings_B (dict or list): Plot settings for Instrument B.
        plot_id (str): Plot id.
        dpi (int): Dots per inch.
        batch_size (int): Batch size.
    """
    def __init__(self, data, model, plot_settings_A=None, plot_settings_B=None, plot_id="ABA",
                 **kwargs):
        plot_settings_A = plot_settings_A if plot_settings_A is not None else [{"cmap": "gray"}] * model.input_dim_a
        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) else [
                                                                                        plot_settings_A] * model.input_dim_a
        plot_settings_B = plot_settings_B if plot_settings_B is not None else [{"cmap": "gray"}] * model.input_dim_b
        plot_settings_B = plot_settings_B if isinstance(plot_settings_B, list) else [
                                                                                        plot_settings_B] * model.input_dim_b

        plot_settings = [*plot_settings_A, *plot_settings_B, *plot_settings_A]

        super().__init__(data, model, path, plot_id, plot_settings, **kwargs)

    def predict(self, x):
        x_ab, x_aba = self.model.forwardABA(x)
        return x_ab, x_aba


class PlotBAB(BasicPlot):
    """
    Plot callback for visualization of the data and the model predictions for the translation Instrument B -> Instrument A -> Instrument B.

    Args:
        data (Dataset): Data to visualize.
        model (ITIModule): Model to use.
        plot_settings_A (dict or list): Plot settings for Instrument A.
        plot_settings_B (dict or list): Plot settings for Instrument B.
        plot_id (str): Plot id.
        dpi (int): Dots per inch.
        batch_size (int): Batch size.
    """
    def __init__(self, data, model, plot_settings_A=None, plot_settings_B=None, plot_id="BAB",
                 **kwargs):
        plot_settings_A = plot_settings_A if plot_settings_A is not None else [{"cmap": "gray"}] * model.input_dim_a
        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) else [
                                                                                        plot_settings_A] * model.input_dim_a
        plot_settings_B = plot_settings_B if plot_settings_B is not None else [{"cmap": "gray"}] * model.input_dim_b
        plot_settings_B = plot_settings_B if isinstance(plot_settings_B, list) else [
                                                                                        plot_settings_B] * model.input_dim_b

        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) \
            else [plot_settings_A] * model.input_dim_a
        plot_settings_B = plot_settings_B if isinstance(plot_settings_B, list) \
            else [plot_settings_B] * model.input_dim_b

        plot_settings = [*plot_settings_B, *plot_settings_A, *plot_settings_B]

        super().__init__(data, model, path, plot_id, plot_settings, **kwargs)

    def predict(self, x):
        x_ba, x_bab = self.model.forwardBAB(x)
        return x_ba, x_bab


class PlotAB(BasicPlot):
    """
    Plot callback for visualization of the data and the model predictions for the translation Instrument A -> Instrument B.

    Args:
        data (Dataset): Data to visualize.
        model (ITIModule): Model to use.
        plot_settings_A (dict or list): Plot settings for Instrument A.
        plot_settings_B (dict or list): Plot settings for Instrument B.
        plot_id (str): Plot id.
        dpi (int): Dots per inch.
        batch_size (int): Batch size.
    """
    def __init__(self, data, model, plot_settings_A=None, plot_settings_B=None, plot_id="AB",
                 **kwargs):
        plot_settings_A = plot_settings_A if plot_settings_A is not None else [{"cmap": "gray"}] * model.input_dim_a
        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) else [
                                                                                        plot_settings_A] * model.input_dim_a
        plot_settings_B = plot_settings_B if plot_settings_B is not None else [{"cmap": "gray"}] * model.input_dim_b
        plot_settings_B = plot_settings_B if isinstance(plot_settings_B, list) else [
                                                                                        plot_settings_B] * model.input_dim_b

        plot_settings = [*plot_settings_A, *plot_settings_B]

        super().__init__(data, model, path, plot_id, plot_settings, **kwargs)

    def predict(self, input_data):
        x_ab = self.model.forwardAB(input_data)
        return (x_ab,)


class VariationPlotBA(BasicPlot):
    """
    Plot callback for visualization of the data and the model predictions for the variation Instrument B -> Instrument A.

    Args:
        data (Dataset): Data to visualize.
        model (ITIModule): Model to use.
        n_samples (int): Number of samples.
        plot_settings_A (dict or list): Plot settings for Instrument A.
        plot_settings_B (dict or list): Plot settings for Instrument B.
        plot_id (str): Plot id.
        dpi (int): Dots per inch.
        batch_size (int): Batch size.
    """
    def __init__(self, data, model, n_samples, plot_settings_A=None, plot_settings_B=None, plot_id="variation",
                 **kwargs):
        self.n_samples = n_samples

        plot_settings_A = plot_settings_A if plot_settings_A is not None else [{"cmap": "gray"}] * model.input_dim_a
        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) else [
                                                                                        plot_settings_A] * model.input_dim_a
        plot_settings_A = plot_settings_A * n_samples
        plot_settings_B = plot_settings_B if plot_settings_B is not None else [{"cmap": "gray"}] * model.input_dim_b
        plot_settings_B = plot_settings_B if isinstance(plot_settings_B, list) else [
                                                                                        plot_settings_B] * model.input_dim_b

        plot_settings = [*plot_settings_B, *plot_settings_A]

        super().__init__(data, model, plot_id, plot_settings, **kwargs)

    def predict(self, x):
        x_ba = torch.cat([self.model.forwardBA(x) for _ in range(self.n_samples)], 1)
        return (x_ba,)


class SaveCallback(pl.Callback):
    """
    Callback to save the model state and the generator weights.

    Args:
        checkpoint_dir (str): Directory to save the checkpoints.
    """
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        super().__init__()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", module: "ITIModule") -> None:

        state_path = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{trainer.global_step:06d}.pt')
        state = {'gen_ab': module.gen_ab,
                 'gen_ba': module.gen_ba,
                 'noise_est': module.estimator_noise,
                 'disc_a': module.dis_a,
                 'disc_b': module.dis_b,
                 'state_dict': module.state_dict(),
                 'global_step': trainer.global_step}
        torch.save(state, state_path)

        torch.save(state, checkpoint_path)
        torch.save(module.gen_ab, os.path.join(self.checkpoint_dir, 'generator_AB.pt'))
        torch.save(module.gen_ba, os.path.join(self.checkpoint_dir, 'generator_BA.pt'))