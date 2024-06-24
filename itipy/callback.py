import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from itipy.trainer import Trainer


class Callback(ABC):

    def __init__(self, log_iteration=1000):
        self.log_iteration = log_iteration

    def __call__(self, iteration):
        if (iteration + 1) % self.log_iteration == 0:
            self.call(iteration)

    @abstractmethod
    def call(self, iteration, **kwargs):
        raise NotImplementedError()


class BasicPlot(Callback):

    def __init__(self, data, model: Trainer, path, plot_id, plot_settings, dpi=100, batch_size=None, **kwargs):
        self.data = data
        self.path = path
        self.model = model
        self.plot_settings = plot_settings
        self.dpi = dpi
        self.plot_id = plot_id
        self.batch_size = batch_size if batch_size is not None else len(data)

        super().__init__(**kwargs)

    def call(self, iteration, **kwargs):
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
        path = os.path.join(self.path, "%s_iteration%06d.jpg" % (self.plot_id, iteration + 1))
        plt.savefig(path, dpi=self.dpi)
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
    def __init__(self, data, model, path, plot_settings_A=None, plot_settings_B=None, plot_id="ABA",
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
    def __init__(self, data, model, path, plot_settings_A=None, plot_settings_B=None, plot_id="BAB",
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
    def __init__(self, data, model, path, plot_settings_A=None, plot_settings_B=None, plot_id="AB",
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

    def __init__(self, data, model, path, n_samples, plot_settings_A=None, plot_settings_B=None, plot_id="variation",
                 **kwargs):
        self.n_samples = n_samples

        plot_settings_A = plot_settings_A if plot_settings_A is not None else [{"cmap": "gray"}] * model.input_dim_a
        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) else [plot_settings_A] * model.input_dim_a
        plot_settings_A = plot_settings_A * n_samples
        plot_settings_B = plot_settings_B if plot_settings_B is not None else [{"cmap": "gray"}] * model.input_dim_b
        plot_settings_B = plot_settings_B if isinstance(plot_settings_B, list) else [
                                                                                        plot_settings_B] * model.input_dim_b

        plot_settings = [*plot_settings_B, *plot_settings_A]

        super().__init__(data, model, path, plot_id, plot_settings, **kwargs)

    def predict(self, x):
        x_ba = torch.cat([self.model.forwardBA(x) for _ in range(self.n_samples)], 1)
        return (x_ba,)


class HistoryCallback(Callback):
    def __init__(self, trainer: Trainer, path, log_iteration=1):
        self.trainer = trainer
        self.path = path

        self.loss = self.trainer.train_loss
        super().__init__(log_iteration)

    def call(self, iteration, **kwargs):
        self.loss['iteration'] += [iteration]
        self.loss['loss_gen_a_identity'] += [self.trainer.loss_gen_a_identity.cpu().detach().numpy()]
        self.loss['loss_gen_b_identity'] += [self.trainer.loss_gen_b_identity.cpu().detach().numpy()]
        self.loss['loss_gen_a_translate'] += [self.trainer.loss_gen_a_translate.cpu().detach().numpy()]
        self.loss['loss_gen_b_translate'] += [self.trainer.loss_gen_b_translate.cpu().detach().numpy()]
        self.loss['loss_gen_adv_a'] += [self.trainer.loss_gen_adv_a.cpu().detach().numpy()]
        self.loss['loss_gen_adv_b'] += [self.trainer.loss_gen_adv_b.cpu().detach().numpy()]
        self.loss['loss_gen_a_content'] += [self.trainer.loss_gen_a_content.cpu().detach().numpy()]
        self.loss['loss_gen_b_content'] += [self.trainer.loss_gen_b_content.cpu().detach().numpy()]
        self.loss['loss_gen_a_identity_content'] += [self.trainer.loss_gen_a_identity_content.cpu().detach().numpy()]
        self.loss['loss_gen_b_identity_content'] += [self.trainer.loss_gen_b_identity_content.cpu().detach().numpy()]
        self.loss['loss_gen_aba_noise'] += [self.trainer.loss_gen_aba_noise.cpu().detach().numpy()]
        self.loss['loss_gen_a_identity_noise'] += [self.trainer.loss_gen_a_identity_noise.cpu().detach().numpy()]
        self.loss['loss_dis_a'] += [self.trainer.loss_dis_a.cpu().detach().numpy()]
        self.loss['loss_dis_b'] += [self.trainer.loss_dis_b.cpu().detach().numpy()]
        self.loss['loss_gen_diversity'] += [self.trainer.loss_gen_diversity.cpu().detach().numpy()]
        if (iteration + 1) % 100 == 0:
            self.plotAdversarial()
            self.plotContent()
            self.plotDistortion()
            self.plotNoise()

    def plotAdversarial(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_adv_a']), label='Generator A')
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_adv_b']), label='Generator B')
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_dis_a']), label='Discriminator A')
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_dis_b']), label='Discriminator B')
        plt.ylim((0, 0.9))
        plt.legend()
        plt.savefig(os.path.join(self.path, "progress_adversarial.jpg"), dpi=100)
        plt.close()

    def plotContent(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_a_content']), label='Content A')
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_b_content']), label='Content B')
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_a_identity_content']), label='Content A Identity')
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_b_identity_content']), label='Content B Identity')
        plt.legend()
        plt.savefig(os.path.join(self.path, "progress_content.jpg"), dpi=100)
        plt.close()

    def plotNoise(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_a_identity_noise']), label='Noise A Identity')
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_aba_noise']), label='Noise ABA')
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_diversity']), label='Diversity')
        plt.legend()
        plt.savefig(os.path.join(self.path, "progress_noise.jpg"), dpi=100)
        plt.close()

    def plotDistortion(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_a_translate']), label='MAE A')
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_b_translate']), label='MAE B')
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_a_identity_content']), label='MAE A Identity')
        plt.plot(self.loss['iteration'][25:-24], running_mean(self.loss['loss_gen_b_identity_content']), label='MAE B Identity')
        plt.legend()
        plt.savefig(os.path.join(self.path, "progress_distortion.jpg"), dpi=100)
        plt.close()

class ValidationHistoryCallback(Callback):
    def __init__(self, trainer: Trainer, data_set_A, data_set_B, path, log_iteration=1000, num_workers=4):
        self.trainer = trainer
        self.path = path
        self.data_set_A = data_set_A
        self.data_set_B = data_set_B
        self.num_workers = num_workers

        self.loss = self.trainer.valid_loss
        super().__init__(log_iteration)

    def call(self, iteration, **kwargs):
        loss = {'loss_gen_a_identity': [],
                'loss_gen_b_identity': [],
                'loss_gen_a_translate': [],
                'loss_gen_b_translate': [],
                'loss_gen_adv_a': [],
                'loss_gen_adv_b': [],
                'loss_gen_a_content': [],
                'loss_gen_b_content': [],
                'loss_gen_a_identity_content': [],
                'loss_gen_b_identity_content': [],
                'loss_gen_aba_noise': [],
                'loss_gen_a_identity_noise': [],
                'loss_dis_a': [],
                'loss_dis_b': [],
                }
        dl_A, dl_B = DataLoader(self.data_set_A, batch_size=2, shuffle=False, num_workers=self.num_workers),\
                     DataLoader(self.data_set_B, batch_size=2, shuffle=False, num_workers=self.num_workers)
        for x_a, x_b in tqdm(zip(dl_A, dl_B), desc='Validation', total=len(dl_A)):
            x_a, x_b = x_a.float().cuda().detach(), x_b.float().cuda().detach()
            self.trainer.validate(x_a, x_b)
            loss['loss_gen_a_identity'] += [self.trainer.valid_loss_gen_a_identity.cpu().detach().numpy()]
            loss['loss_gen_b_identity'] += [self.trainer.valid_loss_gen_b_identity.cpu().detach().numpy()]
            loss['loss_gen_a_translate'] += [self.trainer.valid_loss_gen_a_translate.cpu().detach().numpy()]
            loss['loss_gen_b_translate'] += [self.trainer.valid_loss_gen_b_translate.cpu().detach().numpy()]
            loss['loss_gen_adv_a'] += [self.trainer.valid_loss_gen_adv_a.cpu().detach().numpy()]
            loss['loss_gen_adv_b'] += [self.trainer.valid_loss_gen_adv_b.cpu().detach().numpy()]
            loss['loss_gen_a_content'] += [self.trainer.valid_loss_gen_a_content.cpu().detach().numpy()]
            loss['loss_gen_b_content'] += [self.trainer.valid_loss_gen_b_content.cpu().detach().numpy()]
            loss['loss_gen_a_identity_content'] += [self.trainer.valid_loss_gen_a_identity_content.cpu().detach().numpy()]
            loss['loss_gen_b_identity_content'] += [self.trainer.valid_loss_gen_b_identity_content.cpu().detach().numpy()]
            loss['loss_gen_aba_noise'] += [self.trainer.valid_loss_gen_aba_noise.cpu().detach().numpy()]
            loss['loss_gen_a_identity_noise'] += [self.trainer.valid_loss_gen_a_identity_noise.cpu().detach().numpy()]
            loss['loss_dis_a'] += [self.trainer.valid_loss_dis_a.cpu().detach().numpy()]
            loss['loss_dis_b'] += [self.trainer.valid_loss_dis_b.cpu().detach().numpy()]

        self.loss['iteration'] += [iteration]
        self.loss['loss_gen_a_identity'].append(np.mean(loss['loss_gen_a_identity']))
        self.loss['loss_gen_b_identity'].append(np.mean(loss['loss_gen_b_identity']))
        self.loss['loss_gen_a_translate'].append(np.mean(loss['loss_gen_a_translate']))
        self.loss['loss_gen_b_translate'].append(np.mean(loss['loss_gen_b_translate']))
        self.loss['loss_gen_adv_a'].append(np.mean(loss['loss_gen_adv_a']))
        self.loss['loss_gen_adv_b'].append(np.mean(loss['loss_gen_adv_b']))
        self.loss['loss_gen_a_content'].append(np.mean(loss['loss_gen_a_content']))
        self.loss['loss_gen_b_content'].append(np.mean(loss['loss_gen_b_content']))
        self.loss['loss_gen_a_identity_content'].append(np.mean(loss['loss_gen_a_identity_content']))
        self.loss['loss_gen_b_identity_content'].append(np.mean(loss['loss_gen_b_identity_content']))
        self.loss['loss_gen_aba_noise'].append(np.mean(loss['loss_gen_aba_noise']))
        self.loss['loss_gen_a_identity_noise'].append(np.mean(loss['loss_gen_a_identity_noise']))
        self.loss['loss_dis_a'].append(np.mean(loss['loss_dis_a']))
        self.loss['loss_dis_b'].append(np.mean(loss['loss_dis_b']))


        self.plotAdversarial()
        self.plotContent()
        self.plotDistortion()
        self.plotNoise()

    def plotAdversarial(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.loss['iteration'], self.loss['loss_gen_adv_a'], label='Generator A')
        plt.plot(self.loss['iteration'], self.loss['loss_gen_adv_b'], label='Generator B')
        plt.plot(self.loss['iteration'], self.loss['loss_dis_a'], label='Discriminator A')
        plt.plot(self.loss['iteration'], self.loss['loss_dis_b'], label='Discriminator B')
        plt.legend()
        plt.savefig(os.path.join(self.path, "valid_progress_adversarial.jpg"), dpi=100)
        plt.close()

    def plotContent(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.loss['iteration'], self.loss['loss_gen_a_content'], label='Content A')
        plt.plot(self.loss['iteration'], self.loss['loss_gen_b_content'], label='Content B')
        plt.plot(self.loss['iteration'], self.loss['loss_gen_a_identity_content'], label='Content A Identity')
        plt.plot(self.loss['iteration'], self.loss['loss_gen_b_identity_content'], label='Content B Identity')
        plt.legend()
        plt.savefig(os.path.join(self.path, "valid_progress_content.jpg"), dpi=100)
        plt.close()

    def plotNoise(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.loss['iteration'], self.loss['loss_gen_a_identity_noise'], label='Noise A Identity')
        plt.plot(self.loss['iteration'], self.loss['loss_gen_aba_noise'], label='Noise ABA')
        plt.legend()
        plt.savefig(os.path.join(self.path, "valid_progress_noise.jpg"), dpi=100)
        plt.close()

    def plotDistortion(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.loss['iteration'], self.loss['loss_gen_a_translate'], label='MAE A')
        plt.plot(self.loss['iteration'], self.loss['loss_gen_b_translate'], label='MAE B')
        plt.plot(self.loss['iteration'], self.loss['loss_gen_a_identity_content'], label='MAE A Identity')
        plt.plot(self.loss['iteration'], self.loss['loss_gen_b_identity_content'], label='MAE B Identity')
        plt.legend()
        plt.savefig(os.path.join(self.path, "valid_progress_distortion.jpg"), dpi=100)
        plt.close()

class ProgressCallback(Callback):
    def __init__(self, trainer: Trainer, log_iteration=10):
        self.trainer = trainer
        super().__init__(log_iteration)

    def call(self, iteration, **kwargs):
        status = '[Iteration %08d] [D diff: %.03f/%.03f] [G loss: adv: %.03f/%.03f, recon: %.03f/%.03f, id: %.03f/%.03f, content: %.03f/%.03f]' % (
            iteration + 1,
            self.trainer.loss_dis_a, self.trainer.loss_dis_b,
            self.trainer.loss_gen_adv_a, self.trainer.loss_gen_adv_b,
            self.trainer.loss_gen_a_translate, self.trainer.loss_gen_b_translate,
            self.trainer.loss_gen_a_identity, self.trainer.loss_gen_b_identity,
            self.trainer.loss_gen_a_content, self.trainer.loss_gen_b_content,
        )
        logging.info('%s' % status)


class SaveCallback(Callback):
    def __init__(self, trainer: Trainer, checkpoint_dir, log_iteration=1000):
        self.trainer = trainer
        self.checkpoint_dir = checkpoint_dir
        super().__init__(log_iteration)

    def call(self, iteration, **kwargs):
        self.trainer.save(self.checkpoint_dir, iteration)


class NormScheduler(Callback):
    def __init__(self, trainer:Trainer, step=10000, gamma=0.5, init_iteration=0):
        self.trainer = trainer
        super(NormScheduler, self).__init__(step)
        self.momentum = 0.8
        self.gamma = gamma
        trainer.updateMomentum(self.momentum)
        [self.call(i) for i in range(init_iteration // step)]

    def call(self, iteration, **kwargs):
        if self.momentum == 0:
            return
        self.momentum *= self.gamma
        if self.momentum < 0.001:
            self.momentum = 0
        logging.info('Update normalization momentum: %.03f' % self.momentum)
        self.trainer.updateMomentum(self.momentum)

def running_mean(x, N=50):
    return np.convolve(x, np.ones((N,))/N, mode='valid')