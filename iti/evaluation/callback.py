import logging
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
import torch
from matplotlib import pyplot as plt

from iti.train.trainer import Trainer


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

    def __init__(self, data, model: Trainer, path, plot_id, plot_settings, dpi=100, **kwargs):
        self.data = data
        self.path = path
        self.model = model
        self.plot_settings = plot_settings
        self.dpi = dpi
        self.plot_id = plot_id

        super().__init__(**kwargs)

    def call(self, iteration, **kwargs):
        data = self.loadData()

        rows = len(data)
        columns = len(data[0])

        f, axarr = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
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
            data = torch.from_numpy(self.data).float().cuda().detach()
            prediction = self.predict(data)
            batch_data = [data, ] + [*prediction]
            return [[d[j, i].cpu().detach().numpy() for d in batch_data for i in range(d.shape[1])] for j in
                    range(len(self.data))]

    def predict(self, input_data):
        raise NotImplementedError()


class ProgressPlot(Callback):

    def __init__(self, path, dpi=100, **kwargs):
        self.path = path
        self.dpi = dpi

        super().__init__(**kwargs)

    def call(self, iteration, **kwargs):
        history = kwargs["history"]
        model = kwargs['model']
        if len(history["loss"]) <= 200:
            return
        x = range(1, len(history["loss"]) + 1)[200:]  # SKIP first 200 entries

        plt.plot(x, np.array(history["content_A_loss"][200:]) * model.lambda_content, label="Content Loss A")
        plt.plot(x, np.array(history["content_B_loss"][200:]) * model.lambda_content, label="Content Loss B")
        plt.plot(x, np.array(history["reconstruction_A_loss"][200:]) * model.lambda_reconstruction,
                 label="Reconstruction A")
        plt.plot(x, np.array(history["reconstruction_B_loss"][200:]) * model.lambda_reconstruction,
                 label="Reconstruction B")
        plt.plot(x, np.array(history["identity_A_loss"][200:]) * model.lambda_reconstruction_id, label="Identity A")
        plt.plot(x, np.array(history["identity_B_loss"][200:]) * model.lambda_reconstruction_id, label="Identity B")
        plt.plot(x, np.array(history["id_content_A_loss"][200:]) * model.lambda_content_id,
                 label="Identity Content Loss A")
        plt.plot(x, np.array(history["id_content_B_loss"][200:]) * model.lambda_content_id,
                 label="Identity Content Loss B")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.path, "progress_distortion.jpg"), dpi=self.dpi)
        plt.close()

        disc_A_loss = history["generator_discriminator_A_loss"][200:]
        disc_B_loss = history["generator_discriminator_B_loss"][200:]
        plt.plot(x, disc_A_loss, label="Discriminator A")
        plt.plot(x, disc_B_loss, label="Discriminator B")
        disc_loss = np.concatenate((disc_A_loss, disc_B_loss))
        plt.ylim((np.mean(disc_loss) - 3 * np.std(disc_loss), np.mean(disc_loss) + 3 * np.std(disc_loss)))
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.path, "progress_perception.jpg"), dpi=self.dpi)
        plt.close()

        fake_loss = history["combined_discriminator_A_fake_0_loss"][200:]
        real_loss = history["combined_discriminator_A_real_0_loss"][200:]
        disc_loss = np.add(real_loss, fake_loss)
        plt.plot(x, fake_loss, label="Fake")
        plt.plot(x, real_loss, label="Real")
        # plt.plot(x, disc_loss, label="Distance")
        # plt.ylim((np.mean(disc_loss) - 3 * np.std(disc_loss), np.mean(disc_loss) + 3 * np.std(disc_loss)))
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.path, "progress_dA.jpg"), dpi=self.dpi)
        plt.close()

        fake_loss = history["combined_discriminator_B_fake_0_loss"][200:]
        real_loss = history["combined_discriminator_B_real_0_loss"][200:]
        disc_loss = np.add(real_loss, fake_loss)
        plt.plot(x, fake_loss, label="Fake")
        plt.plot(x, real_loss, label="Real")
        # plt.plot(x, disc_loss, label="Distance")
        # plt.ylim((np.mean(disc_loss) - 3 * np.std(disc_loss), np.mean(disc_loss) + 3 * np.std(disc_loss)))
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.path, "progress_dB.jpg"), dpi=self.dpi)
        plt.close()


class ValidationProgressPlot(Callback):

    def __init__(self, path, dpi=100, **kwargs):
        self.path = path
        self.dpi = dpi

        super().__init__(**kwargs)

    def call(self, iteration, **kwargs):
        history = kwargs["validation_history"]
        x = range(self.log_iteration, (len(history["disc_A"]) + 1) * self.log_iteration, self.log_iteration)

        plt.plot(x, history["disc_A"], label="Discriminator A")
        plt.plot(x, history["disc_B"], label="Discriminator B")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.path, "validation_disc.jpg"), dpi=self.dpi)
        plt.close()

        plt.plot(x, history["ssim_A"], label="SSIM A")
        plt.plot(x, history["ssim_B"], label="SSIM B")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.path, "validation_ssim.jpg"), dpi=self.dpi)
        plt.close()

        plt.scatter(1 - np.array(history["ssim_A"]), history["disc_A"], label="A")
        plt.scatter(1 - np.array(history["ssim_B"]), history["disc_B"], label="B")
        for i in range(len(history["disc_A"])):
            plt.annotate(str(i + 1), (1 - np.array(history["ssim_A"])[i], history["disc_A"][i]))
            plt.annotate(str(i + 1), (1 - np.array(history["ssim_B"])[i], history["disc_B"][i]))
        plt.xlabel("Distortion (1 - SSIM)")
        plt.ylabel("Perception (W-Distance)")
        plt.gca().set_ylim(bottom=0)
        plt.xlim([0, 1])
        plt.legend()
        plt.savefig(os.path.join(self.path, "perception_distortion.jpg"), dpi=self.dpi)
        plt.close()


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

        plot_settings_A = plot_settings_A if plot_settings_A is not None else [{"cmap": "gray"}] * model.input_dim_a * n_samples
        plot_settings_A = plot_settings_A if isinstance(plot_settings_A, list) else [plot_settings_A] * model.input_dim_a * n_samples
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
        self.history_path = os.path.join(path, 'history.pickle')

        self.loss = {'loss_gen_a_identity': [],
                     'loss_gen_b_identity': [],
                     'loss_gen_a_translate': [],
                     'loss_gen_b_translate': [],
                     'loss_gen_adv_a': [],
                     'loss_gen_adv_b': [],
                     'loss_gen_a_content': [],
                     'loss_gen_b_content': [],
                     'loss_gen_a_identity_content': [],
                     'loss_gen_b_identity_content': [],
                     'loss_gen_ba_noise': [],
                     'loss_gen_aba_noise': [],
                     'loss_gen_a_identity_noise': [],
                     'loss_dis_a': [],
                     'loss_dis_b': [],
                     'loss_gen_diversity': []
                     }
        if os.path.exists(self.history_path):
            with open(self.history_path, 'rb') as f:
                self.loss = pickle.load(f)
        super().__init__(log_iteration)

    def call(self, iteration, **kwargs):
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
        self.loss['loss_gen_ba_noise'] += [self.trainer.loss_gen_ba_noise.cpu().detach().numpy()]
        self.loss['loss_gen_aba_noise'] += [self.trainer.loss_gen_aba_noise.cpu().detach().numpy()]
        self.loss['loss_gen_a_identity_noise'] += [self.trainer.loss_gen_a_identity_noise.cpu().detach().numpy()]
        self.loss['loss_dis_a'] += [self.trainer.loss_dis_a.cpu().detach().numpy()]
        self.loss['loss_dis_b'] += [self.trainer.loss_dis_b.cpu().detach().numpy()]
        self.loss['loss_gen_diversity'] += [self.trainer.loss_gen_diversity.cpu().detach().numpy()]
        if (iteration + 1) % 100 == 0:
            self.plotAdversarial()
            self.plotContent()
            self.plotDistortion()
        with open(self.history_path, 'wb') as f:
            pickle.dump(self.loss, f)

    def plotAdversarial(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.loss['loss_gen_adv_a'], label='Generator A')
        plt.plot(self.loss['loss_gen_adv_b'], label='Generator B')
        plt.plot(self.loss['loss_dis_a'], label='Discriminator A')
        plt.plot(self.loss['loss_dis_b'], label='Discriminator B')
        plt.legend()
        plt.savefig(os.path.join(self.path, "progress_adversarial.jpg"), dpi=100)
        plt.close()

    def plotContent(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.loss['loss_gen_a_content'], label='Content A')
        plt.plot(self.loss['loss_gen_b_content'], label='Content B')
        plt.plot(self.loss['loss_gen_a_identity_content'], label='Content A Identity')
        plt.plot(self.loss['loss_gen_b_identity_content'], label='Content B Identity')
        plt.plot(self.loss['loss_gen_a_identity_noise'], label='Noise A Identity')
        plt.plot(self.loss['loss_gen_ba_noise'], label='Noise BA')
        plt.plot(self.loss['loss_gen_aba_noise'], label='Noise ABA')
        plt.plot(self.loss['loss_gen_diversity'], label='Diversity')
        plt.legend()
        plt.savefig(os.path.join(self.path, "progress_content.jpg"), dpi=100)
        plt.close()

    def plotDistortion(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.loss['loss_gen_a_translate'], label='MAE A')
        plt.plot(self.loss['loss_gen_b_translate'], label='MAE B')
        plt.plot(self.loss['loss_gen_a_identity_content'], label='MAE A Identity')
        plt.plot(self.loss['loss_gen_b_identity_content'], label='MAE B Identity')
        plt.legend()
        plt.savefig(os.path.join(self.path, "progress_distortion.jpg"), dpi=100)
        plt.close()

class ProgressCallback(Callback):
    def __init__(self, trainer: Trainer, log_iteration=1):
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
