import os
import time

from tqdm import tqdm

import torch
import torch.nn as nn

from metrics.loss_metric import LossMetric
from metrics.accuracy_metric import AccuracyMetric
from metrics.classification_learning_curves import ClassificationLearningCurves


class ClassifierTrainer:
    def __init__(self, device, model, training_dataset, validation_dataset, output_path='', epoch_count=10,
                 learning_rate=0.01, batch_size=128):
        self._device = device
        self._output_path = output_path
        os.makedirs(self._output_path, exist_ok=True)

        self._epoch_count = epoch_count
        self._batch_size = batch_size

        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            print("DataParallel - GPU count:", torch.cuda.device_count())
            model = nn.DataParallel(model)

        self._model = model.to(device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, epoch_count)
        self._criterion = nn.CrossEntropyLoss()

        self._training_dataset_loader = torch.utils.data.DataLoader(training_dataset,
                                                                    batch_size=batch_size,
                                                                    shuffle=True,
                                                                    num_workers=4)

        self._validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset,
                                                                      batch_size=batch_size,
                                                                      shuffle=True,
                                                                      num_workers=4)

        self._training_loss_metric = LossMetric()
        self._training_accuracy_metric = AccuracyMetric()
        self._validation_loss_metric = LossMetric()
        self._validation_accuracy_metric = AccuracyMetric()
        self._learning_curves = ClassificationLearningCurves()

    def train(self):
        self._learning_curves.clear()

        for epoch in range(self._epoch_count):
            print('Training - Epoch [{}/{}]'.format(epoch + 1, self._epoch_count))
            time.sleep(0.1) # To prevent tqdm glitches
            self._train_one_epoch()

            print('\nValidation - Epoch [{}/{}]'.format(epoch + 1, self._epoch_count))
            time.sleep(0.1) # To prevent tqdm glitches
            self._validate()
            self._scheduler.step()

            self._print_performances()
            self._save_learning_curves()
            self._save_states(epoch + 1)

    def _train_one_epoch(self):
        self._training_loss_metric.clear()
        self._training_accuracy_metric.clear()

        self._model.train()

        for image, target in tqdm(self._training_dataset_loader):
            predicted_class_scores = self._model(image.to(self._device))
            target = target.to(self._device)

            loss = self._criterion(predicted_class_scores, target)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._training_loss_metric.add(loss.item())
            self._training_accuracy_metric.add(predicted_class_scores, target)

    def _validate(self):
        self._validation_loss_metric.clear()
        self._validation_accuracy_metric.clear()

        self._model.eval()

        for image, target in tqdm(self._validation_dataset_loader):
            predicted_class_scores = self._model(image.to(self._device))
            target = target.to(self._device)
            loss = self._criterion(predicted_class_scores, target)

            self._validation_loss_metric.add(loss.item())
            self._validation_accuracy_metric.add(predicted_class_scores, target)

    def _print_performances(self):
        print('\nTraining : Loss={}, Accuracy={}'.format(self._training_loss_metric.get_loss(),
                                                         self._training_accuracy_metric.get_accuracy()))
        print('Validation : Loss={}, Accuracy={}\n'.format(self._validation_loss_metric.get_loss(),
                                                           self._validation_accuracy_metric.get_accuracy()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_training_accuracy_value(self._training_accuracy_metric.get_accuracy())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_metric.get_accuracy())

        self._learning_curves.save_figure(os.path.join(self._output_path, 'learning_curves.png'))

    def _save_states(self, epoch):
        torch.save(self._model.state_dict(),
                   os.path.join(self._output_path, 'model_checkpoint_epoch_{}.pth'.format(epoch)))
