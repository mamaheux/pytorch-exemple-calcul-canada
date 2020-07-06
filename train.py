import argparse

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.vanilla_cnn import VanillaCnn
from models.dense_block_cnn import DenseBlockCnn

from classifier_trainer import ClassifierTrainer

def main():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'svhn'], help='Choose the dataset', required=True)
    parser.add_argument('--model', choices=['vanilla_cnn', 'dense_block_cnn'], help='Choose the model', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()

    training_dataset, validation_dataset, class_count = create_datasets(args.dataset, args.dataset_root)
    model = create_model(args.model, class_count, args.model_checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    trainer = ClassifierTrainer(device, model, training_dataset, validation_dataset,
                                output_path=args.output_path,
                                epoch_count=args.epoch_count,
                                learning_rate=args.learning_rate,
                                batch_size=args.batch_size)
    trainer.train()


def create_datasets(dataset, dataset_root):
    training_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if dataset == 'cifar10':
        training_dataset = datasets.CIFAR10(dataset_root, train=True, transform=training_transform, download=True)
        validation_dataset = datasets.CIFAR10(dataset_root, train=False, transform=validation_transform, download=True)
        class_count = 10
    elif dataset == 'cifar100':
        training_dataset = datasets.CIFAR100(dataset_root, train=True, transform=training_transform, download=True)
        validation_dataset = datasets.CIFAR100(dataset_root, train=False, transform=validation_transform, download=True)
        class_count = 100
    elif dataset == 'svhn':
        training_dataset = datasets.SVHN(dataset_root, split='train', transform=training_transform, download=True)
        validation_dataset = datasets.SVHN(dataset_root, split='test', transform=validation_transform, download=True)
        class_count = 10
    else:
        raise ValueError('Invalid dataset')

    return training_dataset, validation_dataset, class_count


def create_model(model, class_count, model_checkpoint):
    if model == 'vanilla_cnn':
        model = VanillaCnn(class_count=class_count, use_softmax=False)
    elif model == 'dense_block_cnn':
        model = DenseBlockCnn(class_count=class_count, use_softmax=False)
    else:
        raise ValueError('Invalid dataset')

    if model_checkpoint is not None:
        model.load_state_dict(torch.load(model_checkpoint))

    return model


if __name__ == '__main__':
    main()
