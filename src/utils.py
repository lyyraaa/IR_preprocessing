import numpy as np
import torch
import h5py
import cv2
import math

# Torch dataset that loads entire cores at once
# This is a much simpler dataset class, more akin to a traditional image dataset
class ir_core_dataset_torch(torch.utils.data.Dataset):
    def __init__(self,
            ir_filepaths,
            mask_filepaths,
            annotation_filepaths,
            annotation_class_names,
            annotation_class_colors,
            augment=False,
            generator=None):
        self.ir_filepaths = ir_filepaths
        self.mask_filepaths = mask_filepaths
        self.annotation_filepaths = annotation_filepaths
        self.augment = augment
        self.generator = generator

        # class data
        self.annotation_class_colors = annotation_class_colors
        self.annotation_class_names = annotation_class_names

    def __len__(self):
        return len(self.ir_filepaths)

    # split annotations from H x W x 3 to C x H x W, one/zerohot along C dimension
    def split_annotations(self, annotations_img):
        split = torch.zeros((len(self.annotation_class_colors), *annotations_img.shape[:-1]))
        for c, col in enumerate(self.annotation_class_colors):
            split[c, :, :] = torch.from_numpy(np.all(annotations_img == self.annotation_class_colors[c], axis=-1))
        return split

    def __getitem__(self, idx):
        # get mask
        mask = torch.from_numpy(
            cv2.imread(self.mask_filepaths[idx])[...,0] / 255.0,
        ).unsqueeze(0)

        # get ir
        ir = torch.load(self.ir_filepaths[idx])
        ir *= mask

        # get annotations
        annotations = self.split_annotations(cv2.imread(self.annotation_filepaths[idx])[:, :, ::-1])
        annotations *= mask
        has_annotations = annotations.sum(dim=0) != 0

        if self.augment:
            to_aug = torch.rand((2,),generator=self.generator)
            if to_aug[0] > 0.5:  # hflip
                ir = torch.flip(ir, (-1,))
                annotations = torch.flip(annotations, (-1,))
                has_annotations = torch.flip(has_annotations, (-1,))
                mask = torch.flip(mask, (-1,))
            if to_aug[1] > 0.5:  # vflip
                ir = torch.flip(ir, (-2,))
                annotations = torch.flip(annotations, (-2,))
                has_annotations = torch.flip(has_annotations, (-2,))
                mask = torch.flip(mask, (-2,))

        return ir, annotations, mask, has_annotations

class ir_spectra_dataset(torch.utils.data.Dataset):
    def __init__(self,
            ir_spectra,
            tissue_labels,
            annotation_class_names,
            annotation_class_colors,
            generator = None):

        self.spectra = ir_spectra
        self.labels = torch.from_numpy(tissue_labels)

        self.annotation_class_names = annotation_class_names
        self.annotation_class_colors = annotation_class_colors
        self.generator = generator

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx], self.labels[idx]

def statistics_from_loader(loader,num_features):

    running_sum = torch.zeros((num_features,), dtype=torch.float32)
    running_square = torch.zeros((num_features,), dtype=torch.float32)
    running_pix_total = 0

    for data, _, mask, _ in loader:
        data = data * mask
        running_sum += data.sum((0,2,3))
        running_square += torch.square(data).sum((0, 2, 3))
        running_pix_total += mask.sum().item()

    mean = running_sum / running_pix_total
    variance = (running_square / running_pix_total) - ((2 * mean * running_sum) / running_pix_total) + torch.square(mean)

    return mean, torch.sqrt(variance), running_pix_total

def adjust_learning_rate(optimizer, epoch, max_epochs, warmup_epochs, initial_lr, min_lr): # taken from https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = initial_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (initial_lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr