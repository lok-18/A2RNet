import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class fusion_dataset_gt(Dataset):
    def __init__(self, ir_path=None, vis_path=None, gt_path=None):
        super(fusion_dataset_gt, self).__init__()
        self.filepath_vis, self.filenames_vis = prepare_data_path(vis_path)
        self.filepath_ir, self.filenames_ir = prepare_data_path(ir_path)
        self.filepath_gt, self.filenames_gt = prepare_data_path(gt_path)
        self.length = min(len(self.filenames_vis), len(self.filenames_ir), len(self.filenames_gt))

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        gt_path = self.filepath_gt[index]
        image_vis = np.array(Image.open(vis_path))
        image_inf = cv2.imread(ir_path, 0)
        image_vis = (
                np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
        )
        image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
        image_ir = np.expand_dims(image_ir, axis=0)
        name = self.filenames_vis[index]

        image_gt = np.array(Image.open(gt_path))
        image_gt = (
                np.asarray(Image.fromarray(image_gt), dtype=np.float32).transpose(
                    (2, 0, 1)
                )
                / 255.0
        )

        return (
            torch.tensor(image_vis),
            torch.tensor(image_ir),
            torch.tensor(image_gt),
            name,
        )

    def __len__(self):
        return self.length
