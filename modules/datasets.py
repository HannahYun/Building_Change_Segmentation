"""Datasets
"""

from torch.utils.data import Dataset
import numpy as np
import cv2
import os

class SegDataset(Dataset):
    """Dataset for image segmentation

    Attributs:
        x_dirs(list): 이미지 경로
        y_dirs(list): 마스크 이미지 경로
        input_size(list, tuple): 이미지 크기(width, height)
        scaler(obj): 이미지 스케일러 함수
        logger(obj): 로거 객체
        verbose(bool): 세부 로깅 여부
    """   
    def __init__(self, paths, input_size, scaler, mode='train', logger=None, verbose=False):
        
        self.x_paths = paths
        self.y_paths = list(map(lambda x : x.replace('x', 'y'),self.x_paths))
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.mode = mode


    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, id_: int):
        
        filename = os.path.basename(self.x_paths[id_]) # Get filename for logging
        x = cv2.imread(self.x_paths[id_], cv2.IMREAD_COLOR)
        orig_size = x.shape

        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, self.input_size)
        x = self.scaler(x)
        x = np.transpose(x, (2, 0, 1))

        if self.mode in ['train', 'valid']:
            y = cv2.imread(self.y_paths[id_], cv2.IMREAD_GRAYSCALE)
            y = cv2.resize(y, self.input_size, interpolation=cv2.INTER_NEAREST)

            return x, y, filename

        elif self.mode in ['test']:
            return x, orig_size, filename

        else:
            assert False, f"Invalid mode : {self.mode}"

class SegDataset_TTA(Dataset):
    """Dataset for image segmentation

    Attributs:
        x_dirs(list): 이미지 경로
        y_dirs(list): 마스크 이미지 경로
        input_size(list, tuple): 이미지 크기(width, height)
        scaler(obj): 이미지 스케일러 함수
        logger(obj): 로거 객체
        verbose(bool): 세부 로깅 여부
    """   
    def __init__(self, paths, input_size, scaler, mode='train', logger=None, verbose=False):
        
        self.x_paths = paths
        self.y_paths = list(map(lambda x : x.replace('x', 'y'),self.x_paths))
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.mode = mode


    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, id_: int):
        
        filename = os.path.basename(self.x_paths[id_]) # Get filename for logging
        x = cv2.imread(self.x_paths[id_], cv2.IMREAD_COLOR)
        orig_size = x.shape

        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        im1 = x[:, :754]
        im2 = x[:, 754:]
        
        im1 = cv2.resize(im1, (224, 224), interpolation=cv2.INTER_NEAREST)
        im2 = cv2.resize(im2, (224, 224), interpolation=cv2.INTER_NEAREST)
        x = cv2.hconcat([im1, im2])
        
        im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
        im2 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)
        x2 = cv2.hconcat([im1, im2])
        
        im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
        im2 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)
        x3 = cv2.hconcat([im1, im2])
        
        im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
        im2 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)
        x4 = cv2.hconcat([im1, im2])
        
        
        x = self.scaler(x)
        x2 = self.scaler(x2)
        x3 = self.scaler(x3)
        x4 = self.scaler(x4)

        x = np.transpose(x, (2, 0, 1))
        x2 = np.transpose(x2, (2, 0, 1))
        x3 = np.transpose(x3, (2, 0, 1))
        x4 = np.transpose(x4, (2, 0, 1))

        if self.mode in ['train', 'valid']:
            y = cv2.imread(self.y_paths[id_], cv2.IMREAD_GRAYSCALE)
            y = cv2.resize(y, self.input_size, interpolation=cv2.INTER_NEAREST)

            return x, y, filename

        elif self.mode in ['test']:
            return x, x2, x3, x4, orig_size, filename

        else:
            assert False, f"Invalid mode : {self.mode}"

