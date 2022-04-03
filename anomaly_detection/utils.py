import albumentations as A
import cv2

import torch
import numpy as np
import random

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # https://hoya012.github.io/blog/reproducible_pytorch/ 참조


def trans_form(image, width, height):

    transform = A.Compose([
        A.RandomCrop(width=round(width*0.8),height=round(height*0.8),p=0.5),
        A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.8)
        ])

    transformed = transform(image=image)
    transformedImg = transformed["image"]
    # width, height 정보 필요시 주석제거
    # transformedHeight, transformedWidth = transformedImg.shape[0], transformedImg.shape[1]
    # transformedImg = cv2.resize(transformedImg, (width,height), interpolation=cv2.INTER_CUBIC)
    # 전체 데이터셋 resize도 고려해보기 
    return transformedImg
    