import albumentations as A
import cv2

import imgaug as ia
import torch
import numpy as np
import random

def set_seed(random_seed):
    ia.seed(random_seed)
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
        # A.RandomCrop(width=round(width*0.9),height=round(height*0.9),p=0.5),
        # # randomcrop시 보여야할 부분이 안보이게 되는일이 없도록 처리할것.
        # A.HorizontalFlip(p=0.8),
        # A.VerticalFlip(p=0.5),
        
        
        
        # A.RandomContrast(p=1)
        # 이건 확인해보고 쓸지말지 결정 아마 색갈에 영향을 줄듯...안하는게좋겠다

        # A.RandomBrightnessContrast(p=0.8)
        # testset 확인결과 이건 없음


        ])

    transformed = transform(image=image)
    transformedImg = transformed["image"]
    # width, height 정보 필요시 주석제거
    # transformedHeight, transformedWidth = transformedImg.shape[0], transformedImg.shape[1]
    # transformedImg = cv2.resize(transformedImg, (width,height), interpolation=cv2.INTER_CUBIC)
    # 전체 데이터셋 resize도 고려해보기 
    return transformedImg
    