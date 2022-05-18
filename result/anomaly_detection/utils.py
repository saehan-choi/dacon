import albumentations as A

# import imgaug as ia
import torch
import numpy as np
import random
import cv2

def set_seed(random_seed):
    # ia.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # https://hoya012.github.io/blog/reproducible_pytorch/ 참조


def transform_album(image, RandomCrop_P, HorizontalFlip_P, VerticalFlip_P):

    height, width = image.shape[:2]
    transform = A.Compose([
        A.RandomCrop(width=width, height=height, p=RandomCrop_P),

        # random crop을 width, height 동일하게 해야할듯..?
        # randomcrop시 보여야할 부분이 안보이게 되는일이 없도록 처리할것.
        A.HorizontalFlip(p=HorizontalFlip_P),
        A.VerticalFlip(p=VerticalFlip_P),
        # A.Blur(p=1),
        # 흠이것도.... 보류..
        # A.RandomBrightness(p=1)
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

def run_histogram_equalization(rgb_img):
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return equalized_img
