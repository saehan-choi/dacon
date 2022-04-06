import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import time
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt





#  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ augmentations test

# class AlbumentationsDataset(Dataset):
#     """__init__ and __len__ functions are the same as in TorchvisionDataset"""
#     def __init__(self, file_paths, labels, transform=None):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         label = self.labels[idx]
#         file_path = self.file_paths[idx]

#         image = cv2.imread(file_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         start_t = time.time()
#         if self.transform:
#             augmented = self.transform(image=image)
#             image = augmented['image']

#         total_time = (time.time() - start_t)
#         return image, label, total_time


# albumentations_transform_oneof = A.Compose([
#     A.Resize(256, 256), 
#     A.RandomCrop(224, 224),
#     A.OneOf([
#             A.HorizontalFlip(p=1),
#             A.RandomRotate90(p=1),
#             A.VerticalFlip(p=1)            
#     ], p=1),
#     A.OneOf([
#             A.MotionBlur(p=1),
#             A.OpticalDistortion(p=1),
#             A.GaussNoise(p=1)                 
#     ], p=1),
#     # A.Normalize(mean=0.5, std=0.5),
#     ToTensorV2()
# ])


# albumentations_dataset = AlbumentationsDataset(
#     file_paths=["./test.png"],
#     labels=[1],
#     transform=albumentations_transform_oneof,
# )

# print(albumentations_dataset[0][0])

# num_samples = 5
# fig, ax = plt.subplots(1, num_samples, figsize=(25, 5))
# for i in range(num_samples):
#   ax[i].imshow(transforms.ToPILImage()(albumentations_dataset[0][0]))
#   ax[i].axis('off')



#  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ augmentations test

# img = plt.imread('./anomaly_detection/test.png')

# transform = A.Rotate(p=1)
# transformed = transform(image=img)
# transformImg = transformed["image"]

# print(transformImg.shape)

# plt.imshow(transformImg)
# plt.xticks([]); plt.yticks([])
# plt.show()






#  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ augmentations test

# def transform(image, width, height):

#     transform = A.Compose([
#         # A.RandomCrop(width=round(width*0.8),height=round(height*0.8),p=0.5),

#         A.HorizontalFlip(p=0.8),
#         A.VerticalFlip(p=0.5),
#         A.Rotate(p=0.5),
#         ])

#     transformed = transform(image=image)
#     transformedImg = transformed["image"]

#     # width, height 정보 필요시 주석제거
#     # transformedHeight, transformedWidth = transformedImg.shape[0], transformedImg.shape[1]
#     # transformedImg = cv2.resize(transformedImg, (width,height), interpolation=cv2.INTER_CUBIC)
#     # 전체 데이터셋 resize도 고려해보기 

#     return transformedImg


import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import random
from utils import set_seed, transform_album
seed = 42
#  seed 고정됨을 확인했음. seed 만 빼면 엄청 다양한 이미지가 나오긴함.
def transform(img,seed):
    set_seed(seed)
    # img = cv2.imread(f'./anomaly_detection/dataset/train/{image}')
    # 이거 train으로 해야되네 ㅋㅋㅋㅋ 어차피 transform 은 trainset에만 적용되어야함.

    height, width = img.shape[:2]

    # angle = random.randint(-179,180)
    angle = random.randint(-45,45)
    scale = random.uniform(0.8,1)

    # 비율 불균형하게 잡는방법은 오른쪽왼쪽 절대크기 조절로 변하게한 다음에 계속하면 될듯
    # 비율을 그냥 내가 불균형하게 만들면됨.ㅎ... scale 비율을 조정할수있녜
    # cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None) -> dst

    # • src: 입력 영상
    # • dsize: 결과 영상 크기. (w, h) 튜플. (0, 0)이면 fx와 fy 값을 이용하여 결정.
    # • dst: 출력 영상
    # • fx, fy: x와 y방향 스케일 비율(scale factor). (dsize 값이 0일 때 유효)
    # • interpolation: 보간법 지정. 기본값은 cv2.INTER_LINEAR
    # scale을 랜덤하게 조정하면됨

    M1 = cv2.getRotationMatrix2D((height/2, width/2), angle=angle, scale=scale)
    aug_img = cv2.warpAffine(img, M1, (width, height))
    height, width = aug_img.shape[:2]

    transformed_Img = transform_album(aug_img)
    return transformed_Img

# 이게 시드를 고정하면 한 가지 밖에 안나옴
# 그리고 이미지 확대하는데 좌우 비율 안맞게 확대하는 augmentation 기법도 이용해야함.



# cv2.imwrite(f'./augmentation_ex/{i}.jpg', transformed_Img)

# cv2.imshow('res', transfomed_Img)
# cv2.waitKey(0)