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

#     # width, height ?????? ????????? ????????????
#     # transformedHeight, transformedWidth = transformedImg.shape[0], transformedImg.shape[1]
#     # transformedImg = cv2.resize(transformedImg, (width,height), interpolation=cv2.INTER_CUBIC)
#     # ?????? ???????????? resize??? ??????????????? 

#     return transformedImg


import cv2
import random
from utils import set_seed, transform_album
# seed = 42
# #  seed ???????????? ????????????. seed ??? ?????? ?????? ????????? ???????????? ????????????.
# def transform(img,seed):
#     # set_seed(seed)
#     # img = cv2.imread(f'./anomaly_detection/dataset/train/{image}')
#     # ?????? train?????? ???????????? ???????????? ????????? transform ??? trainset?????? ??????????????????.

#     height, width = img.shape[:2]
#     angle = random.randint(-45,45)
#     scale = random.uniform(0.8,1)
#     M1 = cv2.getRotationMatrix2D((height, width), angle=angle, scale=scale)
#     aug_img = cv2.warpAffine(img, M1, (width, height))
#     return aug_img


import os
import time

path_ = './anomaly_detection/dataAnalysis/train_with_affine_aug/'
path_listdir = os.listdir(path_)

print(path_listdir)

for m in path_listdir:

    path = f'./anomaly_detection/dataAnalysis/train_with_affine_aug/{m}/'
    paths = os.listdir(path)

    for j in paths:
        paths2 = os.listdir(path+j)
        for k in paths2:
            # print(path+j+'/'+k)
            img = cv2.imread(path+j+'/'+k, cv2.IMREAD_COLOR)
            height, width = img.shape[:2]

            M1 = cv2.getRotationMatrix2D((height/2, width/2), angle=45, scale=1)
            aug_img = cv2.warpAffine(img, M1, (width, height))

            cv2.imwrite(path+j+'/'+'aug_affine_45'+k,aug_img)

            M1 = cv2.getRotationMatrix2D((height/2, width/2), angle=20, scale=1)
            aug_img = cv2.warpAffine(img, M1, (width, height))
            cv2.imwrite(path+j+'/'+'aug_affine_20'+k,aug_img)
            
    print(f'{path} is flip')
