import albumentations as A
import cv2

def transform(image, width, height):

    transform = A.Compose([
        A.RandomCrop(width=round(width*0.8),height=round(height*0.8),p=0.5),
        A.HorizontalFlip(p=0.8),
        A.RandomBrightnessContrast(p=0.8)
        ])

    transformed = transform(image=image)
    transformedImg = transformed["image"]
    
    # width, height 정보 필요시 주석제거
    # transformedHeight, transformedWidth = transformedImg.shape[0], transformedImg.shape[1]
    # transformedImg = cv2.resize(transformedImg, (width,height), interpolation=cv2.INTER_CUBIC)
    # 전체 데이터셋 resize도 고려해보기 

    return transformedImg
