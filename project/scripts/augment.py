import os
import glob
import shutil
import random
from pathlib import Path
import cv2
import albumentations as A

# --- Путь к исходному датасету ---
DATASET_ROOT = Path("project/dataset") 
IMAGES_DIR = DATASET_ROOT / "train" / "images"  # папка с исходными изображениями
LABELS_DIR = DATASET_ROOT / "train" / "labels"  # папка с исходными txt

# --- Путь для сохранения аугментированных данных ---
AUG_IMAGES_DIR = DATASET_ROOT / "images_aug" / "train"
AUG_LABELS_DIR = DATASET_ROOT / "labels_aug" / "train"

# --- Создаем папки для аугментированных данных ---
AUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUG_LABELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Определяем аугментации ---
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(
        translate_percent=(-0.1, 0.1),
        scale=(0.9, 1.1),
        rotate=(-15, 15),
        shear=(-10, 10),
        p=0.7
    ),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    A.GaussNoise(p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# --- Функция чтения bbox из YOLO txt ---
def read_yolo_bboxes(txt_path):
    boxes = []
    classes = []
    with open(txt_path, 'r') as f:
        for line in f:
            cls, x_center, y_center, w, h = line.strip().split()
            boxes.append([float(x_center), float(y_center), float(w), float(h)])
            classes.append(int(cls))
    return boxes, classes

# --- Функция записи bbox в YOLO txt ---
def write_yolo_bboxes(txt_path, boxes, classes):
    with open(txt_path, 'w') as f:
        for box, cls in zip(boxes, classes):
            f.write(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

# --- Главный цикл аугментации ---
image_paths = list(IMAGES_DIR.glob("*.*"))

AUG_FACTOR = 3  # во сколько раз увеличить датасет (сколько аугментаций сгенерировать на 1 исходное)

for img_path in image_paths:
    print(f"Images found: {len(list(IMAGES_DIR.glob('*.*')))}")
    print(f"Labels found: {len(list(LABELS_DIR.glob('*.txt')))}")
    stem = img_path.stem
    label_path = LABELS_DIR / f"{stem}.txt"
    if not label_path.exists():
        print(f"Warning: метка не найдена для {img_path.name}, пропускаем.")
        continue

    # Читаем изображение и bbox
    image = cv2.imread(str(img_path))
    boxes, class_labels = read_yolo_bboxes(label_path)

    # Копируем оригинал без изменений в новую папку (можно пропускать, если не нужно)
    shutil.copy2(img_path, AUG_IMAGES_DIR / img_path.name)
    shutil.copy2(label_path, AUG_LABELS_DIR / label_path.name)

    for i in range(AUG_FACTOR):
        try:
            augmented = transform(image=image, bboxes=boxes, class_labels=class_labels)
        except Exception as e:
            print(f"Ошибка аугментации {img_path.name} итерация {i}: {e}")
            continue

        aug_img = augmented['image']
        aug_boxes = augmented['bboxes']
        aug_classes = augmented['class_labels']

        # Пропускаем если аугментация удалила все bbox
        if len(aug_boxes) == 0:
            print(f"Warning: аугментация удалил все bbox в {img_path.name} итерация {i}, пропускаем.")
            continue

        # Сохраняем аугментированное изображение и разметку
        out_img_name = f"{stem}_aug{i}.jpg"
        out_label_name = f"{stem}_aug{i}.txt"

        cv2.imwrite(str(AUG_IMAGES_DIR / out_img_name), aug_img)
        write_yolo_bboxes(AUG_LABELS_DIR / out_label_name, aug_boxes, aug_classes)

print("Аугментация завершена! Новые данные сохранены в:")
print(f"Изображения: {AUG_IMAGES_DIR}")
print(f"Аннотации: {AUG_LABELS_DIR}")