# train.py – пакетное обучение и логирование метрик для YOLOv11
#
# Пакеты: pip install ultralytics==8.* matplotlib pandas

from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import datetime as dt

# Добавляйте/меняйте словари здесь – никаких CLI-аргументов не надо.
EXPERIMENTS = [
    dict(  # 🎯 Продвинутый эксперимент с оптимизированными параметрами
        name="upd_dataset_annotations_no_preprocessing_augment",  # Название эксперимента
        model="runs/detect/YOLOv11_upd_dataset_annotations_preprocessing_augment/weights/best.pt",  # Путь к предобученной модели 
        data="project/dataset/data.yaml",
        
        # ── Оптимизированные базовые параметры ──
        epochs=30,          # Увеличено для лучшей стабилизации
        imgsz=512,         # Увеличено для лучшего качества
        batch=8,           # Уменьшено для более точных градиентов
        device="mps",
        
        cos_lr=False, 
        
        warmup_epochs=2.0,  # Увеличен warmup для лучшей адаптации
        warmup_momentum=0.8,
        momentum=0.95,      # Увеличен momentum для более стабильных градиентов
        weight_decay=0.0008, # Немного увеличен для регуляризации
        
        auto_augment="randaugment", 
        erasing=0.3,        # Немного снижено
        mixup=0.15,         # Добавляем MixUp
        degrees=8.0,        
        translate=0.12,     
        scale=0.6,          
        shear=3.0,          
        perspective=0.0001, 
        
        hsv_h=0.018,        # Немного увеличен hue shift
        hsv_s=0.75,         # Увеличена saturation
        hsv_v=0.45,         # Увеличен value range
        
        close_mosaic=15,    # Отключаем мозаику раньше для fine-tuning
        amp= False,          # Mixed precision для скорости
        
        conf=0.20,          # Немного снижен confidence threshold
        iou=0.60,           # Снижен IoU для less aggressive NMS
        
        
        box=8.0,            
        cls=0.6,            
        dfl=1.8,            
    )
]

ROOT = Path.cwd()
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_CSV = ROOT / "results.csv"

summary_rows = []

for exp in EXPERIMENTS:
    name = exp["name"]
    run_id = f"YOLOv11_{name}"
    print(f"\n=== 🚀 Старт эксперимента {run_id} ===")
    
    # Загружаем модель (весы создадутся заново, если передан YAML-конфиг)
    model = YOLO(exp["model"])
    
    # Параметры обучения; отсекаем ключи, не относящиеся к train()
    train_kwargs = {
        k: v for k, v in exp.items()
        if k in {"data", "epochs", "batch", "imgsz", "device", "lr0"}
    }
    # Префикс run'а, чтобы Ultralytics писал в свою подпапку
    train_kwargs["name"] = run_id
    
    results = model.train(**train_kwargs)
    