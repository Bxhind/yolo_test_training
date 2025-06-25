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
        name="cosine_advanced",
        model="runs/detect/YOLOv11_baseline/weights/best.pt",  # Лучшая модель из v2
        data="project/dataset/data.yaml",
        
        # ── Оптимизированные базовые параметры ──
        epochs=75,          # Увеличено для лучшей стабилизации
        imgsz=640,
        batch=12,           # Уменьшено для более точных градиентов
        device="mps",
        
        cos_lr=True,        # Включаем cosine learning rate scheduler
        lr0=0.008,          # Немного снижен стартовый LR для стабильности  
        lrf=0.0001,  
        
        warmup_epochs=5.0,  # Увеличен warmup для лучшей адаптации
        warmup_momentum=0.8,
        momentum=0.95,      # Увеличен momentum для более стабильных градиентов
        weight_decay=0.0008, # Немного увеличен для регуляризации
        
        auto_augment="randaugment", 
        erasing=0.3,        # Немного снижено
        mixup=0.15,         # Добавляем MixUp
        copy_paste=0.1,     # Copy-paste для малых объектов
        degrees=8.0,        
        translate=0.12,     
        scale=0.6,          
        shear=3.0,          
        perspective=0.0001, 
        
        hsv_h=0.018,        # Немного увеличен hue shift
        hsv_s=0.75,         # Увеличена saturation
        hsv_v=0.45,         # Увеличен value range
        
        close_mosaic=15,    # Отключаем мозаику раньше для fine-tuning
        amp=True,           # Mixed precision для скорости
        
        conf=0.22,          # Немного снижен confidence threshold
        iou=0.65,           # Снижен IoU для less aggressive NMS
        
        
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
    