import cv2
from ultralytics import YOLO

def get_class_colors():
    """Возвращает красивые цвета для разных классов"""
    colors = [
        (255, 87, 51),   # Красно-оранжевый
        (46, 204, 113),  # Зеленый
        (52, 152, 219),  # Синий
        (155, 89, 182),  # Фиолетовый
        (241, 196, 15),  # Желтый
        (230, 126, 34),  # Оранжевый
        (231, 76, 60),   # Красный
        (26, 188, 156),  # Бирюзовый
        (142, 68, 173),  # Темно-фиолетовый
        (39, 174, 96),   # Темно-зеленый
    ]
    return colors

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=10):
    """Рисует прямоугольник с округленными углами"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def draw_text_with_background(img, text, position, font_scale=0.7, thickness=2, 
                            text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Рисует текст с красивым фоном"""
    font = cv2.FONT_HERSHEY_DUPLEX
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    padding = 8
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding//2), 
                  bg_color, -1)
    
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding//2), 
                  text_color, 1)
    
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

def main(video_path, model_path='runs/detect/YOLOv11_upd_dataset_annotations_no_preprocessing_augment/weights/best.pt'):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    class_colors = get_class_colors()

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 кодек
    out = None
    
    frame_count = 0

    previous_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        frame = cv2.bilateralFilter(frame, 9, 75, 75)  
        
        results = model(frame)
        frame_count += 1

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                class_name = model.names[class_id]
                
                color = class_colors[class_id % len(class_colors)]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                inner_color = tuple(min(255, c + 30) for c in color)
                cv2.rectangle(frame, (x1+1, y1+1), (x2-1, y2-1), inner_color, 1)
                
                confidence_percent = int(score * 100)
                label = f"{class_name}: {confidence_percent}%"
                
                draw_text_with_background(frame, label, (x1, y1 - 10), 
                                        font_scale=0.8, thickness=2,
                                        text_color=(255, 255, 255), bg_color=color)
                
                confidence_bar_width = int((x2 - x1) * score)
                cv2.rectangle(frame, (x1, y2 + 2), (x1 + confidence_bar_width, y2 + 8), color, -1)
                cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 8), color, 1)

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter('output_vertical_hq.mp4', fourcc, 60.0, (w, h))

        out.write(frame)
        
        if previous_frame is not None and frame_count % 2 == 0:  
            alpha = 0.5
            interpolated_frame = cv2.addWeighted(previous_frame, alpha, frame, 1-alpha, 0)
            out.write(interpolated_frame)
        
        previous_frame = frame.copy()
        
        display_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        cv2.imshow('YOLOv11 Detection - 60FPS', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Обработано {frame_count} кадров из видео {video_path}")

if __name__ == '__main__':
    import sys
    video_path = "data/videos/4.MOV"
    main(video_path)