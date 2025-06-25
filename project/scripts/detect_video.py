import cv2
from ultralytics import YOLO

def main(video_path, model_path='project/best.pt'):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        # Рисуем bbox на кадре
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # xyxy координаты
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]}: {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Инициализируем запись видео один раз, когда известен размер
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))

        out.write(frame)
        cv2.imshow('YOLOv11 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    video_path = "data/videos/2_1.MOV"
    main(video_path)