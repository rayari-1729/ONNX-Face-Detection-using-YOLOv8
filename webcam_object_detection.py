import cv2

from yolov8 import YOLOv8

if __name__ == "__main__":

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    # Initialize YOLOv8 object detector
    model_path = "models/yolov8n-face-lindevs-fixed.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)
    while cap.isOpened():
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        # Update object localizer
        boxes, scores, class_ids = yolov8_detector(frame)
        combined_img = yolov8_detector.draw_detections(frame)
        cv2.imshow("Detected Objects", combined_img)
        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
