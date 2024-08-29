import cv2
from yolov8 import YOLOv8

try:
    import os 
    os.makedirs("output")
except:
    pass
if __name__ == "__main__":
    cap = cv2.VideoCapture("test_inputs/test_video.mp4")

    # Initialize  model
    model_path = "models/yolov8n-face-lindevs-fixed.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)
    out = cv2.VideoWriter('output/detected_output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        try:
            # Read frame from the video
            ret, frame = cap.read()
            if not ret:
                break
        except Exception as e:
            print(e)
            continue
        # Update object localizer
        boxes, scores, class_ids = yolov8_detector(frame)
        combined_img = yolov8_detector.draw_detections(frame)
        # Write the frame to the output video file
        out.write(combined_img)
    

