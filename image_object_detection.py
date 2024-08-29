import cv2
from yolov8 import YOLOv8

try:
    import os 
    os.makedirs("output")
except:
    pass
if __name__ == "__main__":

    # Initialize yolov8 object detector
    model_path = "models/yolov8n-face-lindevs-fixed.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)
    # loading image
    img_url = "test_inputs/test_image.jpg"
    img = cv2.imread(img_url)
    # Detect Objects
    boxes, scores, class_ids = yolov8_detector(img)
    # Draw detections
    combined_img = yolov8_detector.draw_detections(img)
    cv2.imwrite("output/detected_output.jpg", combined_img)
