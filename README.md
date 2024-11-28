# Introduction 
- This project detects faces in images, videos, and webcam feeds.
-  It utilizes an ONNX model for inference.
-  The project contains a test video and image present inside the test_inputs folder.
- All the outputs produced are dumped to output/ folder.

# Prequisites
- OS: Linux/Unix/ Windows

- Python : 3.10 \
**Use conda to create an environment**
```
conda create -n yolov8 python=3.10 -y
conda  activate yolov8
``` 
# Requirements

* Check the **requirements.txt** file.
* For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu** use 
```pip install onnxruntime-gpu```
, otherwise use the **onnxruntime** library use ```pip install onnxruntime```.

# How to run 
1. Clone the repository
```
git clone https://github.com/rayari-1729/ONNX-Face-Detection-using-YOLOv8/
cd yolov8-onnx-py
```
2. Install the requirements
```
pip install -r requirments.txt
```
3. Download the model
```

mkdir models
wget https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx -P models/

```
Convert the onnx model to fixed input dimensions
```
python onnx_convert_fixed_dims.py <path_to_onnx_model> <destination_path>

```
# Running Inference 

* Image Input : 
    ```
    python image_object_detection.py

* Video Input : 
    ```
    python video_object_detection.py
    ```
* Webcam Input : 
    ```
    python webcam_object_detection.py
    ```
    Note: Webcam mode requires a display port for .```'imshow' ```.


