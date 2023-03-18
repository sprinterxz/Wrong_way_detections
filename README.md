# Wrong_way_detections_UI
wrong wat detections UI counting a vehicle wth yolov5 deepsort and Pyqt5

## Prepare
1. Clone the repository recursively.
~~~
git clone https://github.com/sprinterxz/Wrong_way_detections.git
Cd Wrong_way_detections
~~~

2 Create a virtual environment with Python >=3.9
~~~
python -m venv Wrong_way_detections
#activate venv
Wrong_way_detections_test\Scripts\activate.bat
~~~

3 Install requirements with mentioned command below.
~~~
pip install -r requirements.txt
#install torch and torchvision
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
~~~

4 For other YOLOv5 models, you can be downloaded at https://github.com/ultralytics/yolov5
~~~
#Place the file in the "wrong_way_detection/yolo_model" directory.
~~~

5 Run main.py to use it right away.
