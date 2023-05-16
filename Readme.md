# Object Detection and Speed Estimation

This project aims to develop an advanced system that performs real-time object detection using YOLOv5 and accurate speed estimation using Lucas-Kanade optical flow. The system combines these capabilities to enable applications such as surveillance, autonomous systems, and traffic monitoring.

## Features

- Real-time object detection using YOLOv5.
- Accurate speed estimation of detected objects using Lucas-Kanade optical flow.
- Seamless integration of object detection and speed estimation for enhanced analysis.
- Handling of challenges such as occlusions, object ID switches, and track fragmentation.
- Performance evaluation measures including precision, recall, mAP, and velocity estimation accuracy.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rohithps98/Speed-Estimation-using-Lucas-Kanade.git

## Install the required dependencies

pip install -r requirements.txt

## Download the YOLOv5 pre-trained weights

cd object-detection-speed-estimation
python download_weights.py

## Usage
- Prepare your video dataset in a compatible format.

- Run the object detection and speed estimation pipeline
python main.py --video <path_to_video_file>

- Monitor the console output for object detection bounding boxes and speed estimation results.

## Evaluation
To evaluate the performance of the system, follow these steps:

- Prepare a separate evaluation dataset with ground truth annotations for object detection and known object speeds.

- Run the evaluation script:
python evaluate.py --detections <path_to_detection_results> --gt <path_to_ground_truth_annotations>

- The evaluation script will output metrics such as precision, recall, mAP, and velocity estimation accuracy.

## Contributing
Contributions to the project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
