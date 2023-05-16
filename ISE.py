import cv2
import numpy as np
import time
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
prev_time = time.time()

# video file
vid_file = '/Users/rohith/Desktop/thesis/videos/video.mov'

# Load input video
cap = cv2.VideoCapture(vid_file)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # Codec used to encode video frames
out = cv2.VideoWriter('my.avi', fourcc, 20.0, (frame_width, frame_height),
                      True)  # Output video file path, codec, frame rate, and frame size

fps = cap.get(cv2.CAP_PROP_FPS)
classes = model.module.names if hasattr(model, 'module') else model.names

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(20, 20),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))

# Initialize reference frame and object depth dictionary
ret, frame_ref = cap.read()
frame_ref_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)

avg_running_time = 0
num_frames = 0

while True:
    # Read input video frame
    ret, frame = cap.read()
    if not ret:
        break
    t0 = time.time()
    # Object detection using YOLOv5
    results = model(frame)
    boxes = results.xyxy[0][:, :4].tolist()
    labels = results.xyxy[0][:, -1].tolist()
    confidences = results.xyxy[0][:, 4].cpu().numpy()
    class_ids = np.array(results.xyxy[0])[:, 5].astype(int)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.65, 0.4)
    indices = np.array(indices)
    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        # Extract object ROI and estimate depth using Lucas-Kanade optical flow
        roi = frame[int(y):int(y + h), int(x):int(x + w)]
        if not roi.any():
            continue
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, maxCorners=100, qualityLevel=0.9, minDistance=1, blockSize=1)
        if p0 is not None and len(p0) > 0:
            # Resize reference frame to match ROI size
            p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.resize(frame_ref_gray, (roi_gray.shape[1], roi_gray.shape[0])),
                                                   roi_gray, p0, None, **lk_params)
            p0 = p0[st == 1]
            p1 = p1[st == 1]
            good_err = err[st == 1]
        if p0 is None or good_err is None:
            continue
        if len(p1) > 0:
            object_point = p1[np.argmin(good_err)]
            depth = np.sqrt((object_point[0] - p0[np.argmin(good_err)][0]) ** 2 + (
                        object_point[1] - p0[np.argmin(good_err)][1]) ** 2)
            displacement = depth * 3 / w
            speed = displacement * fps * 3.6
            cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)
            cv2.putText(frame, '{:.2f} km/h'.format(speed), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2, cv2.LINE_AA)

        p0 = p1
    # Update reference frame
    frame_ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    # Display the output frame
    cv2.imshow('Object Detection with Depth Estimation', frame)
    t1 = time.time()
    time_elapsed = t1 - t0
    avg_running_time += time_elapsed
    num_frames += 1

    # Check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()

avg_running_time = avg_running_time / num_frames
print("Average running time is ", avg_running_time * 1000, " ms per frame")
