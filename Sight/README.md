# Sight Instructions.

This component consists of 3 computer vision programs that attempt to count the frequency of swallowing.

- Eularian  Video Magnification
- Gabor Heat Map
- Optical Flow

# Eularian Video Magnification

This solution allows the user to magnify subtle movements in video. It can be applied to any .mp4 video but in this case the region of interest is to target the throat. Once a user has recorded a video follow these steps.

1. Open evm_roi.py
2. Write input path to video (File location must be in project. Starts from project root)
3. Write output path to video
4. Run python script 'python evm_roi.py'
5. Select region of interest on first frame with cursor
6. View output path for new video overlay

# Gabor Heat Map

This solution allows the user to highlight textures with a heatmap overaly. It can be applied to any .mp4 video but in this case the region of interest is to target the throat. Once a user has recorded a video follow these steps.

1. Open of3.py
2. Write input path to video (File location must be in project. Starts from project root)
3. Write output path to video
4. Run python script 'python gabor.py'
5. Select region of interest on first frame with cursor
6. View output path for new video overlay

# Optical Flow

This solution allows the user to measure the changes of pixels over frames. It can be applied to any .mp4 video but in this case the region of interest is to target the throat. Once a user has recorded a video follow these steps.

1. Open of3.py
2. Write input path to video (File location must be in project. Starts from project root)
3. Write output path to video
4. Run python script 'python of3.py'
5. Select region of interest on first frame with cursor
6. Review instant playback
7. View output path for new video overlay

# Yolo
This solution allows the user to track objects over frames. It can be applied to any .mp4 video. Once a user has recorded a video follow these steps.

## Annotate Data
Refer to SIGHT - Installation Guide.docx
1. Install anaconda
2. Activate anaconda environment
3. Install labelimg
4. Run labelimg pointing to dataset
5. Annotate dataset using yolo

## Train Model
Refer to SIGHT - Yolo11 Train.docx
1. Activate anaconda environment
2. Install yolo11
3. Execute yolo train pointing to annotated dataset

## Predict Model
Refer to SIGHT - Yolo11 Predict.docx
1. Activate anaconda environment
2. Execute yolo predict pointing to best.pt (trained model built from yolo train)

