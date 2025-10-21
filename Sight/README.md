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
