from decord import VideoReader
from decord import cpu
import cv2
import glob

path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Tests\\New folder\\'
videos = glob.glob(path + '*.mp4')
sr = 333
frame_width = 1088
frame_height = 140
for filename in videos:
    out = cv2.VideoWriter(filename[:-4] + '_crop.mp4', cv2.VideoWriter_fourcc(*'XVID'), sr, (frame_width, frame_height), True)
    vidObj = VideoReader(filename, ctx=cpu(0))  # read the video
    frames_total = len(vidObj)
    for frameNr in range(frames_total):
        frame = vidObj[frameNr]
        frame_np = frame.asnumpy()
        out.write(frame_np[280:, :, :])
