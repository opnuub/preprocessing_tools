from glob import glob
import cv2
import os

def with_opencv(filename):

    video = cv2.VideoCapture(filename)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return frame_count

ROOT_DIR = os.getcwd()

files = glob(ROOT_DIR + "\\*.avi")
files = [i.replace("\\", "/").split("/")[-1].split(".avi")[0] for i in files]

for file in files:
    video_path = file + '.avi'
    frames = with_opencv(video_path) # /10
    for frame in range(100, int(frames), 100):
        input_video = cv2.VideoCapture(video_path)
        w, h = int(input_video.get(3)), int(input_video.get(4))
        size = (w, h)
        start, end, id = frame-100, frame, int(frame/10)
        output_video = cv2.VideoWriter(f'{id}_{file}.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, size)
        while start <= end:
            input_video.set(cv2.CAP_PROP_POS_FRAMES, start)
            ret, frame = input_video.read()
            if ret == True:
                output_video.write(frame)
            start += 1
        output_video.release()
    input_video.release()
    try:
        os.remove(f"./{file}.avi")
        print(f'{file}.avi is removed.')
    except Exception:
        print(f'{file}.avi was unable to be removed.')