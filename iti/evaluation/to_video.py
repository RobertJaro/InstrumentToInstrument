import glob

import cv2
from tqdm import tqdm

image_folder = '/Users/robert/PycharmProjects/InstrumentToInstrument/result/hmi_v12/series_20141122/*.jpg'  # '/Users/robert/PycharmProjects/NewDawnProject/newdawnpy/anomaly/results/conf2/kso_full_series/*'
video_name = '/Users/robert/PycharmProjects/InstrumentToInstrument/result/hmi_v12/series_20141122.mp4'  # save as .avi
# load files
files = sorted(glob.glob(image_folder))
# set frame
x = cv2.imread(files[0])
width = x.shape[1]
height = x.shape[0]
# this fourcc best compatible for avi
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 9, (width, height))

for i in tqdm(files):
    x = cv2.imread(i)
    # x = cv2.resize(x, (width, height))
    video.write(x)

cv2.destroyAllWindows()
video.release()
