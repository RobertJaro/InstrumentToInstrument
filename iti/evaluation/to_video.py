import glob

import cv2
from tqdm import tqdm

image_folder = '/Users/robert/PycharmProjects/InstrumentToInstrument/result/stereo_v6/series/*.jpg'  # '/Users/robert/PycharmProjects/NewDawnProject/newdawnpy/anomaly/results/conf2/kso_full_series/*'
video_name = '/Users/robert/PycharmProjects/InstrumentToInstrument/result/stereo_v6/series.mp4'  # save as .avi
# is changeable but maintain same h&w over all  frames
width = 3600
height = 1800
# this fourcc best compatible for avi
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 9, (width, height))

for i in tqdm(sorted(glob.glob(image_folder))):
    x = cv2.imread(i)
    # x = cv2.resize(x, (width, height))
    video.write(x)

cv2.destroyAllWindows()
video.release()
