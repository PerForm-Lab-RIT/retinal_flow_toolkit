# 2023 Gabriel J. Diaz @ RIT
import json
import os
import sys
import numpy as np
import av
import logging
import pickle
from tqdm import tqdm

import warnings
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import h5py

warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    os.add_dll_directory("D://opencvgpu//opencv_build_310//bin")
    os.add_dll_directory("C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.8//bin")
    import cv2
except:
    import cv2

sys.path.append('core')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

###############################

file_name = "linear_travel.mp4"
#file_name = "dash_cam.mp4"
file_prefix = file_name.split('.')[0]

a_file_path = os.path.join("demo_input_video", file_name)
lower_mag_threshold = 0
upper_mag_threshold = 30
algorithm= 'nvidia2'
visualize_as = 'hsv_overlay'
scale = 1

import matplotlib.pyplot as plt
from matplotlib import use as mpl_use

hist_params = (100, 0, 40)
cumulative_mag_hist = None
magnitude_bins = None

def show_image(image):

    cv2.imshow('temp', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

###############################
def save_hist(cumulative_mag_hist, magnitude_bins, out_path):

    mpl_use('qtagg')

    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.hist(magnitude)
    ax.bar(magnitude_bins[:-1], np.cumsum(cumulative_mag_hist) / sum(cumulative_mag_hist))
    ax.grid(True)
    # ax.set_title('flow magnitude')
    ax.set_xlabel('vector length')
    ax.set_ylabel('likelihood')

    if os.path.isdir(out_path) is False:
        os.makedirs(out_path)

    plt.savefig(os.path.join(out_path, 'mag_hist'))

def append_to_mag_histogram(index, magnitude, cumulative_mag_hist):

    mag_hist = np.histogram(magnitude, hist_params[0], (hist_params[1], hist_params[2]))
    magnitude_bins = mag_hist[1]

    # Store the histogram of avg magnitudes
    if index == 1:
        # Store the first flow histogram
        cumulative_mag_hist = mag_hist[0]

    else:
        # Calc cumulative avg flow magnitude by adding the first flow histogram in a weighted manner
        cumulative_mag_hist = np.divide(
            np.sum([np.multiply((index - 1), cumulative_mag_hist), mag_hist[0]], axis=0), index)

    return cumulative_mag_hist, magnitude_bins

video_out_path = os.path.join("flow_out", file_prefix)
video_out_name = file_prefix + '_' + algorithm + '_' + visualize_as + '_script.mp4'

if os.path.isdir(video_out_path) is False:
    os.makedirs(video_out_path)

image1_gpu = cv2.cuda_GpuMat()
image2_gpu = cv2.cuda_GpuMat()

cap = cv2.VideoCapture(a_file_path)
ret, first_frame = cap.read()

clahe = cv2.cuda.createCLAHE(clipLimit=1.0, tileGridSize=(7, 7))

_, mask = cv2.threshold(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_TOZERO)
first_frame = cv2.bitwise_and(first_frame, first_frame, mask=mask)

image2_gpu.upload(first_frame)
rescaled_size = (int(image2_gpu.size()[0]*scale), int(image2_gpu.size()[1]*scale))

# image2_gpu = cv2.cuda.pyrUp(image2_gpu)
image2_gpu = cv2.cuda.resize(image2_gpu, rescaled_size, interpolation=cv2.INTER_AREA)
image2_gpu = cv2.cuda.cvtColor(image2_gpu, cv2.COLOR_BGR2GRAY)

width = image2_gpu.size()[0]
height = image2_gpu.size()[1]

if algorithm == 'tvl1':
    flow_algo = cv2.cuda_OpticalFlowDual_TVL1.create()
    flow_algo.setNumScales(30)  # (1/5)^N-1 def: 5
    flow_algo.setLambda(0.1)  # default 0.15. smaller = smoother output.

elif algorithm == 'nvidia':

    params = {'perfPreset': cv2.cuda.NvidiaOpticalFlow_1_0_NV_OF_PERF_LEVEL_SLOW}
    flow_algo = cv2.cuda.NvidiaOpticalFlow_1_0.create((width, width), **params)

elif algorithm == 'nvidia2':

    params = {'perfPreset': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_SLOW,
              'outputGridSize': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_OUTPUT_VECTOR_GRID_SIZE_1,
              'hintGridSize': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_HINT_VECTOR_GRID_SIZE_1,
              'enableTemporalHints': True,
              'enableCostBuffer': True}

    flow_algo = cv2.cuda.NvidiaOpticalFlow_2_0_create((width, height), **params)

vid_out_full_path = os.path.join(video_out_path, video_out_name)

video_out = cv2.VideoWriter(vid_out_full_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in tqdm(range(1, num_frames), desc="Generating " + vid_out_full_path, unit='frames', total=num_frames):

    success, frame = cap.read()
    if not success:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_TOZERO)
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # thresh1, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_TOZERO)

    image1_gpu.upload(frame)

    if rescaled_size != 1:
        image1_gpu = cv2.cuda.resize(image1_gpu, rescaled_size, interpolation=cv2.INTER_AREA)

    image1_gpu = cv2.cuda.cvtColor(image1_gpu, cv2.COLOR_BGR2GRAY)
    # image1_gpu = clahe.apply(image1_gpu, cv2.cuda_Stream.Null())
    flow = flow_algo.calc(image1_gpu, image2_gpu, None)

    if type(flow_algo) == cv2.cuda.NvidiaOpticalFlow_1_0:
        flow = flow_algo.upSampler(flow[0], (frame.shape[1], frame.shape[0]), flow_algo.getGridSize(), None)

    elif type(flow_algo) == cv2.cuda.NvidiaOpticalFlow_2_0:
        flow = flow_algo.convertToFloat(flow[0], None)

    flow = flow.download()

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    cumulative_mag_hist, magnitude_bins = append_to_mag_histogram(i, magnitude, cumulative_mag_hist) # 29.6

    image1_gray = image1_gpu.download()

    _, mask = cv2.threshold(image1_gray, 50, 255, cv2.THRESH_BINARY)
    magnitude = cv2.bitwise_and(magnitude, magnitude, mask=mask)

    # magnitude = self.apply_magnitude_thresholds_and_rescale(magnitude, lower_mag_threshold, upper_mag_threshold)
    # magnitude = np.clip(magnitude, lower_mag_threshold, upper_mag_threshold)
    if lower_mag_threshold: # 7.3
        magnitude[magnitude < lower_mag_threshold] = lower_mag_threshold

    if upper_mag_threshold:
        magnitude[magnitude > upper_mag_threshold] = upper_mag_threshold

    magnitude = ((magnitude-lower_mag_threshold) / (upper_mag_threshold-lower_mag_threshold)) * 255.0

    # 56.205658
    # 43.350235
    # 57.2053

    # magnitude = cv2.normalize(magnitude, None, 0, np.nanmax(magnitude), cv2.NORM_MINMAX, -1)
    hsv = np.zeros([np.shape(magnitude)[0], np.shape(magnitude)[1], 3], np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2BGR)
    show_image(bgr)
    video_out.write(bgr)
    image2_gpu = image1_gpu.clone()

save_hist(cumulative_mag_hist, magnitude_bins, video_out_path)

cv2.destroyAllWindows()
video_out.release()
cap.release()
video_out.release()


print('Done!')
sys.exit(1)
