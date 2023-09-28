# 2023 Gabriel J. Diaz @ RIT
import json
import os
import sys
import numpy as np
import logging
import pickle
from tqdm import tqdm

import numcodecs

import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# import h5py
import zarr

from tkinter import Tk
from tkinter.filedialog import askopenfilename

warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    os.add_dll_directory("D://opencvgpu//opencv_build_310//bin")
    os.add_dll_directory("C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.8//bin")
    import cv2
except:
    import cv2

sys.path.append('core')

logger = logging.getLogger(__name__)

# These lines allow me to see logging.info messages in my jupyter cell output
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

def show_image(image):

    if type(image) == cv2.cuda_GpuMat:
        image = image.download()

    cv2.imshow('temp', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class video_source():

    def __init__(self, file_path, out_parent_dir="flow_out"):

        self.file_path = file_path
        self.source_file_name = os.path.split(file_path)[-1].split('.')[0]
        self.source_file_suffix = os.path.split(file_path)[-1].split('.')[1]

        self.out_parent_dir = out_parent_dir
        self.raw_frames_out_path = os.path.join(out_parent_dir, self.source_file_name, 'raw')
        self.flow_frames_out_path = os.path.join(out_parent_dir, self.source_file_name, 'flow')
        self.video_out_path = os.path.abspath(os.path.join(out_parent_dir, self.source_file_name))

        self.magnitude_out_path = os.path.join(out_parent_dir, self.source_file_name, 'magnitude_data')

        self.video_target_path = False
        self.cumulative_mag_hist = None
        self.magnitude_bins = None
        self.hist_params = (100, 0, 40)

        self.flow_algo = False
        self.clahe = False

        self.cuda_enabled = True

        self.save_midpoint_images = False

        self.streamline_points = None

        self.current_gpu = False
        self.previous_gpu = False

        self.current_gray = False
        self.previous_gray = False

    def init_flow(self, algorithm, height_width=False):

        # Even if self.cuda_is_enabled=True, some algos don't support cuda!
        algo_supports_cuda = False

        if algorithm == False:
            
            return False

        elif algorithm == "deepflow":

            self.flow_algo = cv2.optflow.createOptFlow_DeepFlow()

        elif algorithm == "farneback":

            if not self.cuda_enabled:
                self.flow_algo = cv2.optflow.createOptFlow_Farneback()
                # self.flow_algo.setFastPyramids(False)
                # self.flow_algo.setFlags(0)
                self.flow_algo.setNumIters(10) #10
                self.flow_algo.setNumLevels(25)
                self.flow_algo.setPolyN(5) # 5
                self.flow_algo.setPolySigma(1.1)
                self.flow_algo.setPyrScale(10.5) # 0.5
                self.flow_algo.setWinSize(3) # 13

            else:
                algo_supports_cuda = True
                self.flow_algo = cv2.cuda_FarnebackOpticalFlow.create()

        elif algorithm == "tvl1":

            if not self.cuda_enabled:
                logger.error('Non-cuda optical flow calculation not yet supported for ' + algorithm,
                             stack_info=True, exc_info=True)
                sys.exit(1)

            algo_supports_cuda = True
            self.flow_algo = cv2.cuda_OpticalFlowDual_TVL1.create()
            self.flow_algo.setNumScales(30)  # (1/5)^N-1 def: 5

            # self.flow_algo.setScaleStep(0.7)  #
            # self.flow_algo.setLambda(0.5)  # default 0.15. smaller = smoother output.
            # self.flow_algo.setScaleStep(0.7)  # 0.8 by default. Not well documented.  0.7 did better with dots?
            # self.flow_algo.setEpsilon(0.005)  # def: 0.01
            # self.flow_algo.setTau(0.5)
            # self.flow_algo.setGamma(0.5) # def 0

        elif algorithm == "brox":

            if not self.cuda_enabled:
                logger.error('Non-cuda optical flow calculation not yet supported for ' + algorithm,
                             stack_info=True, exc_info=True)
                sys.exit(1)

            algo_supports_cuda = True
            self.flow_algo = cv2.cuda_BroxOpticalFlow.create()
            # self.flow_algo.setFlowSmoothness() # def alpha 0.197
            # self.flow_algo.setGradientConstancyImportance() # def gamma 0
            # self.flow_algo.setInnerIterations() # def 5
            # self.flow_algo.setOuterIterations() # def 150
            self.flow_algo.setPyramidScaleFactor(5) # def 0
            self.flow_algo.setSolverIterations(2) # def 0

        elif algorithm == "pyrLK":

            if not self.cuda_enabled:
                logger.error('Non-cuda optical flow calculation not yet supported for ' + algorithm, stack_info=True,
                             exc_info=True)
                sys.exit(1)

            algo_supports_cuda = True
            self.flow_algo = cv2.cuda_DensePyrLKOpticalFlow.create()
            self.flow_algo.setMaxLevel(6)  # default 3
            self.flow_algo.setWinSize((41, 41))  # default 13, 13
            #self.flow_algo.setNumIters(40) # 30 def

        elif algorithm == "nvidia1" or algorithm == "nvidia":
            algo_supports_cuda = True

            params = {'perfPreset': cv2.cuda.NvidiaOpticalFlow_1_0_NV_OF_PERF_LEVEL_SLOW}
            self.flow_algo = cv2.cuda.NvidiaOpticalFlow_1_0.create((height_width[1], height_width[0]), **params)

        elif algorithm == "nvidia2":
            algo_supports_cuda = True

            params = {'perfPreset': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_SLOW,
                      'outputGridSize': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_OUTPUT_VECTOR_GRID_SIZE_1,
                      'hintGridSize': cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_HINT_VECTOR_GRID_SIZE_1,
                      'enableTemporalHints': True,
                      'enableCostBuffer': True}

            self.flow_algo = cv2.cuda.NvidiaOpticalFlow_2_0_create((height_width[1], height_width[0]), **params)

        else:
            logger.error('Optical flow algorithm not implemented, or a typo', stack_info=True,exc_info=True)
            sys.exit(1)

        return algo_supports_cuda


    def set_stream_dimensions(self, stream, visualize_as, height, width):

        stream.width = width

        if visualize_as == "hsv_stacked":
            stream.height = height * 2
        else:
            stream.height = height

        return stream

    def get_hdf5_filepath(self, algorithm: str, gaze_centered=False):

        if gaze_centered:
            hdf5_out_name = self.source_file_name + '_' + algorithm + '_gaze-centered_flow.zarr'
        else:
            hdf5_out_name = self.source_file_name + '_' + algorithm + '_flow.zarr'

        return os.path.join(self.video_out_path,hdf5_out_name)

    def open_hdf5_file(self, algorithm: str, gaze_centered):

        if not os.path.exists(self.get_hdf5_filepath(algorithm, gaze_centered)):
            logger.error('HDF5 flow file not found.  Try calculating optic flow with that algorithm first.', stack_info=True, exc_info=True)

            sys.exit(1)

        hdf5_flow = zarr.convenience.open(self.get_hdf5_filepath(algorithm, gaze_centered), 'r')

        return hdf5_flow

    def calculate_flow(self,
                       video_out_name=False,
                       algorithm="nvidia2",
                       gaze_centered=False,
                       preprocess_frames = True,
                       save_input_images=False,
                       save_output_images=False):


        video_in = cv2.VideoCapture(self.file_path)

        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_in.get(cv2.CAP_PROP_FPS)
        num_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        algo_supports_cuda = self.init_flow(algorithm, (height, width))

        if algo_supports_cuda and self.cuda_enabled:
            self.current_gpu = cv2.cuda_GpuMat()
            self.previous_gpu = cv2.cuda_GpuMat()
            # self.clahe = cv2.cuda.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))

        count = 0
        success = 1

        hdf5_filepath = self.get_hdf5_filepath(algorithm,gaze_centered=gaze_centered)

        # hdf5_file_out = zarr.open(hdf5_filepath, mode='w')

        # hdf5_file_out = zarr.open(hdf5_filepath, mode='w',
        #               shape = (num_frames, height, width, 2),
        #               chunks = (20, height, width, 2),
        #               dtype = np.float32, compressor = numcodecs.Blosc(cname='zstd', clevel=3))

        # root = hdf5_file_out.group()

        zarr_file_out = zarr.group(store = zarr.DirectoryStore(hdf5_filepath), overwrite=True)
        flow_dataset = zarr_file_out.create_dataset('flow',
                                             shape=(num_frames, height, width, 2),
                                             chunks=(1, height, width, 1),
                                             dtype=np.float32,
                                             compressor=numcodecs.Blosc(cname='zstd', clevel=3))

        # cname options: ['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']

        flow_dataset.attrs['approx_fps'] = fps
        flow_dataset.attrs['num_source_frames'] = num_frames
        h = int(video_in.get(cv2.CAP_PROP_FOURCC))
        flow_dataset.attrs['source_fourcc'] = h
        flow_dataset.attrs['source_codec'] = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)

        # flow_hdf5_dataset.attrs['approx_fps'] = fps
        # flow_hdf5_dataset.attrs['num_source_frames'] = num_frames
        # h = int(video_in.get(cv2.CAP_PROP_FOURCC))
        # flow_hdf5_dataset.attrs['source_fourcc'] = h
        # flow_hdf5_dataset.attrs['source_codec'] = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)

        # ##############################
        # # Iterate through frames
        for index in tqdm(range(num_frames), desc="Saving flow to  " + hdf5_filepath, unit='frames', total=num_frames):

            success, frame = video_in.read()

            if not success:
                print(f'Frame {index}: video_in.read() unsuccessful')
                continue

            if index == 0:

                prev_frame = frame

                if preprocess_frames:

                    prev_frame = self.preprocess_frame(prev_frame)

                # Apply histogram normalization.
                if algo_supports_cuda and self.cuda_enabled:
                    self.previous_gpu.upload(prev_frame)
                    self.previous_gpu = cv2.cuda.cvtColor(self.previous_gpu, cv2.COLOR_BGR2GRAY)
                    # self.previous_gpu = self.clahe.apply(previous_gpu, cv2.cuda_Stream.Null())
                else:
                    self.previous_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    # image2_gray = self.clahe.apply(image2_gray)

                # Write out a blank first frame
                image_out = np.zeros([np.shape(frame)[0], np.shape(frame)[1], 3], dtype=np.uint8)
                # video_out.write(image_out)
                continue
            else:

                # If is gaze data, recenter
                if gaze_centered:
                    self.recenter_frame_on_gaze(frame, index)

                current_bgr_processed = self.preprocess_frame(frame)

            if algo_supports_cuda and self.cuda_enabled:
                self.current_gpu.upload(current_bgr_processed)
                self.current_gpu = cv2.cuda.cvtColor(self.current_gpu, cv2.COLOR_BGR2GRAY)
            else:
                self.current_gray = cv2.cvtColor(current_bgr_processed, cv2.COLOR_BGR2GRAY)

            flow = self.calculate_flow_for_frame(index,algo_supports_cuda)

            # Save optical flow to HDF5
            # flow_hdf5_dataset.resize((flow_hdf5_dataset.shape[0] + 1, flow.shape[0], flow.shape[1], 2))
            flow_dataset[index,...] = flow

            if self.cuda_enabled:
                self.previous_gpu = self.current_gpu.clone()

            self.previous_gray = self.current_gray

            # Save input images?
            if save_input_images:
                if os.path.isdir(self.raw_frames_out_path) is False:
                    os.makedirs(self.raw_frames_out_path)
                cv2.imwrite(str(os.path.join(self.raw_frames_out_path, 'frame-{}.png'.format(index))),
                            current_bgr_processed)

            # Save output images?
            if save_output_images:
                if os.path.isdir(self.flow_frames_out_path) is False:
                    os.makedirs(self.flow_frames_out_path)
                cv2.imwrite(str(os.path.join(self.flow_frames_out_path, 'frame-{}.png'.format(index))),
                            image_out)

        video_in.release()
        hdf5_file_out.flush()

    def calculate_flow_for_frame(self, index, algo_supports_cuda):

        if self.cuda_enabled and algo_supports_cuda:

            if type(self.flow_algo) == cv2.cuda.BroxOpticalFlow:
                self.current_gpu.upload(self.current_gpu.download().astype('float32'))
                self.previous_gpu.upload(self.previous_gpu.download().astype('float32'))
                flow = self.flow_algo.calc(self.current_gpu, self.previous_gpu, None)

            else:
                flow = self.flow_algo.calc(self.current_gpu, self.previous_gpu, None)

            if type(self.flow_algo) == cv2.cuda.NvidiaOpticalFlow_1_0:
                flow = self.flow_algo.upSampler(flow[0],(self.current_gpu.size()[0],self.current_gpu.size()[1]), self.flow_algo.getGridSize(),None)

            if type(self.flow_algo) == cv2.cuda.NvidiaOpticalFlow_2_0:
                flow = self.flow_algo.convertToFloat(flow[0], None)

            flow = flow.download()

        else:
            flow = self.flow_algo.calc(self.current_gray, self.previous_gray, None)
            flow = flow[0]

        return flow


    # def flow_to_mag_angle(self, index, flow, lower_mag_threshold=False, upper_mag_threshold=False):
    #
    #     # Convert flow to mag / angle
    #         magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    #         magnitude = self.filter_magnitude(magnitude)  # 29.6,
    #         self.append_to_mag_histogram(index, magnitude)  # 7.3
    #
    #         magnitude = self.filter_magnitude(magnitude)  # 29.6,
    #         magnitude = self.apply_magnitude_thresholds_and_rescale(magnitude,
    #                                                                 lower_mag_threshold=lower_mag_threshold,
    #                                                                 upper_mag_threshold=upper_mag_threshold)
    #
    #         return magnitude, angle

    def save_out_mag_histogram(self, algorithm):

        #  Save out magnitude data pickle and image
        if os.path.isdir(self.magnitude_out_path) is False:
            os.makedirs(self.magnitude_out_path)

        mag_pickle_filename = self.source_file_name + '_' + algorithm + '_mag.pickle'
        dbfile = open(os.path.join(self.magnitude_out_path, mag_pickle_filename), 'wb')
        pickle.dump({"values": self.cumulative_mag_hist, "bins": self.magnitude_bins[1]}, dbfile)
        dbfile.close()

        mag_image_filename = self.source_file_name + '_' + algorithm + '_mag.jpg'
        mag_image_fileloc = os.path.join(self.magnitude_out_path, mag_image_filename)
        # self.generate_mag_histogram(mag_image_fileloc, cumulative_mag_hist, mag_hist[1])

        import matplotlib.pyplot as plt
        # This prevents a warning that was present on some versions of mpl
        from matplotlib import use as mpl_use

        mpl_use('qtagg')

        fig, ax = plt.subplots(figsize=(8, 4))
        # ax.bar(bins[:-1], mag_values, width = .75 * (bins[1]-bins[0]))

        ax.bar(self.magnitude_bins[:-1], np.cumsum(self.cumulative_mag_hist) / sum(self.cumulative_mag_hist))
               # width=.75 * (self.magnitude_bins[-1] - self.magnitude_bins[0]))

        # tidy up the figure
        ax.grid(True)

        # ax.set_title('flow magnitude')
        ax.set_xlabel('vector length')
        ax.set_ylabel('likelihood')

        print( f'Magnitude saved to {self.magnitude_out_path}')

        plt.savefig(mag_image_fileloc)

    def calculate_magnitude_distribution(self,algorithm, gaze_centered=False, stride_length = 20):
        # Calculates magnitude distribution of flow vectors for every stride_length frame in central 2/3 of video

        if isinstance(self, pupil_labs_source) == False & gaze_centered == True:
            logger.error('Can\'t perform gaze centered operations for video_sources.  Ignoring request!',stack_info=True, exc_info=True)
            sys.exit(1)

        def append_to_mag_histogram(index, magnitude):

            mag_hist = np.histogram(magnitude, self.hist_params[0], (self.hist_params[1], self.hist_params[2]))

            # Store the histogram of avg magnitudes
            if index == 0:
                # Store the first flow histogram
                self.cumulative_mag_hist = mag_hist[0]
                self.magnitude_bins = mag_hist[1]
            else:
                # Calc cumulative avg flow magnitude by adding the first flow histogram in a weighted manner
                cumulative_mag_hist = np.divide(
                    np.sum([np.multiply((index - 1), self.cumulative_mag_hist), mag_hist[0]], axis=0), index)

        print('Calculating magnitude distribution.')

        hdf5_object = self.open_hdf5_file(algorithm,gaze_centered)
        flow = hdf5_object['flow']

        num_rows = np.shape(flow)[0]
        one_third = int(num_rows/3)

        row_indices = np.arange(one_third,num_rows-one_third, stride_length)
        count = 0

        for frame_idx in tqdm(row_indices, desc="Calculating magnitude distribution.", unit='frames', total=len(row_indices)):
        # for count, frame_idx in enumerate(np.arange(one_third,num_rows-one_third, stride_length)):

            magnitude, angle = cv2.cartToPolar(flow[frame_idx,:,:,0], flow[frame_idx,:,:,1]);
            append_to_mag_histogram(count, magnitude)
            count = count+1

            plt.close('all')

        self.save_out_mag_histogram(algorithm)

    def apply_magnitude_thresholds_and_rescale(self, magnitude, lower_mag_threshold=False, upper_mag_threshold=False):

        '''
        This method first clips the flow magnitude to the range lower_mag_threshold - upper_mag_threshold.
        It then rescales from the range min-max into 0-255.
        '''

        if lower_mag_threshold:
            magnitude[magnitude < lower_mag_threshold] = 0

        if upper_mag_threshold:
            magnitude[magnitude > upper_mag_threshold] = upper_mag_threshold

        magnitude = (magnitude / upper_mag_threshold) * 255.0

        return magnitude

    def convert_flow_to_magnitude_angle(self,flow,
                                        bgr_world_in,
                                        lower_mag_threshold = False,
                                        upper_mag_threshold = False):

        magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])

        # Clips to lower-upper thresh and then rescales into range 0-255
        magnitude = self.apply_magnitude_thresholds_and_rescale(magnitude,
                                                                  lower_mag_threshold=lower_mag_threshold,
                                                                  upper_mag_threshold=upper_mag_threshold)

        # A custom filter.  Mine masks out black patches in the input video.
        # Due to compression, they can be filled with noise.
        processed_bgr_world = self.preprocess_frame(bgr_world_in)

        processed_gray_world = cv2.cvtColor(processed_bgr_world, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(processed_gray_world, 10, 255, cv2.THRESH_TOZERO)

        magnitude = cv2.bitwise_and(magnitude, magnitude, mask=mask)
        angle = cv2.bitwise_and(angle, angle, mask=mask)

        return magnitude, angle


    @staticmethod
    def preprocess_frame(frame):

        def create_sky_mask_on_hsv(hsv_in):
            lower = np.array([100, 45, 100])
            upper = np.array([120, 140, 260])

            mask = cv2.inRange(hsv_in, lower, upper)
            mask = 255 - mask

            return mask

        def create_road_mask_on_hsv(hsv_in):
            lower = np.array([0, 240, 200])
            upper = np.array([40, 255, 255])
            mask1 = cv2.inRange(hsv, lower, upper)

            lower = np.array([170, 200, 100])
            upper = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower, upper)

            mask = mask1 + mask2
            mask = np.clip(mask, 0, 255).astype(np.uint8)

            mask = cv2.dilate(mask, kernel=np.ones((5, 5), np.uint8), iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
            mask = 255 - mask

            return mask

        def remove_noise_bgr_dark_patches(bgr_image):
            _, mask = cv2.threshold(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY), 30, 250, cv2.THRESH_TOZERO)
            bgr_image = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
            return bgr_image

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        frame = cv2.bitwise_and(frame, frame, mask=create_sky_mask_on_hsv(hsv))
        frame = cv2.bitwise_and(frame, frame, mask=create_road_mask_on_hsv(hsv))

        frame = remove_noise_bgr_dark_patches(frame)
        return frame

    def create_video_objects(self,algorithm, visualize_as, gaze_centered, video_out_filename):

        video_in = cv2.VideoCapture(self.file_path)

        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))

        # If a stacked video viz (e.g., hsv_stacked), double the height
        if 'stacked' in visualize_as:
            height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 2
        else:
            height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = video_in.get(cv2.CAP_PROP_FPS)
        video_out = cv2.VideoWriter(os.path.join(self.video_out_path, video_out_filename),cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

        return video_in, video_out

    def create_visualization(self,
                             algorithm: str,
                             gaze_centered: False,
                             visualize_as: str,
                             lower_mag_threshold = False,
                             upper_mag_threshold = False,
                             ):

        if isinstance(self, pupil_labs_source) == False and gaze_centered == True:
            logger.error('You can only perform gaze centered operations when using pupil_labs_sources.', stack_info=True, exc_info=True)
            sys.exit(1)

        # Create video out filename
        video_out_filename = []

        lower_thresh_string = 'None' if not lower_mag_threshold else str(lower_mag_threshold)
        upper_thresh_string = 'None' if not upper_mag_threshold else str(upper_mag_threshold)
        thresh_string = f'{lower_thresh_string}-{upper_thresh_string}'

        if gaze_centered:
            path, input_video_filename = os.path.split(self.file_path)
            input_video_filename = input_video_filename.split('.')[0]
            video_out_filename = f'{input_video_filename}_gaze_centered_{algorithm}_{thresh_string}.mp4'
        else:
            video_out_filename = f'{self.source_file_name}_{algorithm}_{thresh_string}.mp4'

        # create video containers and streams
        video_in, video_out = self.create_video_objects(algorithm, visualize_as, gaze_centered, video_out_filename)

        hdf5_object = self.open_hdf5_file(algorithm, gaze_centered)
        flow_ds = hdf5_object['flow']
        attrs = flow_ds.attrs

        width = np.shape(flow_ds[2])
        height = np.shape(flow_ds[1])
        fps = attrs['approx_fps']
        num_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

        assert np.shape(flow_ds)[0] == attrs['num_source_frames'],"The number of frames from the source does not equal the number of frames in the hdf5 flow file."

        buffer_len = 10
        from collections import deque
        flow_buffer = deque([], maxlen=buffer_len)

        for index in tqdm(range(num_frames), desc="Generating " + video_out_filename, unit='frames', total=num_frames):

            success, bgr_world = video_in.read()

            if not success:
                print(f'Frame {index}: video_in.read() unsuccessful')
                continue

            if gaze_centered:
                bgr_world = self.recenter_frame_on_gaze(bgr_world, index)

            flow_frame = np.array(flow_ds[index,...])
            flow_buffer.appendleft(flow_frame)

            if visualize_as == "vector_buffer":
                flow_frame = np.sum(flow_buffer, axis=0)
                flow_frame = cv2.blur(flow_frame, [7, 7])

            magnitude, angle = self.convert_flow_to_magnitude_angle(flow_frame,bgr_world,
                                                                    lower_mag_threshold=lower_mag_threshold,
                                                                    upper_mag_threshold=upper_mag_threshold)

            image_out = self.visualize_frame(index,
                                             bgr_world,
                                             flow_frame,
                                             magnitude,
                                             angle,
                                             visualize_as)

            # Add packet to video
            video_out.write(image_out)

        video_out.release()
        video_in.release()


    def visualize_frame(self,
                        index,
                        current_bgr,
                        flow,
                        magnitude,
                        angle,
                        visualize_as):

        if visualize_as == "streamlines":

            image_out = self.visualize_flow_as_streamlines(current_bgr, flow)

        elif visualize_as == "vectors" or visualize_as == "vector_buffer":

            image_out = self.visualize_flow_as_vectors(current_bgr, magnitude, angle)

        elif visualize_as == "hsv_overlay" or visualize_as == "hsv_stacked":

            image_out = self.visualize_flow_as_hsv(magnitude, angle)

            if visualize_as == "hsv_stacked":
                image_out = np.concatenate((current_bgr, image_out), axis=0)

            # Save midpoint images?
            if self.save_midpoint_images:
                if os.path.isdir(self.mid_frames_out_path) is False:
                    os.makedirs(self.mid_frames_out_path)
                cv2.imwrite(str(os.path.join(self.mid_frames_out_path, '{:06d}.png'.format(index))),
                            image_out, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        else:
            logger.error('Visualization method not implemented, or a typo', stack_info=True, exc_info=True)
            sys.exit(1)

        return image_out

    def visualize_flow_as_streamlines(self, frame, flow):

        x = np.arange(0, np.shape(frame)[0], 1)
        y = np.arange(0, np.shape(frame)[1], 1)
        grid_x, grid_y = np.meshgrid(y, x)

        start_grid_res = 10
        start_pts = np.array(
            [[y, x] for x in np.arange(0, np.shape(frame)[0], start_grid_res)
             for y in np.arange(0, np.shape(frame)[1], start_grid_res)])

        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)

        plt.streamplot(grid_x, grid_y, -flow[...,0], -flow[...,1],
                       start_points=start_pts,
                       color='w',
                       maxlength=.4,
                       arrowsize=0,
                       linewidth=.8)  # density = 1

        plt.axis('off')
        plt.imshow(frame)

        canvas = FigureCanvas(fig)
        canvas.draw()  # draw the canvas, cache the renderer

        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
        plt.close('all')
        return image

    def visualize_flow_as_hsv(self, magnitude, angle):
        '''
        Note that to perform well, this function really needs an upper_bound, which also acts as a normalizing term.
        
        '''

        # create hsv output for optical flow
        hsv = np.zeros([np.shape(magnitude)[0], np.shape(magnitude)[1], 3], np.uint8)

        hsv[..., 0] = angle * 180 / np.pi / 2 # angle_rads -> degs 0-360 -> degs 0-180
        hsv[..., 1] = 255
        hsv[..., 2] = magnitude
        # cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        hsv_8u = np.uint8(hsv)
        bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

        return bgr

    def visualize_flow_as_vectors(self, image_in, magnitude, angle, skippts=15, scale=.1,scale_units='width',
                            width=.003, return_image=True):

        dpi = 100
        fig, ax = plt.subplots()

        canvas = FigureCanvas(fig)
        ax.margins(0)

        s = slice(None, None, skippts)

        xmax = np.shape(image_in)[1]
        xpoints = int(np.shape(image_in)[1])
        x = np.linspace(0, np.shape(image_in)[1], xmax)

        ymax = np.shape(image_in)[0]
        ypoints = int(np.shape(image_in)[0])
        y = np.linspace(0, np.shape(image_in)[0], ymax)

        x = x[s]
        y = y[s]
        x2d, y2d = np.meshgrid(x, y, indexing='xy')

        u = magnitude * np.cos(angle) * -1
        v = magnitude * np.sin(angle)

        u = (u[s, s] / 255) * scale
        v = (v[s, s] / 255) * scale

        plt.imshow(cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB))
        ax.axis('off')

        plt.quiver(x2d, y2d, u, v, color='white', alpha=0.7, width=width, scale=scale,
                   scale_units='inches')

        #if return_image:
        canvas.draw()  # draw the canvas, cache the renderer
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close('all')

        return cv2.cvtColor(image_from_plot,cv2.COLOR_BGR2RGB)


    def set_video_target(self, initial_dir=False, file_path=False):

        if not file_path:
            if initial_dir:
                Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
                file_path = askopenfilename(title="Select the video target",initialdir=initial_dir)
            else:
                title="Select the video target"
                Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
                file_path = askopenfilename(title="Select the video target", initialdir=self.video_out_path)

        self.video_target_path = file_path
        return True

    def avg_flow_magnitude_by_direction(self,start_frame=False,play_video=False,save_video=False):

        from matplotlib import cm
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        if self.video_target_path == False:
            self.set_video_target()
        video_in = cv2.VideoCapture(self.video_target_path)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_in.get(cv2.CAP_PROP_FPS)

        if save_video:

            path, input_video_filename = os.path.split(self.video_target_path)
            input_video_filename = input_video_filename.split('.')[0]
            video_out_name = f'{input_video_filename}_avg-mag-by-direction.mp4'
            video_out = cv2.VideoWriter(os.path.join(self.export_folder, video_out_name),
                                        cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, 2*height))



        # from pathlib import Path
        # p = Path(
        #     #'D:/Github/retinal_flow_toolkit/pupil_labs_data/cb13/S001/PupilData/000/exports/000/cb13_world_nvidia2_hsv_overlay.mp4'
        #     'D:\Github\retinal_flow_toolkit\demo_input_video\dash_cam.mp4'
        #     ).as_posix()
        # p = Path('/Users/gjdpci/Documents/Data/Steering/cb13/S001/PupilData/000/exports/000/cb13_world_nvidia2_hsv_overlay.mp4 ')
        # self.video_target_path = p.as_posix()
        # video = cv2.VideoCapture('cb13_world_nvidia2_hsv_overlay.mp4')

        count = 0
        success = 1

        hist_params = (4 *2, 0, 360)
        bins = np.linspace(hist_params[1], hist_params[2], hist_params[0] + 1)
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

        cvals = np.array(bin_centers) / 359 + 0.5
        cvals[cvals > 1] = cvals[cvals > 1] - 1
        hsv_map = cm.get_cmap('hsv').reversed()
        bar_colors = hsv_map(cvals)

        from collections import deque
        bin_rad_hist = deque(maxlen=30)

        # start_frame = 16200  # right turn, high flow
        #start_frame = 17800  # lef turn, high flow
        # start_frame = 15725  # right turn, med flo
        # start_frame = 30000
        num_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

        if start_frame:
            video_in.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            start_frame= 0

        for index in tqdm(range(num_frames-start_frame), desc="Calculating avg vector magnitude", unit='frames', total=num_frames-start_frame):

            success, image = video_in.read()

            if success:
                dpi = 100
                fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), subplot_kw=dict(projection='polar'))
                canvas = FigureCanvas(fig)
                ax.margins(0)
                ax.set_ylim([0, 200])
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                h = hsv_image[..., 0]
                v = hsv_image[..., 2]

                mag_flat = v.flatten()[v.flatten() > 10]
                hue_flat = h.flatten()[v.flatten() > 10]

                hue_flat = hue_flat * 2.0

                bin_rad = []
                for b_idx in range(1, len(bins)):
                    idx = np.where(np.logical_and(hue_flat >= bins[b_idx - 1], hue_flat <= bins[b_idx]))
                    bin_rad.append(np.mean(mag_flat[idx]))

                bin_rad = np.nan_to_num(bin_rad)
                bin_rad_hist.appendleft(bin_rad)

                ax.bar(np.deg2rad(bin_centers), np.mean(bin_rad_hist,axis=0), width=2 * (np.pi / (len(bins))), color=bar_colors);

                canvas.draw()  # draw the canvas, cache the renderer
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                combined_image = np.vstack([image, cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)])
                plt.close('all')


                if save_video:
                    video_out.write(combined_image)

                if play_video:
                    cv2.imshow(f'Image', combined_image)
                    key = cv2.waitKey(25)  # pauses for N ms before fetching next image
                    if key == 27:  # if ESC is pressed, exit loop

                        if save_video:
                            video_out.release()
                            video_in.release()

                        cv2.destroyAllWindows()
                        return

        cv2.destroyAllWindows()
        video_out.release()
        video_in.release()

    def recenter_frame_on_gaze(self, frame, frame_index):
        logger.error('Gaze centering is only possible with pupil labs data sources',
                     stack_info=True, exc_info=True)
        sys.exit(1)

class pupil_labs_source(video_source):

    def __init__(self, pupil_labs_parent_folder,
                 session_number=False,
                 recording_number=False,
                 export_number=False,
                 analysis_parameters_file='default_analysis_parameters.json',
                 recalculate_processed_gaze = False):

        self.analysis_parameters = json.load(open(analysis_parameters_file))

        self.pupil_labs_parent_folder = pupil_labs_parent_folder

        self.session_number = session_number
        self.recording_number = recording_number
        self.export_number = export_number

        self.recording_folder = self.set_recording_folder()
        self.export_folder = self.set_export_folder()

        self.world_video_path = os.path.join(self.recording_folder, 'world.mp4')
        self.gaze_overlay_video_out_name = False

        self.gaze_data = False
        self.processed_gaze_data = False

        # Load processed gaze data from file
        proc_gaze_file_path = os.path.join(self.export_folder, 'processed_gaze.pkl')
        if recalculate_processed_gaze is False and os.path.exists(proc_gaze_file_path):
            with open(proc_gaze_file_path, 'rb') as handle:
                self.processed_gaze_data = pickle.load(handle)

        super().__init__(self.world_video_path, out_parent_dir=self.export_folder)

        self.raw_frames_out_path = os.path.join(self.out_parent_dir, 'world_images')
        self.mid_frames_out_path = os.path.join(self.out_parent_dir, 'flow_mid_images')
        self.flow_frames_out_path = os.path.join(self.out_parent_dir, 'flow_images')
        self.video_out_path = self.out_parent_dir
        self.magnitude_out_path = os.path.join(self.out_parent_dir, 'flow_magnitude_data')

        f = open(os.path.join(self.recording_folder, 'info.player.json'))
        self.player_info = json.load(f)
        f.close()

    def calculate_flow(self,
                       video_out_name=False,
                       algorithm="nvidia2",
                       preprocess_frames=True,
                       gaze_centered=False,
                       save_input_images=False,
                       save_output_images=False):

        if gaze_centered:
            self.import_gaze_from_exports()
            self.process_gaze_data()

        super().calculate_flow(video_out_name=video_out_name,
                               algorithm=algorithm,
                               preprocess_frames=preprocess_frames,
                               gaze_centered=gaze_centered,
                               save_input_images=save_input_images,
                               save_output_images=save_output_images)

    def modify_frame(self, frame, frame_index):

        self.gaze_center_Frame(frame, frame_index)

    def get_median_gaze_for_frame(self, frame_index):

        gaze_samples_on_frame = self.gaze_data[self.gaze_data['world_index'] == frame_index]
        gaze_samples_on_frame = gaze_samples_on_frame[
            gaze_samples_on_frame['confidence'] > self.analysis_parameters['pl_confidence_threshold']]

        if len(gaze_samples_on_frame) == 0:
            # Out of frame
            return (np.NAN, np.NAN)

        median_x = np.nanmedian(gaze_samples_on_frame['norm_pos_x'])
        median_y = 1 - np.nanmedian(gaze_samples_on_frame['norm_pos_y'])

        if median_x < 0 or median_y < 0 or median_x > 1 or median_y > 1:
            # Out of frame
            return (np.NAN, np.NAN)

        return median_x, median_y

    def process_gaze_data(self):

        idx_list = np.unique(self.gaze_data.world_index)

        med_xy = []
        for idx in tqdm(range(len(idx_list)), desc=f"Calculating median gaze locations.", unit='frames',
                          total=len(idx_list)):
            med_xy.append( self.get_median_gaze_for_frame(idx_list[idx]))

            # med_xy = [self.get_median_gaze_for_frame(idx) for idx in idx_list]

        med_x, med_y = zip(*med_xy)

        processed_gaze_data = pd.DataFrame({'median_x': med_x, 'median_y': med_y})

        processed_gaze_data.rolling(3,center=True).apply(lambda x: np.nanmean(x)) # TODO:  Move magic number into json!

        # Save processed gaze to export folder
        proc_gaze_file_path = os.path.join(self.export_folder, 'processed_gaze.pkl')
        with open(proc_gaze_file_path, 'wb') as handle:
            pickle.dump(processed_gaze_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.processed_gaze_data = processed_gaze_data

    def recenter_frame_on_gaze(self, frame, frame_index):

        height = np.shape(frame)[0]
        width = np.shape(frame)[1]

        # if self.processed_gaze_data is False:
        #     self.process_gaze_data()

        if not (frame_index - 1 in self.processed_gaze_data.index):
            return np.zeros((height, width, 3), np.uint8)

        data = self.processed_gaze_data.loc[frame_index - 1]

        if np.isnan(data['median_x']) or np.isnan(data['median_y']):
            return np.zeros((height, width, 3), np.uint8)

        median_x = data['median_x']
        median_y = data['median_y']

        new_image = np.zeros((height*3,width*3,3), np.uint8)

        center_x = width * 1.5
        center_y = height * 1.5
        medianpix_x = int(median_x * width)
        medianpix_y = int(median_y * height)

        x1 = int(center_x - medianpix_x)
        x2 = int(center_x + width - medianpix_x)
        y1 = int(center_y - medianpix_y)
        y2 = int(center_y + height - medianpix_y)

        new_image[ y1:y2, x1:x2,:] = frame

        # remove padding
        new_image = new_image[ height:height*2, width:width*2,:]

        return new_image

    def overlay_gaze_on_frame(self, frame_in, frame_index):

        if not type(self.processed_gaze_data) == pd.DataFrame:
            self.import_gaze_from_exports()
            self.process_gaze_data()

        if frame_index - 1 in self.processed_gaze_data.index:

            med_xy = self.processed_gaze_data.loc[frame_index-1]

            if not np.isnan(med_xy['median_x']) and not np.isnan(med_xy['median_y']):

                median_x = med_xy['median_x']
                median_y = med_xy['median_y']

                height = np.shape(frame_in)[0]
                width = np.shape(frame_in)[1]

                frame = cv2.line(frame_in, (int(width * median_x), 0), (int(width * median_x), height),
                                 (255, 0, 0, 1), thickness=2)

                frame = cv2.line(frame_in, (0, int(height * median_y)), (width, int(height * median_y)),
                                 (255, 0, 0, 1), thickness=2)

        return frame_in

    def import_gaze_from_exports(self):

        gaze_positions_path = os.path.join(self.export_folder, 'gaze_positions.csv')

        if os.path.exists(gaze_positions_path) is False:
            logger.error('No gaze_positions found in the exports folder.', stack_info=True, exc_info=True)
            sys.exit(1)

        # Defaults to the most recent pupil export folder (highest number)
        self.gaze_data = pd.read_csv(gaze_positions_path)
        
        return True

    def set_export_folder(self):

        exports_parent_folder = os.path.join(self.recording_folder, 'exports')

        if os.path.exists(exports_parent_folder) is False:

            os.mkdir(exports_parent_folder)
            export_folder_path = os.path.join(exports_parent_folder, '001')
            os.mkdir(export_folder_path)
            return export_folder_path

        else:
            if self.export_number is False:
                export_folder_list = [x.name for x in os.scandir(exports_parent_folder)]
                self.export_number = export_folder_list[-1]

            export_folder_path = os.path.join(exports_parent_folder, self.export_number)
            return export_folder_path

    def set_recording_folder(self):
        '''
        :param pupil_session_idx:  the index of the pupil session to use with respect to the list of folders in the
        # session directoy.  Typically, these start at 000 and go up from there.

        :return: void
        '''

        def get_highest_value_folder(parent_folder):
            sub_folder_list = []
            [sub_folder_list.append(name) for name in os.listdir(parent_folder) if name[0] != '.']
            if len(sub_folder_list) == 0:
                logger.warning('No sub folders found in ' + parent_folder)
                return False
            return sub_folder_list[-1]

        # Defaults to the last session
        if self.session_number is False:
            self.session_number = get_highest_value_folder(self.pupil_labs_parent_folder)

        session_folder = os.path.join(self.pupil_labs_parent_folder, self.session_number, 'PupilData')

        # Defaults to the last recording
        if self.recording_number is False:
            recording_folder_list = []
            self.recording_number = get_highest_value_folder(session_folder)

        recording_folder = os.path.join(session_folder, self.recording_number)

        return recording_folder

    # def overlay_gaze_on_video(self, input_video=False, file_name_out='gaze_overlay.mp4'):
    #
    #     Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    #     input_video_fullpath = askopenfilename(title="Select the video on which to overlay gaze",
    #                                            initialdir=self.recording_folder)
    #
    #     input_video_path, input_video_filename = os.path.split(input_video_fullpath)
    #     input_video_filename = input_video_filename.split('.')[0]
    #     file_name_out = f'{input_video_filename}_gaze_overlay.mp4'
    #
    #     video_in, video_out = self.create_video_objects(algorithm=False, visualize_as='gaze_overlay')
    #     num_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    #     width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     algo_supports_cuda = self.init_flow(False, (height, width))
    #
    #     for index in tqdm(range(num_frames), desc="Generating " + self.video_out_name, unit='frames', total=num_frames):
    #
    #         success, current_bgr = video_in.read()
    #         overlay_frame = self.overlay_gaze_on_frame(current_bgr, index)
    #         video_out.write(overlay_frame)
    #
    #     self.encode_frame(container_out, stream_out, current_bgr_processed, raw_frame, stream_in, flush=True)
    #
    #     video_out.release()
    #     video_in.release()

if __name__ == "__main__":

    a_file_path = os.path.join("D:\\", "Data", "Driving_1","Aware-AI","CM_reduced")
    source = pupil_labs_source(a_file_path,session_number='S001',recording_number='000')

    source.calculate_flow(algorithm='nvidia2',
                          preprocess_frames = True,
                          gaze_centered = False,
                          save_input_images=False,
                          save_output_images=False)

    source.calculate_magnitude_distribution(algorithm='nvidia2',gaze_centered = False)

    source.create_visualization(algorithm='nvidia2', gaze_centered=False, visualize_as='vectors',
                                lower_mag_threshold=0.25, upper_mag_threshold=30)

    # file_name = "dash_cam.mp4"
    # a_file_path = os.path.join("demo_input_video", file_name)
    # source = video_source(a_file_path)
    # source.cuda_enabled = True
    #
    # source.calculate_flow(algorithm='nvidia2',
    #                       preprocess_frames = True,
    #                       save_input_images=False,
    #                       save_output_images=False)
    #
    # source.calculate_magnitude_distribution(algorithm='nvidia2',gaze_centered = False)
    #
    # source.create_visualization(algorithm='nvidia2', gaze_centered = False, visualize_as='vector_buffer',upper_mag_threshold=20)




