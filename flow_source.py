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


class video_source():

    def __init__(self, file_path, out_parent_dir="flow_out"):

        self.file_path = file_path
        self.source_file_name = os.path.split(file_path)[-1].split('.')[0]
        self.source_file_suffix = os.path.split(file_path)[-1].split('.')[1]

        self.out_parent_dir = out_parent_dir
        self.raw_frames_out_path = os.path.join(out_parent_dir, self.source_file_name, 'raw')
        self.mid_frames_out_path = os.path.join(out_parent_dir, self.source_file_name, 'mid_images')
        self.flow_frames_out_path = os.path.join(out_parent_dir, self.source_file_name, 'flow')
        self.video_out_path = os.path.join(out_parent_dir, self.source_file_name)
        self.magnitude_out_path = os.path.join(out_parent_dir, self.source_file_name, 'magnitude_data')

        self.flow_algo = False
        self.clahe = False

        self.cuda_enabled = True
        self.streamline_points = None

    @staticmethod
    def generate_mag_histogram(mag_image_fileloc, mag_values, bins):

        import matplotlib.pyplot as plt

        # This prevents a warning that was present on some versions of mpl
        from matplotlib import use as mpl_use
        mpl_use('qtagg')

        fig, ax = plt.subplots(figsize=(8, 4))
        # ax.bar(bins[:-1], mag_values, width = .75 * (bins[1]-bins[0]))
        ax.bar(bins[:-1], np.cumsum(mag_values) / sum(mag_values), width=.75 * (bins[1] - bins[0]))

        # tidy up the figure
        ax.grid(True)

        ax.set_title('flow magnitude')
        ax.set_xlabel('value')
        ax.set_ylabel('likelihood')

        plt.savefig(mag_image_fileloc)

    def set_flow_object(self, algorithm, height_width=False):

        # Even if self.cuda_is_enabled=True, some algos don't support cuda!
        algo_supports_cuda = False

        if algorithm == "deepflow":

            self.flow_algo = cv2.optflow.createOptFlow_DeepFlow()

        elif algorithm == "farneback":

            if not self.cuda_enabled:
                self.flow_algo = cv2.optflow.createOptFlow_Farneback()
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

        elif algorithm == "pyrLK":

            if not self.cuda_enabled:
                logger.error('Non-cuda optical flow calculation not yet supported for ' + algorithm, stack_info=True,
                             exc_info=True)
                sys.exit(1)

            algo_supports_cuda = True
            self.flow_algo = cv2.cuda_DensePyrLKOpticalFlow.create()
            self.flow_algo.setMaxLevel(10)  # default 3
            self.flow_algo.setWinSize((3, 3))  # default 13, 13
            # flow_algo.setNumIters(30) # 30 def

        elif algorithm == "nvidia2":
            # This method was a mess when I attempted to implement it.
            # From what I've seen on the net, others have not had much success, either.

            logger.error('Optical flow calculation not yet supported for ' + algorithm, stack_info=True,
                         exc_info=True)
            sys.exit(1)

            # use_cuda = True
            # flow_algo = cv2.cuda_NvidiaOpticalFlow_2_0.create((height_width[1], height_width[0]),
            #                                             outputGridSize=1,  # 1,2, 4.  Higher is less accurate.
            #                                             enableCostBuffer=True,
            #                                             enableTemporalHints=True, )

        else:
            logger.error('Optical flow algorithm not yet implemented.')

        return algo_supports_cuda

    def add_visualization(self, frame, flow, magnitude, angle, visualize_as, upper_mag_threshold, mask=None,
                              image_1_gray=None, vector_scalar=1):

        # if visualize_as in ['gaze_shifted_hsv']:
        #     logger.exception('This visualization method is only available for pupil labs data folders')
        image_out = False
        frame_out = False

        if visualize_as == "streamlines":

            image_out = self.visualize_flow_as_streamlines(frame, flow)
            frame_out = av.VideoFrame.from_ndarray(image_out, format='bgr24')

        elif visualize_as == "vectors":

            image_out = self.visualize_flow_as_vectors(frame, magnitude, angle, vector_scalar=vector_scalar)
            frame_out = av.VideoFrame.from_ndarray(image_out, format='bgr24')

        elif visualize_as == "hsv_overlay" or visualize_as == "hsv_stacked":

            hsv_flow = self.visualize_flow_as_hsv(magnitude, angle, upper_mag_threshold)

            if visualize_as == "hsv_overlay":
                #  Crazy that I'm making two color conversion here
                image_out = cv2.addWeighted(cv2.cvtColor(image_1_gray, cv2.COLOR_GRAY2BGR), 0.1, hsv_flow, 0.9, 0)

            elif visualize_as == "hsv_stacked":
                image_out = np.concatenate((frame, hsv_flow), axis=0)
            else:
                logger.error('Visualization method not recognized.')

            frame_out = av.VideoFrame.from_ndarray(image_out, format='bgr24')

        return image_out, frame_out

    def set_stream_dimensions(self, stream, visualize_as, height, width):

        stream.width = width

        if visualize_as == "hsv_stacked":
            stream.height = height * 2
        else:
            stream.height = height

        return stream

    def calculate_flow(self, video_out_name=False, algorithm="deepflow", visualize_as="hsv_stacked",
                       hist_params=(100, 0, 40),
                       vector_scalar=1,
                       lower_mag_threshold=False,
                       upper_mag_threshold=False,
                       save_input_images=False,
                       save_midpoint_images=False,
                       save_output_images=False):

        def encode_frame(outgoing_frame, raw_frame_in, stream_in):
            for packet_out in stream.encode(outgoing_frame):
                packet_out.stream = stream_in
                packet_out.time_base = time_base
                packet_out.pts = raw_frame_in.pts
                packet_out.dts = raw_frame_in.dts
                container_out.mux(packet_out)

        container_in = av.open(self.file_path)
        average_fps = container_in.streams.video[0].average_rate
        num_frames = container_in.streams.video[0].frames
        time_base = container_in.streams.video[0].time_base

        height = container_in.streams.video[0].height
        width = container_in.streams.video[0].width

        video_out_name = self.source_file_name + '_' + algorithm + '_' + visualize_as + '.mp4'


        # container_in.sort_dts = True
        # container_in.flush_packets = True

        ##############################
        # prepare video out
        if os.path.isdir(self.video_out_path) is False:
            os.makedirs(self.video_out_path)

        container_out = av.open(os.path.join(self.video_out_path, video_out_name), mode="w", timeout=None)
        try:
            subprocess.check_output('nvidia-smi')
            #print('Nvidia GPU detected!')#
            stream = container_out.add_stream("h264_nvenc", framerate=average_fps)
        except Exception:  # this command not being found can raise quite a few different errors depending on the configuration
            stream = container_out.add_stream("libx264", framerate=average_fps)
            # print('No Nvidia GPU in system!  Defaulting to a different encoder')

        stream.options["crf"] = "10"
        stream.pix_fmt = container_in.streams.video[0].pix_fmt
        stream.time_base = time_base
        stream = self.set_stream_dimensions(stream, visualize_as, height, width)

        ##############################
        # Prepare for flow calculations

        algo_supports_cuda = self.set_flow_object(algorithm, (height, width))

        if algo_supports_cuda and self.cuda_enabled:

            image1_gpu = cv2.cuda_GpuMat()
            image2_gpu = cv2.cuda_GpuMat()
            # flow_out = cv2.cuda_GpuMat()
            # foreground_mask_gpu = cv2.cuda_GpuMat()
            self.clahe = cv2.cuda.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))

        else:
            self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))

        world_index = 0
        for raw_frame in tqdm(container_in.decode(video=0), desc="Generating " + video_out_name, unit='frames',
                              total=num_frames):
            # First frame
            if raw_frame.index == 0:

                prev_frame = raw_frame.to_ndarray(format='bgr24')
                prev_frame = self.filter_frame(prev_frame)

                # Apply histogram normalization.
                if algo_supports_cuda and self.cuda_enabled:
                    image2_gpu.upload(prev_frame)
                    image2_gpu = cv2.cuda.cvtColor(image2_gpu, cv2.COLOR_BGR2GRAY)
                    image2_gpu = self.clahe.apply(image2_gpu, cv2.cuda_Stream.Null())
                else:
                    image2_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    image2_gray = self.clahe.apply(image2_gray)

                # Save input images?
                if save_input_images:
                    if os.path.isdir(self.raw_frames_out_path) is False:
                        os.makedirs(self.raw_frames_out_path)
                    raw_frame.to_image().save(
                        os.path.join(self.raw_frames_out_path, '{:06d}.png'.format(raw_frame.index)))

                prev_raw_frame = raw_frame

                # Write out a blank first frame
                frame_out = np.zeros([raw_frame.height, raw_frame.width, 3], dtype=np.uint8)
                frame_out = av.VideoFrame.from_ndarray(frame_out, format='bgr24')

                encode_frame(frame_out, raw_frame, stream)

                continue

            else:
                frame = raw_frame.to_ndarray(format='bgr24')
                # self.modify_frame(frame, raw_frame.index)
                frame = self.filter_frame(frame)

            # Calculate flow
            if algo_supports_cuda and self.cuda_enabled:
                flow, image1_gray, image2_gpu = self.calculate_flow_for_frame(frame,
                                                     image1_gpu,
                                                     image2_gpu,
                                                     use_cuda=True)
            else:
                flow, image1_gray, image2_gray = self.calculate_flow_for_frame(frame,
                                                     cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                                     image2_gray,
                                                     use_cuda=False)

            # Convert flow to mag / angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            angle = np.pi + angle
            magnitude = self.filter_magnitude(magnitude, frame)
            magnitude = self.apply_magnitude_thresholds_and_rescale(magnitude, lower_mag_threshold, upper_mag_threshold)

            # Convert flow to visualization
            image_out, frame_out = self.add_visualization(frame,
                                                          flow,
                                                          magnitude,
                                                          angle,
                                                          visualize_as,
                                                          upper_mag_threshold,
                                                          image_1_gray=image1_gray,
                                                          vector_scalar=vector_scalar)

            # Store the histogram of avg magnitudes
            if raw_frame.index == 1:
                mag_hist = np.histogram(magnitude, hist_params[0], (hist_params[1], hist_params[2]))
                # Store the first flow histogram
                cumulative_mag_hist = mag_hist[0]
            else:
                # Calc cumulative avg flow magnitude by adding the first flow histogram in a weighted manner
                cumulative_mag_hist = np.divide(
                    np.sum([np.multiply((raw_frame.index - 1), cumulative_mag_hist), mag_hist[0]], axis=0),
                    raw_frame.index - 1)

            # Save input images?
            if save_input_images:
                if os.path.isdir(self.raw_frames_out_path) is False:
                    os.makedirs(self.raw_frames_out_path)
                raw_frame.to_image().save(
                    os.path.join(self.raw_frames_out_path, '{:06d}.png'.format(raw_frame.index)))

            # Save midpoint images?
            if save_midpoint_images:
                if os.path.isdir(self.mid_frames_out_path) is False:
                    os.makedirs(self.mid_frames_out_path)
                cv2.imwrite(str(os.path.join(self.mid_frames_out_path, '{:06d}.png'.format(raw_frame.index))),
                            image1_gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # Save output images?
            if save_output_images:
                if os.path.isdir(self.flow_frames_out_path) is False:
                    os.makedirs(self.flow_frames_out_path)
                cv2.imwrite(str(os.path.join(self.flow_frames_out_path, 'frame-{}.png'.format(raw_frame.index))),
                            image_out)

            # Add packet to video
            frame_out.pts = raw_frame.pts
            frame_out.time_base = raw_frame.time_base
            encode_frame(frame_out, raw_frame, stream)

            prev_raw_frame = raw_frame
            world_index = world_index + 1

        # for packet in stream.encode():
        #     packet.stream = stream
        #     packet.time_base = time_base
        #     packet.pts = raw_frame.pts + 1000 + world_index
        #     packet.dts = raw_frame.pts + 1000 + world_index
        #     container_out.mux(packet)

        # Close the file
        container_out.close()
        container_in.close()

        #  Save out magnitude data pickle and image
        if os.path.isdir(self.magnitude_out_path) is False:
            os.makedirs(self.magnitude_out_path)

        mag_pickle_filename = self.source_file_name + '_' + algorithm + '_' + visualize_as + '_mag.pickle'
        dbfile = open(os.path.join(self.magnitude_out_path, mag_pickle_filename), 'wb')
        pickle.dump({"values": cumulative_mag_hist, "bins": mag_hist[1]}, dbfile)
        dbfile.close()

        mag_image_filename = self.source_file_name + '_' + algorithm + '_' + visualize_as + '_mag.jpg'
        mag_image_fileloc = os.path.join(self.magnitude_out_path, mag_image_filename)
        self.generate_mag_histogram(mag_image_fileloc, cumulative_mag_hist, mag_hist[1])

    def calculate_flow_for_frame(self, raw_frame, frame1, frame2, use_cuda):

        # Calculate flow.  If possible, use cuda.
        if use_cuda:
            # self.temp_fun(frame,raw_frame)
            frame1.upload(raw_frame)
            image1_gpu = cv2.cuda.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            image1_gpu = self.clahe.apply(image1_gpu, cv2.cuda_Stream.Null())
            flow = self.flow_algo.calc(image1_gpu, frame2, flow=None)
            # move images from gpu to cpu
            image1_gray = image1_gpu.download()
            image2_gpu = image1_gpu.clone()
            flow = flow.download()

            return flow, image1_gray, image2_gpu

        else:
            image1_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            image1_gray = self.clahe.apply(image1_gray)
            flow = self.flow_algo.calc(image1_gray, frame2, flow=None)
            image2_gray = image1_gray

            return flow, image1_gray, image2_gray

    def modify_frame(self,frame,frame_index):
        pass

    def convert_nvidia_flow(self, flow_algo, flow_out, image1_gpu, image2_gpu):

        flow_gpu = cv2.cuda_GpuMat(image1_gpu.size())
        flow = flow_algo.calc(image1_gpu, image2_gpu, flow_gpu, cv2.cuda_Stream.Null())

        flow_nums = np.array(image1_gpu.size())
        flow_algo.convertToFloat(flow[0].download(), flow_nums)

        # flowOut = np.array(flow[0].size())
        # f0 = flow[0]
        # f1 = flow[1]
        # a=1
        # flow = flow_algo.convertToFloat(flow[0],flowOut)

    def apply_magnitude_thresholds_and_rescale(self, magnitude, lower_mag_threshold=False, upper_mag_threshold=False):

        if lower_mag_threshold:
            magnitude[magnitude < lower_mag_threshold] = 0

        if upper_mag_threshold:
            magnitude[magnitude > upper_mag_threshold] = upper_mag_threshold

        magnitude = cv2.normalize(magnitude, None, 0, np.nanmax(magnitude), cv2.NORM_MINMAX, -1)

        # magnitude = np.nan_to_num(magnitude) #nans set to 0, inf set to np.max(magnitude)

        return magnitude

    def visualize_flow_as_streamlines(self, frame, flow):

        # dbfile = open(os.path.join(self.video_out_path,"streamlines.pickle"), 'wb')
        # pickle.dump({"frame": frame, "flow": flow},dbfile)
        # dbfile.close()

        x = np.arange(0, np.shape(frame)[0], 1)
        y = np.arange(0, np.shape(frame)[1], 1)
        grid_x, grid_y = np.meshgrid(y, x)

        start_grid_res = 10
        start_pts = np.array(
            [[y, x] for x in np.arange(0, np.shape(frame)[0], start_grid_res)
             for y in np.arange(0, np.shape(frame)[1], start_grid_res)])

        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)

        plt.streamplot(grid_x, grid_y, -flow[..., 0], -flow[..., 1],
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

    def visualize_flow_as_hsv(self, magnitude, angle, upper_bound=False):
        '''
        Note that to perform well, this function really needs an upper_bound, which also acts as a normalizing term.
        
        '''

        # create hsv output for optical flow
        hsv = np.zeros([np.shape(magnitude)[0], np.shape(magnitude)[1], 3], np.uint8)

        hsv[..., 0] = angle * 180 / np.pi / 2

        # set saturation to 1
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, -1)

        hsv_8u = np.uint8(hsv)
        return cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

    def visualize_flow_as_vectors(self, frame, magnitude, angle, divisor=15, vector_scalar=1):

        '''Display image with a visualisation of a flow over the top.
        A divisor controls the density of the quiver plot.'''

        # create a blank mask, on which lines will be drawn.
        mask = np.zeros([np.shape(magnitude)[0], np.shape(magnitude)[1], 3], np.uint8)

        if vector_scalar != 1 & vector_scalar != False:
            magnitude = np.multiply(magnitude, vector_scalar)

        vector_x, vector_y = cv2.polarToCart(magnitude, angle)

        for r in range(1, int(np.shape(magnitude)[0] / divisor)):
            for c in range(1, int(np.shape(magnitude)[1] / divisor)):
                origin_x = c * divisor
                origin_y = r * divisor

                endpoint_x = int(origin_x + vector_x[origin_y, origin_x])
                endpoint_y = int(origin_y + vector_y[origin_y, origin_x])

                mask = cv2.arrowedLine(mask, (origin_x, origin_y), (endpoint_x, endpoint_y), color=(0, 0, 255),
                                       thickness=3, tipLength=0.35)

        return cv2.addWeighted(frame, 0.5, mask, 0.5, 0)

    @staticmethod
    def filter_frame(frame):

        thresh1, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_TOZERO)

        return frame

    @staticmethod
    def filter_magnitude(magnitude, frame):

        image1_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_out = cv2.bitwise_and(frame, frame, mask=cv2.threshold(image1_gray, 127, 255, cv2.THRESH_BINARY)[1])
        _, mask = cv2.threshold(image1_gray, 50, 255, cv2.THRESH_BINARY)
        magnitude = cv2.bitwise_and(magnitude, magnitude, mask=mask)

        return magnitude


class pupil_labs_source(video_source):

    def __init__(self, pupil_labs_parent_folder,
                 session_number=False,
                 recording_number=False,
                 export_number=False,
                 analysis_parameters_file='default_analysis_parameters.json'):

        self.analysis_parameters = json.load(open(analysis_parameters_file))

        self.pupil_labs_parent_folder = pupil_labs_parent_folder

        self.session_number = session_number
        self.recording_number = recording_number
        self.export_number = export_number

        self.recording_folder = self.set_recording_folder()
        self.export_folder = self.set_export_folder()

        self.world_video_path = os.path.join(self.recording_folder, 'world.mp4')
        self.gaze_data = self.import_gaze_from_exports()
        self.processed_gaze_data = False

        super().__init__(self.world_video_path, out_parent_dir=self.export_folder)

        self.raw_frames_out_path = os.path.join(self.out_parent_dir, 'world_images')
        self.mid_frames_out_path = os.path.join(self.out_parent_dir, 'flow_mid_images')
        self.flow_frames_out_path = os.path.join(self.out_parent_dir, 'flow_images')
        self.video_out_path = self.out_parent_dir
        self.magnitude_out_path = os.path.join(self.out_parent_dir, 'flow_magnitude_data')

        f = open(os.path.join(self.recording_folder, 'info.player.json'))
        self.player_info = json.load(f)
        f.close()

    def modify_frame(self, frame, frame_index):

        self.gaze_center_Frame(frame, frame_index)

    def get_median_gaze_for_frame(self, frame_index):

        gaze_samples_on_frame = self.gaze_data[self.gaze_data['world_index'] == frame_index]
        gaze_samples_on_frame = gaze_samples_on_frame[
            gaze_samples_on_frame['confidence'] > self.analysis_parameters['pl_confidence_threshold']]

        if len(gaze_samples_on_frame) == 0:
            # Out of frame
            return False

        median_x = np.median(gaze_samples_on_frame['norm_pos_x'])
        median_y = 1 - np.median(gaze_samples_on_frame['norm_pos_y'])

        if median_x < 0 or median_y < 0 or median_x > 1 or median_y > 1:
            # Out of frame
            return False

        return (median_x, median_y)

    def process_gaze_data(self):

        idx_list = np.unique(self.gaze_data.world_index)
        med_xy = [self.get_median_gaze_for_frame(idx) for idx in idx_list]
        med_x, med_y = zip(*med_xy)

        processed_gaze_data = pd.DataFrame({'median_x': med_x, 'median_y': med_y})
        processed_gaze_data.rolling(3).median()

        self.processed_gaze_data = processed_gaze_data

    def visualize_gaze_centered_frame(self, frame, frame_index):

        if self.processed_gaze_data is False:
            self.process_gaze_data()

        for world_index, data in self.processed_gaze_data.iterrows():

            median_x = data['median_x']
            median_y = data['median_y']

            height = np.shape(frame)[0]
            width = np.shape(frame)[1]

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

            # cv2.imwrite('temp.png', new_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            return new_image

    def draw_gaze_in_head(self, frame, frame_index):

        med_xy = self.get_median_gaze_for_frame(frame_index)

        if med_xy:

            median_x, median_y = med_xy

            height = np.shape(frame)[0]
            width = np.shape(frame)[1]

            frame = cv2.line(frame, (int(width * median_x), 0), (int(width * median_x), height),
                             (255, 0, 0), thickness=2)

            frame = cv2.line(frame, (0, int(height * median_y)), (width, int(height * median_y)),
                             (255, 0, 0), thickness=2)

            cv2.imwrite('temp.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            return frame

    def import_gaze_from_exports(self):
        gaze_positions_path = os.path.join(self.export_folder, 'gaze_positions.csv')

        if os.path.exists(gaze_positions_path) is False:
            logger.error('No gaze_positions found in the exports folder.')

        # Defaults to the most recent pupil export folder (highest number)
        return pd.read_csv(gaze_positions_path)

    def set_export_folder(self):

        exports_parent_folder = os.path.join(self.recording_folder, 'exports')

        if os.path.exists(exports_parent_folder) is False:

            os.mkdir(exports_parent_folder)
            export_folder_path = os.path.join(exports_parent_folder, '001')
            os.mkdir(export_folder_path)
            return export_folder_path

        else:
            if self.export_number is False:
                export_folder_list = []
                [export_folder_list.append(name) for name in os.listdir(exports_parent_folder) if name[0] != '.']
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

    def add_visualization(self,
                          frame,
                          flow,
                          magnitude,
                          angle,
                          visualize_as,
                          upper_mag_threshold,
                          mask=None,
                          image_1_gray=None,
                          vector_scalar=1):

        if visualize_as in ['hsv_stacked']:
            logger.exception('This visualization method is not available for pupil labs data folders')

        image_out, frame_out = super().add_visualization(frame,
                                                         flow,
                                                         magnitude,
                                                         angle,
                                                         visualize_as,
                                                         upper_mag_threshold,
                                                         mask=mask,
                                                         image_1_gray=image_1_gray,
                                                         vector_scalar=vector_scalar)

        if visualize_as == "gaze_centered_hsv":

            image_out = self.visualize_gaze_centered_frame(frame)
            frame_out = av.VideoFrame.from_ndarray(image_out, format='bgr24')

        return image_out, frame_out

if __name__ == "__main__":
    a_file_path = os.path.join("pupil_labs_data", "GD-Short-Driving-Video")

    source = pupil_labs_source(a_file_path,recording_number='007')
    source.cuda_enabled = True

    source.calculate_flow(algorithm='tvl1', visualize_as="hsv_overlay", lower_mag_threshold=False,
                          upper_mag_threshold=25,
                          vector_scalar=3, save_input_images=False, save_output_images=True)

    ## STOPPED HERE
    # Gaze centered normal frame, but not flow
    # Must figure out how to center before calculating flow
    # If statement to center, then pass in centered frame, then remove padding?

    # a_file_path = os.path.join("videos", "cb1.mp4")
    # # a_file_path = os.path.join("demo_input_video", "linear_travel.mp4")
    # a_file_path = os.path.join("videos", "tamer.mp4")
    # source = video_source(a_file_path)
    # source.cuda_enabled = True
    # source.calculate_flow(algorithm='deepflow', visualize_as="hsv_overlay", lower_mag_threshold=False, upper_mag_threshold=5,
    #                       vector_scalar=3, save_input_images=False, save_output_images=False)

    # # a_file_path = os.path.join("videos", "640_480_60Hz.mp4")
    # a_file_path = os.path.join("videos", "tamer.mp4")
    # source = video_source(a_file_path)
    # source.cuda_enabled = True
    # source.calculate_flow(algorithm='tvl1', visualize_as="hsv_overlay", lower_mag_threshold = False, upper_mag_threshold=25,
    #                        vector_scalar=3, save_input_images=False, save_output_images=False)

    #
    # a_file_path = os.path.join("demo_input_video", "moving_sphere.mp4")
    # source = video_source(a_file_path)
    # source.cuda_enabled = True
    # source.calculate_flow(algorithm='tvl1', visualize_as="vectors", lower_mag_threshold = False, upper_mag_threshold=25,
    #                        vector_scalar=3, save_input_images=False, save_output_images=False, fps = 30)
