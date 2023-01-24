# 2023 Gabriel J. Diaz @ RIT

import os
import sys
import numpy as np
import av
import logging
import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


try:
    os.add_dll_directory("D://opencvgpu//opencv_build_310//bin")
    os.add_dll_directory("C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.8//bin")
    import cv2
except:
    import cv2

sys.path.append('core')

# np.set_printoptions(precision=3)
# np.set_printoptions(suppress=True)
logger = logging.getLogger(__name__)

# These lines allow me to see logging.info messages in my jupyter cell output
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
# logger.setLevel(logging.DEBUG)


class flow_source():

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

        self.cuda_enabled = True

        self.streamline_points = None

    def generate_mag_histogram(self, mag_image_fileloc, mag_values,bins):

        import matplotlib.pyplot as plt

        # This prevents a warning that was present on some versions of mpl
        from matplotlib import use as mpl_use
        mpl_use('qtagg')

        fig, ax = plt.subplots(figsize=(8, 4))
        # ax.bar(bins[:-1], mag_values, width = .75 * (bins[1]-bins[0]))
        ax.bar(bins[:-1], np.cumsum(mag_values) / sum(mag_values) , width = .75 * (bins[1]-bins[0]))

        # tidy up the figure
        ax.grid(True)

        ax.set_title('flow magnitude')
        ax.set_xlabel('value')
        ax.set_ylabel('likelihood')

        plt.savefig( mag_image_fileloc )

    def create_flow_object(self, algorithm, height_width = False ):

        use_cuda = False

        if algorithm == "deepflow":

            flow_algo = cv2.optflow.createOptFlow_DeepFlow()

        elif algorithm == "farneback":

            if self.cuda_enabled == False:
                flow_algo = cv2.optflow.createOptFlow_Farneback()
            else:
                use_cuda = True
                flow_algo = cv2.cuda_FarnebackOpticalFlow.create()

        elif algorithm == "tvl1":

            if self.cuda_enabled == False:
                logger.error('Non-cuda optical flow calculation not yet supported for ' + algorithm, stack_info=True, exc_info=True)
                sys.exit(1)



            use_cuda = True
            flow_algo = cv2.cuda_OpticalFlowDual_TVL1.create()
            flow_algo.setNumScales(30) # (1/5)^N-1 def: 5
            # flow_algo.setScaleStep(0.7)  #
            # flow_algo.setLambda(0.5)  # default 0.15. smaller = smoother output.
            # flow_algo.setScaleStep(0.7)  # 0.8 by default. Not well documented.  0.7 did better with dots?
            # flow_algo.setEpsilon(0.005)  # def: 0.01
            # flow_algo.setTau(0.5)
            # flow_algo.setGamma(0.5) # def 0

        elif algorithm == "pyrLK":

            if self.cuda_enabled == False:
                logger.error('Non-cuda optical flow calculation not yet supported for ' + algorithm, stack_info=True, exc_info=True)
                sys.exit(1)

            use_cuda = True
            flow_algo = cv2.cuda_DensePyrLKOpticalFlow.create()
            flow_algo.setMaxLevel(10) # def 3
            flow_algo.setWinSize((3,3)) #def 13, 13
            # flow_algo.setNumIters(30) # 30 def

        elif algorithm == "nvidia2":

            logger.error('Optical flow calculation not yet supported for ' + algorithm, stack_info=True,
                         exc_info=True)
            sys.exit(1)

            if self.cuda_enabled == False:
                logger.error('Non-cuda optical flow calculation not yet supported for ' + algorithm, stack_info=True, exc_info=True)
                sys.exit(1)

            use_cuda = True
            flow_algo = cv2.cuda_NvidiaOpticalFlow_2_0.create((height_width[1], height_width[0]),
                                                        outputGridSize=1,  # 1,2, 4.  Higher is less accurate.
                                                        enableCostBuffer=True,
                                                        enableTemporalHints=True, )

        else:
            logger.error('Optical flow algorithm not yet implemented.')

        return use_cuda, flow_algo

    def convert_flow_to_frame(self, frame, flow, magnitude, angle, visualize_as, upper_mag_threshold, mask = None, image_1_gray = None, vector_scalar = 1):

        if visualize_as == "streamlines":

            image_out = self.visualize_flow_as_streamlines(frame, flow)
            frame_out = av.VideoFrame.from_ndarray(image_out, format='bgr24')

        elif visualize_as == "vectors":

            image_out = self.visualize_flow_as_vectors(frame, magnitude, angle, vector_scalar = vector_scalar)
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

        else:
            logger.exception('visualize_as string not supported')

        return image_out, frame_out


    def calculate_flow(self, video_out_name = False, algorithm = "deepflow", visualize_as="hsv_stacked",
            hist_params = (100, 0,40),
            vector_scalar = 1,
            lower_mag_threshold = False, 
            upper_mag_threshold = False,
            save_input_images = False,
            save_midpoint_images=False,
            save_output_images = False ):

        container_in = av.open(self.file_path)
        average_fps = container_in.streams.video[0].average_rate
        num_frames = container_in.streams.video[0].frames

        video_out_name = self.source_file_name + '_' + algorithm + '_' + visualize_as + '.mp4'

        ##############################
        # prepare video out
        if os.path.isdir(self.video_out_path) is False:
            os.makedirs(self.video_out_path)

        container_out = av.open(os.path.join(self.video_out_path, video_out_name), mode="w", timeout = None)
        stream = container_out.add_stream("libx264", framerate=average_fps)
        stream.options["crf"] = "20"
        stream.pix_fmt = container_in.streams.video[0].pix_fmt
        stream.time_base = container_in.streams.video[0].time_base

        ##############################
        # Prepare for flow calculations

        height = container_in.streams.video[0].height
        width = container_in.streams.video[0].width
        use_cuda, flow_algo = self.create_flow_object(algorithm, (height, width))

        if use_cuda and self.cuda_enabled:
            image1_gpu = cv2.cuda_GpuMat()
            image2_gpu = cv2.cuda_GpuMat()
            flow_out = cv2.cuda_GpuMat()
            foreground_mask_gpu = cv2.cuda_GpuMat()
            clahe = cv2.cuda.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
        else:
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))

        background_subtractor = cv2.createBackgroundSubtractorKNN()

        first_pts = []

        for raw_frame in tqdm(container_in.decode(video=0), desc="Generating " + video_out_name, unit= 'frames',total = num_frames):

            if raw_frame.index == 0:

                stream.width = raw_frame.width

                if visualize_as == "hsv_stacked":
                    stream.height = raw_frame.height * 2
                else:
                    stream.height = raw_frame.height

                prev_frame = raw_frame.to_ndarray(format='bgr24')
                prev_frame = self.filter_frame(prev_frame)

                foreground_mask = background_subtractor.apply(prev_frame)

                if use_cuda and self.cuda_enabled:

                    image2_gpu.upload(prev_frame)
                    if algorithm != "nvidia2":
                        image2_gpu = cv2.cuda.cvtColor(image2_gpu, cv2.COLOR_BGR2GRAY)
                        image2_gpu = clahe.apply(image2_gpu, cv2.cuda_Stream.Null())

                else:
                    image2_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    image2_gray = clahe.apply(image2_gray)

                if save_input_images:

                    if os.path.isdir(self.raw_frames_out_path) is False:
                        os.makedirs(self.raw_frames_out_path)

                    raw_frame.to_image().save(os.path.join(self.raw_frames_out_path, '{:06d}.png'.format(raw_frame.index)))

                continue

            else:

                frame = raw_frame.to_ndarray(format='bgr24')
                frame = self.filter_frame(frame)
                foreground_mask = background_subtractor.apply(frame)

            if save_input_images:

                if os.path.isdir(self.raw_frames_out_path) is False:
                    os.makedirs(self.raw_frames_out_path)

                raw_frame.to_image().save(os.path.join(self.raw_frames_out_path, '{:06d}.png'.format(raw_frame.index)))

            # Calculate flow.  If possible, use cuda.
            if use_cuda and self.cuda_enabled:

                image1_gpu.upload(frame)

                image1_gpu = cv2.cuda.cvtColor(image1_gpu, cv2.COLOR_BGR2GRAY)
                image1_gpu = clahe.apply(image1_gpu, cv2.cuda_Stream.Null())

                # self.convert_nvidia_flow(flow_algo, flow_out, image1_gpu, image2_gpu)
                flow = flow_algo.calc(image1_gpu, image2_gpu, flow=None)

                # move images from gpu to cpu
                image1_gray = image1_gpu.download()
                image2_gpu = image1_gpu.clone()
                flow = flow.download()

            else:
                image1_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image1_gray = clahe.apply(image1_gray)
                flow = flow_algo.calc(image1_gray, image2_gray, flow=None)
                image2_gray = image1_gray

            if save_midpoint_images:

                if os.path.isdir(self.mid_frames_out_path) is False:
                    os.makedirs(self.mid_frames_out_path)

                cv2.imwrite(str(os.path.join(self.mid_frames_out_path, '{:06d}.png'.format(raw_frame.index))), image1_gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # Convert flow to mag / angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            angle = np.pi + angle

            magnitude = self.filter_magnitude(magnitude,foreground_mask)

            # Store the histogram of avg magnitudes
            if raw_frame.index == 1:
                mag_hist = np.histogram(magnitude, hist_params[0], (hist_params[1], hist_params[2]))
                # Store the first flow histogram
                cumulative_mag_hist = mag_hist[0]
            else:
                # Calc cumulative avg flow magnitude by adding the first flow histogram in a weighted manner
                cumulative_mag_hist = np.divide(np.sum([np.multiply((raw_frame.index - 1), cumulative_mag_hist), mag_hist[0]], axis=0), raw_frame.index - 1)

            magnitude = self.apply_magnitude_thresholds_and_rescale(magnitude, lower_mag_threshold, upper_mag_threshold)

            # Convert flow to visualization
            image_out, frame_out = self.convert_flow_to_frame(frame,
                                                              flow,
                                                              magnitude,
                                                              angle,
                                                              visualize_as,
                                                              upper_mag_threshold,
                                                              image_1_gray = image1_gray,
                                                              vector_scalar = vector_scalar)


            if save_output_images:

                if os.path.isdir(self.flow_frames_out_path) is False:
                    os.makedirs(self.flow_frames_out_path)

                cv2.imwrite(str(os.path.join(self.flow_frames_out_path, 'frame-{}.png'.format(raw_frame.index))), image_out )

            # Add packet to video
            for packet in stream.encode(frame_out):
                container_out.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container_out.mux(packet)

        # Close the file
        container_out.close()

        #  Save out magnitude data pickle and image
        if os.path.isdir(self.magnitude_out_path) is False:
            os.makedirs(self.magnitude_out_path)

        mag_pickle_filename = self.source_file_name + '_' + algorithm + '_' + visualize_as + '_mag.pickle'
        dbfile = open(os.path.join(self.magnitude_out_path, mag_pickle_filename), 'wb')
        pickle.dump( {"values": cumulative_mag_hist, "bins": mag_hist[1]}, dbfile)
        dbfile.close()

        mag_image_filename = self.source_file_name + '_' + algorithm + '_' + visualize_as + '_mag.jpg'
        mag_image_fileloc = os.path.join(self.magnitude_out_path, mag_image_filename)
        self.generate_mag_histogram(mag_image_fileloc, cumulative_mag_hist, mag_hist[1])

    def convert_nvidia_flow(self,flow_algo, flow_out, image1_gpu, image2_gpu):

        flow_gpu = cv2.cuda_GpuMat(image1_gpu.size())
        flow = flow_algo.calc(image1_gpu, image2_gpu, flow_gpu, cv2.cuda_Stream.Null())

        flow_nums = np.array(image1_gpu.size())
        flow_algo.convertToFloat(flow[0].download(), flow_nums)

        # flowOut = np.array(flow[0].size())
        # f0 = flow[0]
        # f1 = flow[1]
        # a=1
        # flow = flow_algo.convertToFloat(flow[0],flowOut)


    def apply_magnitude_thresholds_and_rescale(self, magnitude, lower_mag_threshold = False, upper_mag_threshold = False):

        if lower_mag_threshold:
            magnitude[magnitude<lower_mag_threshold] = 0

        if upper_mag_threshold:
            magnitude[magnitude>upper_mag_threshold] = upper_mag_threshold

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
                       linewidth=.8) # density = 1

        plt.axis('off')
        plt.imshow(frame)

        canvas = FigureCanvas(fig)
        canvas.draw()  # draw the canvas, cache the renderer

        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
        plt.close('all')
        return image

    def visualize_flow_as_hsv(self, magnitude, angle, upper_bound = False):
        '''
        Note that to perform well, this function really needs an upper_bound, which also acts as a normalizing term.
        
        '''

        # create hsv output for optical flow
        hsv = np.zeros([np.shape(magnitude)[0], np.shape(magnitude)[1], 3], np.uint8)

        hsv[..., 0] =  angle * 180 / np.pi / 2

        # set saturation to 1
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, -1)

        hsv_8u = np.uint8(hsv)
        return cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

    def visualize_flow_as_vectors(self, frame, magnitude, angle, divisor=15, vector_scalar = 1):

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

                mask = cv2.arrowedLine(mask, (origin_x, origin_y), (endpoint_x, endpoint_y),  color=(0, 0, 255), thickness = 3, tipLength = 0.35)


        return cv2.addWeighted(frame, 0.5, mask, 0.5, 0)

    def filter_frame(self,frame):

        thresh1, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_TOZERO)

        # frame = cv2.fastNlMeansDenoising(frame, None, 5, 7, 21)

        # frame = cv2.bilateralFilter(frame,7, 40, 40)
        # frame = cv2.GaussianBlur(frame, (3,3),0)

        return frame

    def filter_magnitude(self,magnitude,foreground_mask):

        magnitude = cv2.bitwise_and(magnitude, magnitude, mask = foreground_mask)

        return magnitude

if __name__ == "__main__":

    a_file_path = os.path.join("demo_input_video", "linear_travel.mp4")
    source = flow_source(a_file_path)
    source.cuda_enabled = True
    source.calculate_flow(algorithm='tvl1', visualize_as="vectors", lower_mag_threshold = False, upper_mag_threshold=25,
                           vector_scalar=3, save_input_images=False, save_output_images=False)

    #
    # a_file_path = os.path.join("demo_input_video", "moving_sphere.mp4")
    # source = flow_source(a_file_path)
    # source.cuda_enabled = True
    # source.calculate_flow(algorithm='tvl1', visualize_as="vectors", lower_mag_threshold = False, upper_mag_threshold=25,
    #                        vector_scalar=3, save_input_images=False, save_output_images=False, fps = 30)