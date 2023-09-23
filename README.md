# Retinal flow toolkit 
( generate gaze-centered optical flow from a movie )

A simple module for the calculation of optic flow.  

The most relevant file is video_source.py, in which lives the video_source class for converting a video into optic flow.  You also have the option of saving out intermediate frames of image data.

It will take advantage of CUDA if enabled for opencv. 

To run, create a video_source object from an input movie file, and then calculate_flow()

```
a_file_path = os.path.join("demo_input_video", "linear_travel.mp4")
source = video_source(a_file_path)
source.calculate_flow(algorithm='tvl1', visualize_as="hsv_stacked", lower_mag_threshold = False, upper_mag_threshold=25,
                           vector_scalar=3, save_input_images=False, save_output_images=False, fps = 30)

```

If you don't have cuda enabled for opencv, set <video_source>.cuda_enabled = False

## Flow algorithms and settings.

Nothing elegant here - you'll have to go into the code for more info. I encourage the user to look at <video_source>.create_flow_object() to see the different algorithms that are enabled, and to see the parameters for existing flow algorithms.  I have not made all parameters for all algorithms visible.  Use the opencv docs to play around.

At the time of writing this, I have implemented...
* deepflow
* farneback
* brox with cuda support
* tvl1 with cuda support
* pyrLK with cuda support

I've had the best quality output with tvl1. \n
I tried to implement cuda_NvidiaOpticalFlow_2_0, but ran into indecipherable errors.  So, it lingers, half impelemented.

## Options for "visualize_as" include:

"hsv_stacked"

![image](https://user-images.githubusercontent.com/8962011/212419240-33461130-e360-4fcd-b19a-da44854cfd65.png)

"hsv_overlay"

![image](https://user-images.githubusercontent.com/8962011/212419759-4bea48b8-a649-4f36-9422-5bea5d1226a2.png)

"vectors"

![image](https://user-images.githubusercontent.com/8962011/212420144-616493dd-4b6f-4cf8-af41-ece5906df25b.png)

"streamlines"

![image](https://user-images.githubusercontent.com/8962011/212419849-328487d7-694f-458c-bf71-82fd5aa83851.png)

## Output

Output frames and movies will be organized in sensibly-named subfolders of /flow_out/<movie_name>. 

## Setting the upper_mag_threshold

Output will be placed in /flow_out/<movie_name>.
HSV color representations of flow are best if scaled to a range defined by a reasonable maximum value. For that reason, after processing with upper_mag_threshold = False,
/flow_out/<movie_name>/magnitude_data will contain images of the cumulative distribution function of magnitude of flow vector length.  Set your threshold somewhere around where the CDF reaches 1. This will vary for algorithm and by the motion statistics of the environment / task / stimulus class (?) represented in your video.

![image](https://user-images.githubusercontent.com/8962011/212422892-a28d9352-9b66-471e-a26f-13cd9f050c49.png)

## Dependencies

Tested with py3.8 and opencv 4.6 compiled with CUDA.

Dependencies include...

* numpy
* av for movie makin'
* logging
* pickle
* tqdm for the nifty progress bar that shows up in your console.
* matplotlib

![image](https://user-images.githubusercontent.com/8962011/212423219-734e351a-0139-4596-ac8b-8d5bc28c7316.png)

-------------------
Some notes on installing OpenCV with CUDA:

* This is my fav guide: https://jordanbenge.medium.com/anaconda3-opencv-with-cuda-gpu-support-for-windows-10-e038569e228

* When downloading opencv and opencv contribs, make sure you get the latest source code realease. 

* Depending on how you installed anaconda, your environments will be found in either 
     - C:\Users\<username>\anaconda3\envs
     - or C:\Users\<username>\.conda\envs  (.conda is a hidden folder.  If you can't see it, google how to show hidden folders in windows explorer)

* If installing to a new env, make sure numpy is installed

* all \ in the *.bat files have to changed to /

* When calling the bat files, you have to first activate your target conda environment

* I had to include the CMAKE build argument  -DBUILD_SHARED_LIBS=OFF . 

* In the configure_and_build_opencv.bat, you must update DCUDA_ARCH_BIN based on your card's cuda architecture (https://en.wikipedia.org/wiki/CUDA)

* It looks like cuda and cudnn can both be installed using anaconda's gui.  I haven't tried this yet.  If it works, it would save a lot of time, because installing them is a pain 🙂

* If you need to, here's how to install Cuda and CUDANN : https://medium.com/geekculture/install-cuda-and-cudnn-on-windows-linux-52d1501a8805  .

* Note that CUDANN does not yet work for the newest version of Cuda, version 12, but does work for version 11.  Here is a link where you can download Cuda 11. https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10

* Install requires that you have installed visual studio using certain settings that enable cmake support.  If you aren't sure that you did that, here's how to modify Visual STudio to connect to Cmake without uninstalling/reinstalling. https://learn.microsoft.com/en-us/cpp/linux/download-install-and-setup-the-linux-development-workload?view=msvc-170 

* If after installing properly, you may still get a dependency error when attempting to load the cuda module.  Some wonderful person on the OpenCV forums helped me to resolve this issue.  Not ideal, but it requires manually adding those dependencies to the path in the python files.  Here's the link! 
https://github.com/cudawarped/opencv-experiments/blob/master/nbs/ImportError_dll_load_failed_while_importing_cv2.ipynb

