# Retinal flow toolkit

A simple module for the calculation of optic flow.  

The most relevant file is flow_source.py, in which lives the flow_source class for converting a video into optic flow.  You also have the option of saving out intermediate frames of image data.

It will take advantage of CUDA if enabled for opencv. 

To run, create a flow_source object from an input movie file, and then calculate_flow()

```
a_file_path = os.path.join("demo_input_video", "linear_travel.mp4")
source = flow_source(a_file_path)
source.calculate_flow(algorithm='tvl1', visualize_as="hsv_stacked", lower_mag_threshold = False, upper_mag_threshold=25,
                           vector_scalar=3, save_input_images=False, save_output_images=False, fps = 30)

```

If you don't have cuda enabled for opencv, set <flow_source>.cuda_enabled = False
I encourage the user to look at <flow_source>.create_flow_object() to see the different algorithms that are enabled, and to see the parameters for existing flow algorithms.

## Options for "visualize_as" include:

"hsv_stacked"

![image](https://user-images.githubusercontent.com/8962011/212419240-33461130-e360-4fcd-b19a-da44854cfd65.png)

"hsv_overlay"

![image](https://user-images.githubusercontent.com/8962011/212419759-4bea48b8-a649-4f36-9422-5bea5d1226a2.png)

"vectors"

![image](https://user-images.githubusercontent.com/8962011/212420144-616493dd-4b6f-4cf8-af41-ece5906df25b.png)

"streamlines"

![image](https://user-images.githubusercontent.com/8962011/212419849-328487d7-694f-458c-bf71-82fd5aa83851.png)

## Setting the upper_mag_threshold

Output will be placed in /flow_out/<movie_name>.
HSV color representations of flow are best if scaled to a range defined by a reasonable maximum value. For that reason, after processing with upper_mag_threshold = False,
/flow_out/<movie_name>/magnitude_data will contain images of the cumulative distribution function of magnitude of flow vector length.  Set your threshold somewhere around where the CDF reaches 1. This will vary for algorithm and by the motion statistics of the environment / task / stimulus class (?) represented in your video.

![image](https://user-images.githubusercontent.com/8962011/212422892-a28d9352-9b66-471e-a26f-13cd9f050c49.png)

## Dependencies


* numpy
* av for movie makin'
* logging
* pickle
* tqdm for the nifty progress bar that shows up in your console.
* matplotlib

![image](https://user-images.githubusercontent.com/8962011/212423219-734e351a-0139-4596-ac8b-8d5bc28c7316.png)




