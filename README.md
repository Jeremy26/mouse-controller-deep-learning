# Computer Pointer Controller

*TODO:* Write a short introduction to your project

In this project, we aim to build a mouse pointer controller based on eye movement detection.
This could help disabled people, and could be a nice feature to add.
A relevant and useful way to use Deep Learning.

In this project, we are going to run 4 models at the same time in OpenVino Toolkit.
It should follow this pipeline:
![pipeline schema](pipeline.png)

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

To set up the project:<p>
* Download the Git: `git clone https://github.com/Jeremy26/mouse-controller-deep-learning.git`
* Source the Open Vino environment 
`source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5`
* Install the environment based on the requirements.txt file (virtualenv or conda is prefered)
* Activate it in the Folder: `source bin/activate`
* Download the 4 models we'll use: [Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_model_face_detection_adas_0001.html), [Head Pose](https://docs.openvinotoolkit.org/latest/omz_models_model_head_pose_estimation_adas_0001.html), [Landmark Detection](https://docs.openvinotoolkit.org/latest/omz_models_model_landmarks_regression_retail_0009.html), [Gaze Estimation](https://docs.openvinotoolkit.org/latest/omz_models_model_gaze_estimation_adas_0002.html).
* These Models are also in the OpenVino Model Zoo. Use the model downloader to get them:
Example of a command line I ran:
`/opt/intel/openvino_2021/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o Downloads/starter/models`


## Demo
*TODO:* Explain how to run a basic demo of your model.
I have added a Main.py file that has for mission to implement the pipeline shown above.
Running main.py will run the project.

python``` python main.py -f <Path of xml file of face detection model> \
-fl <Path of xml file of facial landmarks detection model> \
-hp <Path of xml file of head pose estimation model> \
-g <Path of xml file of gaze estimation model> \
-i <Path of input video file or enter cam for taking input video from webcam>```

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

To visualize the output of a model, we'll use the flags.
Example of a command line I ran:
python```python3 main.py -f /Users/jeremycohen/Downloads/starter/models/intel/face-detection-adas-0001/FP16/f
ace-detection-adas-0001.xml -hp /Users/jeremycohen/Downloads/starter/models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -g /Users/jerem
ycohen/Downloads/starter/models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -fl /Users/jeremycohen/Downloads/starter/models/intel/landmarks-regre
ssion-retail-0009/FP16/landmarks-regression-retail-0009.xml -i cam -flags fd ge hp```

The Arguments are:
* -f, -hp, -g, -fl: These are the locations of our models XML files. **Mandatory**
* -i: This is the location of our video, or "CAM" if we're using our webcam. **Mandatory**
* -flags - fd, ge, hp, fld: This is to add flags and visualize our outputs. For example -flags fld is used to visualize the output of the facial landmarks detection model.
<p>
### A note on Unsupported Layers
Given that my computer runs Open Vino 2021.3, Unsupported Layers code from the course has been deprecated.<p>
However, I commented it, and it turns out that it's not needed in the code.<p>
I kept it commented just in case.<p>

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

Taken for the Gaze Estimation on MacOS.

### FP16
* Loading Time: 0.06 s
* Inference Time: 0.002 s
* Preprocessing Time: 7.10^-5 s
* PostProcessing: 1.10^-5s

### FP32
* Loading Time: 0.07s
* Inference Time: 0.002s
* Preprocessing Time, PostProcessing Time: 6.10^-5

### FP16-INT8
* Loading Time: 0.14s (Longer)
* Inference Time: 0.002s
* Preprocessing Time, PostProcessing Time: 6.10^-5

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

Here's the result!<p>
![pipeline img](pipeline_image.png)

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.
* We could add a way to see if the eyes are closed. If they are, then we can consider its a click!

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
