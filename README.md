**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/data_exploration.png
[image2]: ./output_images/color_hist.png
[image3]: ./output_images/bin_spatial.png
[image4]: ./output_images/hog_features.png
[image5]: ./output_images/extract_feature.png
[image6]: ./output_images/window_search.png
[image7]: ./output_images/multiscale.png
[image8]: ./output_images/heatmap.png
[image9]: ./output_images/thresholded_heatmap.png
[image10]: ./output_images/final_box.png
[image11]: ./output_images/pipeline0.png
[image12]: ./output_images/pipeline1.png
[image13]: ./output_images/pipeline2.png
[image14]: ./output_images/pipeline3.png
[image15]: ./output_images/pipeline4.png
[image16]: ./output_images/pipeline5.png

## Data Exploration

Let's take a look at the data set before extracting features and training the classifier. These datasets are comprised of images taken from the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself. You can download the project dataset for [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).

The downloaded data consists of 8792 car images and 8968 non-car images. The shape of the image is (64, 64, 3), and the data type of the image is float32. I have written the `data_look()` function to extract this information. Shown below is an example of each class (vehicle, non-vehicle) of the data set.

![Data Exploration][image1]

## Extract Features
### Histograms of Color
In image processing, a color histogram is a representation of the distribution of colors in an image. A histogram of an image is produced first by discretization of the colors in the image into a number of bins, and counting the number of image pixels in each bin.

In the `color_hist()` function, I split the image into three channels, and then I got each histogram. `np.histogram()` returns a tuple of two arrays. `rhist[0]` contains the counts in each of the bins and `rhist [1]` contains the bin edges. I can compute the bin centers from the bin edges.

Which gives us this result:

![Histograms][image2]

### Spatial Binning of Color
Raw pixel values are quite useful to include in feature vector in searching for cars. I could perform spatial binning on an image and still retain enough information to help in finding vehicles. I wrote a `bin_spatial()` function to convert test image into a feature vector.

Which gives us this result:

![Binned color features][image3]

### HOG Features
The histogram of oriented gradients(HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image.

The `scikit-image` package has a built in function to extract Histogram of Oriented Gradient features. Using this built in function, I defined a `get_hog_features()` function to return HOG features and visualization. The main parameters of this function are `orient`, `pix_per_cell`, and `cell_per_block`.

Car images of grayscale for testing and its corresponding HOG visulization, they look like this:

![HOG Vis][image4]

### Combine and Normalize Features
I have written a function that extracts a feature vector from an image by combining the three techniques shown above. 
`extract_features()` function takes in a list of image filenames, reads them one by one, then applies a color conversion and uses `bin_spatial()` and `color_hist()` and `get_hog_features()` to generate feature vectors.

I almost ready to train a classifier, but I need to normalize my data. `sklearn` package provides me with the `StandardScaler()` method to accomplish this task.

The result of extracting the feature vector from the image and normalizing it is as follows:

![Extract Feature][image5]

## Train a classifier
I tried training a classifier on my dataset. To do this, I defined a labels vector, shuffle and split the data into training and testing sets, and finally, define a classifier and train it. I used the functions defined in previous and `train_test_split()` function in `sklearn` package. In 'Tweak parameters' section of my notebook(p5.ipynb), you're given all the code to extract features and train a linear SVM.

I tweaked many parameters and see how the results change. Repeated experiments have found a combination of parameters that yield the best results. After learning about the whole data using the parameters, I saved the parameters and classifier using pickle.

## Hog Sub-sampling Window Search
Now it's time to search for cars and I have all the tools for it. I trained my classifier, then ran sliding window search, extracted features, and predicted whether each window contains a car or not.

In my first implementation, I extracted HOG features from each individual window, but it was inefficient. To speed up, I modified the code by extracting HOG features just once for the entire region of interest and subsampling that array for each sliding window. In this way, I implemented `find_cars()`, an efficient function that extracts HOG features only once. In `find_cars()` function, each window is defined by scaling factor where a scale of 1 would result in a window that's 8x8 cells then the overlap of each window is in terms of the cell distance. I ran `find_cars()` function multiple times for different scale values to generate multiple-scaled search windows.

The results are as follows:

|Window Search|Multiple Scaled|
|-------------|---------------|
|![WS][image6]|![MS][image7]  |

## False Positive
As you can see from the image above, there may be overlapping detection of the vehicle. It can also detect objects that are not vehicles. I build a heat-map from these detections in order to combine overlapping detections and remove false positives. The "hot part" of the heat map is the location of the car, and I removed false positives by applying a threshold. To do this, I wrote two functions, `get_heatmap ()` and `apply_threshold ()`.

Finally, find final boxes from heatmap and put bounding boxes around the labeled regions. I used the `label()` function from `scikit-image` and wrote a `draw_labeled_bboxes()` function. The following images show this process.

|Heatmap           |Thresholded Heatmap           |Final Box            |
|------------------|------------------------------|---------------------|
|![Heatmap][image8]|![Thresholded Heatmap][image9]|![Final Box][image10]|

## Pipeline
I have built a `pipeline()` function that combines all the work so far. This function detects a car by inputting a single image and returns an image showing the position of the car as a box. I tested with images in the `test_images` directory.

![pipeline test][image11]
![pipeline test][image12]
![pipeline test][image13]
![pipeline test][image14]
![pipeline test][image15]
![pipeline test][image16]

I confirmed that it works well and applied it to video. This completed video can be found [here](./project_video_output.mp4).

## Discussion