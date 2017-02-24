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

## Hog Sub-sampling Window Search

## False Positive

## Pipeline

## Discussion