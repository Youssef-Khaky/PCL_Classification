PCL Classification Project
===================================

In this project I am using an RGBD camera to collect point cloud data of items in the the environemnt, then classify the items using an SVM.  
This is a Udacity project in the Udacity Robotics Nanodegree.

Libraries and frameworks used
--------------

- ROS
- PCL

Here are the steps in this project
---------------

1. Summon each item 100 times at random orientations and extract the normal vector features and colour features. Fit these features in a hustogram.
2. Train an SVM on the histogram for each object
3. Collect point clout data of environemnt that has the objects
4. Perform the following operations on the point cloud data  
  a. Statistical Outlier Filtering  
  b. Voxel grid downsampling  
  c. Passthrough filter  
  d. RANSAC plane segmentation  
  e. Euclidian Clustering  
  f. Extract histogram  
  g. Identification cluster based on histogram  
  h. Extract the location of the items and write them to a YAML file  

UDacity Project can be found here  
https://github.com/udacity/RoboND-Perception-Project
