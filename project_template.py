#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    #print("scene is of type " , type(test_scene_num.data), "and value ". test_scene_num.data)
    #print("scene is of type " , type(test_scene_num.data), "and value ". arm_name.data)
    #print("scene is of type " , type(test_scene_num.data), "and value ". object_name.data)
    #print("scene is of type " , type(test_scene_num.data), "and value ". test_scene_num.data)
    #print("scene is of type " , type(test_scene_num.data), "and value ". test_scene_num.data)


    yaml_dict["test_scene_num"] = int(test_scene_num.data)
    yaml_dict["arm_name"]  = str(arm_name.data)
    yaml_dict["object_name"] = str(object_name.data)
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    print(type(data_dict))
    print(dict_list)
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    print("started callback")
    #pcl_pub.publish(pcl_msg)
# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_cloud=ros_to_pcl(pcl_msg)
   # print(pcl_cloud)
    
    
    # TODO: Statistical Outlier Filtering
    # Much like the previous filters, we start by creating a filter object: 
    outlier_filter = pcl_cloud.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(100)

    # Set threshold scale factor
    x = 1.0

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter() 

    # TODO: Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()


    # TODO: PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()
    
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.60
    axis_max = 0.80
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    
    passthroughx = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthroughx.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthroughx.set_filter_limits(axis_min, axis_max)
    
    cloud_filtered = passthroughx.filter()

    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance 
    # for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    Table = cloud_filtered.extract(inliers, negative=False)
    # extracted_outliers... objects
    extracted_outliers = cloud_filtered.extract(inliers, negative=True) ## the objects
    
    
    
    print("done filtering")
    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)# Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(40)
    ec.set_MaxClusterSize(1000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    
    
    # TODO: Convert PCL data to ROS messages Publish ROS messages
   # print(cluster_cloud)
    ros_cloud_objects = pcl_to_ros(extracted_outliers)
    pcl_objects_pub.publish(ros_cloud_objects)
    pc = pcl_to_ros(cluster_cloud)
    pcl_pub.publish(pc)
    print("done clustering")
# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = extracted_outliers.extract(pts_list)
 
        # TODO: convert the cluster from pcl to ROS using helper function
        pcl_ROScluster=pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # TODO: complete this step just as is covered in capture_features.py


        # Extract histogram features
        chists = compute_color_histograms(pcl_ROScluster, using_hsv=True)
        normals = get_normals(pcl_ROScluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Compute the associated feature vector

        # Make the prediction
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = pcl_ROScluster
        detected_objects.append(do)
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))


    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)
    


    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

    # function to load parameters and request PickPlace service
    
    
def pr2_mover(object_list):

    # TODO: Initialize variables
    dict_list = []
    test_scene_num = Int32()
   # print(type(test_scene_num))
    test_scene_num.data = 1
   # print(type(test_scene_num.data))
    
    
    
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    objects_list_yaml = rospy.get_param('/object_list')
    
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    for item in object_list:
        labels.append(item.label)
        points_arr = ros_to_pcl(item.cloud).to_array()
        cn = np.mean(points_arr, axis=0)[:3]
        x = cn[0]
        y = cn[1]
        z = cn[2]
        
        cn = (x,y,z)
        
        #print(cn)
        centroids.append(cn)
       # print(centroids)
        
        
        
    
    
    for i in range (len(objects_list_yaml)):
        object_name.data = objects_list_yaml[i]['name']
        print("type(object_name)",type(object_name))
        
        destination = objects_list_yaml[i]['group']
        
        if destination == 'green':
            arm_name.data = 'right'
            place_pose.position.x = 0
            place_pose.position.y = 0.71
            place_pose.position.z = 0.605
            
        else:
            arm_name.data = 'left'
            place_pose.position.x = 0
            place_pose.position.y = -0.71
            place_pose.position.z = 0.605
        print("type(place_pose.position.x)",type(place_pose.position.x))
        
        for j in range (len(labels)):
            if labels[j] == object_name.data:
                pick_pose.position.x = np.asscalar(centroids[j][0])
                print("type(pick_pose.position.x)",type(pick_pose.position.x))
                
                pick_pose.position.y = np.asscalar(centroids[j][1])
                pick_pose.position.z = np.asscalar(centroids[j][2])
                
        #print(type(test_scene_num))
        yaml_list = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_list)
        send_to_yaml('output_3.yaml',dict_list)
        
if __name__ == '__main__':


    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)
    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_pub = rospy.Publisher("/pr2/world/all1", PointCloud2, queue_size=1) #detected objects monocolour
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1) #the detected objects 
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1) #labels?
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
