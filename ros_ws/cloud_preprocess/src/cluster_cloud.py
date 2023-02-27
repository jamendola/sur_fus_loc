#!/usr/bin/env python2

import sys
import os

os.environ["OPENBLAS_CORETYPE"] = "nehalem"
import math
import rospy
import numpy as np
import sys
import ros_numpy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from cluster_methods import cluster_dbscan, preprocess_cloud, voxelize, pcl_dbscan, remove_all_horizontal_planes, octree_change_detector
import tf2_ros
from cloud_preprocess.msg import CloudCluster
import datetime
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import time

first_frame = True

tracks = dict()
last_centroids = list()
last_points = np.empty(shape=(0,3))
last_pose_transform = None
tf_buffer = None
time_list = list()


def send_cloud(data, colors=None, output_frame='map'):
    cloud_type = [
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('r', np.float32),
        ('g', np.float32),
        ('b', np.float32)]
    if colors is None:
        colors = np.tile(np.array([0.0, 0.0, 0.0]), (data.shape[0], 1))
    # print('Debugging send cloud ', colors.shape, data.shape)
    complete_data = np.column_stack((data, colors))
    cloud_arr = np.array([tuple(i) for i in complete_data], dtype=cloud_type)
    cloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(cloud_arr, frame_id=output_frame)
    cloud_msg.header.frame_id = 'map'
    filter_cloud_pub.publish(cloud_msg)


def send_cluster(centroids, cluster_sizes, header=None):
    msg = CloudCluster()
    if header is not None:
        msg.header = header
    msg.sizes = cluster_sizes
    point_list = list()
    for centroid in centroids:
        point = ros_numpy.msgify(Point, centroid)
        point_list.append(point)
    msg.centroids = point_list
    # print('Debug cluster msg output ', msg)
    cluster_pub.publish(msg)



def send_converted_cloud(cld, output_frame='kitti360_cam_00'):
    # try:
    transformation_msg = tf_buffer.lookup_transform('kitti360_cam_00',
                                                'map', rospy.Time(0), rospy.Duration(0.1))
    # except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
    #         tf2_ros.ExtrapolationException):
    #     rospy.logerr('Unable to find the transformation')
    # tf_matrix = ros_numpy.numpify(transformation_msg.transform)
    cld = do_transform_cloud(cld, transformation_msg)
    # print('debug tf2_ros',  transformation_msg, tf_matrix)

    pts = preprocess_cloud(cld)
    send_cloud(pts, output_frame=output_frame)


def receive_cloud(cld):
    print('RECEIVING CLOUD', cld.header, cld.header.seq)
    global first_frame
    global last_centroids
    global last_points
    global last_pose_transform
    global tf_buffer
    global time_list
    # start_time = datetime.datetime.now()
    # acc_pts = np.vstack((pts, last_points))


    # get the tf at first available time
    try:

        transformation_msg = tf_buffer.lookup_transform('map',
                                                    'kitti360_velodyne', rospy.Duration(0), rospy.Duration(0.1))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException):
        print('Failed to get transform')
        rospy.logerr('Unable to find the transformation')
    # tf_matrix = ros_numpy.numpify(transformation_msg.transform)
    org_pts = preprocess_cloud(cld)
    cld = do_transform_cloud(cld, transformation_msg)
    # print('debug tf2_ros',  transformation_msg, tf_matrix)
    start_time = datetime.datetime.now()
    pts = preprocess_cloud(cld)
    pts = pts[(org_pts[:, 2] > -0.7) & (org_pts[:, 2] < -0.2) & (org_pts[:, 1] < 4.0) & (org_pts[:, 1] > -3.0) \
              & ~((org_pts[:, 0] > -10.0) & (org_pts[:, 0] < 2.0) & (org_pts[:, 1] > -2.0) & (org_pts[:, 1] < 2.0))]
    # send_converted_cloud(cld, output_frame='kitti360_cam_00')
    # points = np.hstack((pts, np.ones((pts.shape[0], 1))))
    # shifted_pts = np.einsum('ij,kj->ki', points, tf_matrix)
    # shifted_pts = shifted_pts.T[:, :3].astype(np.float32)
    # print('debug shifted_pts', shifted_pts.shape, shifted_pts.dtype)
    # print('debug pts', pts.shape, pts[:,0])

    # print('debug pts b', pts.shape)
    print('single callback Cloud header : ', cld.header)
    if first_frame:
        print(' First frame ')
        simplified_pts = voxelize(pts)
        cluster_eps = 1.0
        cluster_min_pts = 2
    else:
        cluster_eps = 0.2
        cluster_min_pts = 5
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()
    print('time spend on PREPROCESS and voxel filter ', execution_time)

    start_time = datetime.datetime.now()
    if not first_frame:
        diff_pts = octree_change_detector(last_points, pts)
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time_b = time_diff.total_seconds()
    print('time spend on octree change ', execution_time_b)
    time_list.append((execution_time+ execution_time_b))
    if not first_frame:
        pts_to_send = diff_pts
    else:
        pts_to_send = simplified_pts

    # send_cloud(pts_to_send)

    if pts_to_send.shape[0] > 0:
        print('start clustering of n points', pts_to_send.shape[0])
        label_colors, centroids, cluster_sizes = cluster_dbscan(pts_to_send, eps=cluster_eps, min=cluster_min_pts)
        print('DEBUG CENTROID COORDS', centroids)
        start_time = time.time()
        send_track_markers(centroids, color=(0,200,0), sizes=cluster_sizes)
        print('DEBUG sending clusters with headerseq', cld.header.seq)
        send_cluster(centroids, cluster_sizes, header=cld.header)
        send_cloud(pts_to_send, label_colors)


    # moving = filter_moving_centroids(centroids)
    # send_track_markers(moving)
    first_frame = False
    last_points = pts
    last_pose_transform = transformation_msg


def filter_moving_centroids(centroids):
    global last_centroids
    if len(centroids) > 0:
        centroids = np.vstack(centroids)
    if first_frame:
        moving = centroids
    else:
        tree = KDTree(last_centroids[:, :])
        distances, indices = tree.query(centroids[:, :], 1)
        distances = np.squeeze(distances)
        boolean = np.logical_and(distances > 0.2, distances < 1.0)
        moving = centroids[boolean, :]
    last_centroids = centroids
    return moving


def send_track_markers(moving, color=(200,0,0), sizes=None):
    msg = MarkerArray()
    del_marker = Marker()
    del_marker.header.frame_id = 'map'
    del_marker.id = 0
    del_marker.type = 0
    del_marker.action = 3
    del_marker.scale.x = 0.1
    del_marker.scale.y = 0.1
    del_marker.scale.z = 0.1
    msg.markers.append(del_marker)
    marker_pub.publish(msg)
    msg = MarkerArray()
    for i,mov_track in enumerate(moving):
        obj_marker = Marker()
        obj_marker.type = 2
        obj_marker.id = i
        obj_marker.header.frame_id = 'map'
        obj_marker.color.r = color[0]
        obj_marker.color.g = color[1]
        obj_marker.color.b = color[2]
        obj_marker.color.a = 1.0
        if sizes is None:
            size = 10
        else:
            size = sizes[i]
        obj_marker.scale.x = 0.01*size
        obj_marker.scale.y = 0.01*size
        obj_marker.scale.z = 0.01*size
        obj_marker.action = 0
        obj_marker.pose.position.x = mov_track[0]
        obj_marker.pose.position.y = mov_track[1]
        obj_marker.pose.position.z = mov_track[2]
        msg.markers.append(obj_marker)
    marker_pub.publish(msg)


def detect_changed_tracks(centroids, sizes):
    #Assuming there is not splia and merge and also different clusters do not get too close to each other
    # to steal the track
    global tracks
    moving = list()
    deleted = list()
    new = list()
    if first_frame:
        for idx, centroid in enumerate(centroids):
            tracks[idx] = [centroid, sizes[idx]]
            new.append(idx)
    else:
        for track_id, data_list in tracks.items():
            track_point = data_list[0]
            min_dist = 100000000000
            closest_centroid_id = -1
            association_list = list()
            for idx, centroid in enumerate(centroids):
                dist = np.linalg.norm(track_point - centroid)
                if dist < min_dist and idx not in association_list:
                    min_dist = dist
                    closest_centroid_id = idx
            if closest_centroid_id == -1:
                deleted.append(track_id)
            else:
                tracks[track_id] = [centroids[closest_centroid_id], sizes[closest_centroid_id]]
                if min_dist < 0.2 and min_dist > 0.1 :
                    moving.append(track_id)

    return moving, deleted, new

def tf_callback(data):
    # Retrieve the position and orientation information from the message
    position = data.transform.translation
    orientation = data.transform.rotation

    # convert the quaternion to Euler angles
    quat = (orientation.x, orientation.y, orientation.z, orientation.w)
    euler = tf.transformations.euler_from_quaternion(quat)

    # Print the position and orientation information
    print("Position: ", position)
    print("Orientation: ", euler)
    print("Displacement: ", (position.x, position.y, position.z))


def shutdown_cbk():
    global time_list
    avg = np.mean(time_list)
    std = np.std(time_list)
    print('### CLOUD PROCESSING TIME METRICS ###', avg, std)



if __name__ == '__main__':
    try:
        print("Start node")
        rospy.init_node('cloud_cluster')
        # raw_image_var = "/cn/camera/image_raw"
        node_name = rospy.get_name()
        cloud_name = rospy.resolve_name('/kitti360/cloud')
        print('Waiting for point cloud data from ' + cloud_name)

        cloud_sub = rospy.Subscriber(cloud_name, PointCloud2, receive_cloud)

        # tr_listener = tf2_ros.TransformListener(tfBuffer)
        # pos_sub = rospy.Subscriber("/tf", TransformStamped, tf_callback)

        tf_buffer = tf2_ros.Buffer(rospy.Duration(0.5))
        tf2_ros.TransformListener(tf_buffer)

        cluster_pub_name = rospy.resolve_name(node_name + '/clusters')
        cluster_pub = rospy.Publisher(cluster_pub_name, CloudCluster, queue_size=1)

        filter_cloud_name = rospy.resolve_name(node_name + '/filtered_cloud')

        filter_cloud_pub = rospy.Publisher(filter_cloud_name, PointCloud2, queue_size=1)



        marker_name = rospy.resolve_name(node_name + '/debug_markers')

        marker_pub = rospy.Publisher(marker_name, MarkerArray, queue_size=1)
        print("its publishing")
        rospy.on_shutdown(shutdown_cbk)
        # test = transform.lookupTransform('os1_lidar', 'os1_sensor',rospy.Time(0))
        rospy.spin()

    except rospy.ROSInterruptException:
        pass