#!/usr/bin/env python
import sys
import os

os.environ["OPENBLAS_CORETYPE"] = "nehalem"
import math
import rospy
import numpy as np
import sys
import cv2
import ros_numpy
import cv_bridge
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from vision_msgs.msg import VisionInfo, Detection2DArray, Detection2D, ObjectHypothesisWithPose
import message_filters
from overlay_utils import preprocess_cloud, convert_cloud_to_pixel_space, get_lidar_mask_overlay, \
    return_minimum_distance_from_origin, show_image_overlay, get_cloud_color_list, get_lidar_bbox_overlay_workaround, \
    get_label_name, get_distance, get_highest_score_id, get_closest_point_to_sensor
from calibration import lidar_2_cam_north, camera_matrix_north, dist_coeffs_north, new_lidar_2_cam_north, \
    new_camera_matrix_north, new_dist_coeffs_north, new_lidar_2_cam_east, new_lidar_2_cam_west, new_lidar_2_cam_south
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import time
from cloud_preprocess.msg import CloudCluster
import scipy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import pandas as pd
import datetime


bridge = cv_bridge.CvBridge()
image_overlay_flag = False
filter_cloud_flag = False
# color_filter_cloud_flag = False

last_det = None
# lidar_cam_transf = new_lidar_2_cam_north

tracks = dict()  # tuple with point, label, associated_flag, overlay_type(0:no update, 1:overlay, 2:purecloud)
last_overlaid_cloud_seq = -1

overlay_threshold = 0.6
pure_cloud_threshold = 0.6

left_clusters = list()
just_created_tracks = dict()

save_report = True
report_df = pd.DataFrame(columns=['seq', 'id', 'label', 'x', 'y', 'overlay_type'])
report_df['seq'] = report_df['seq'].astype(int)

def update_track(track_id, point, overlay_type):
    global tracks
    print('Updating track ', track_id)
    tracks[track_id][1] = point
    tracks[track_id][2] = True
    tracks[track_id][3] = overlay_type
    pose = PoseStamped()
    pose.pose.position.x = point[0]
    pose.pose.position.y = point[1]
    pose.pose.position.z = 0
    tracks[track_id][4].poses.append(pose)


def create_new_track(label, point):
    global tracks
    global just_created_tracks
    new_track_id = len(tracks.keys()) + len(just_created_tracks.keys())
    path = Path()
    just_created_tracks[new_track_id] = [label, point, False, 1, path]
    print('Creating track ', new_track_id, label)


def try_assign_overlay_to_tracks(point, label):
    global tracks
    assigned_track = -1
    min_dist = overlay_threshold
    for id, data in tracks.items():
        if label == data[0] and data[2] == False:
            dist = np.linalg.norm(data[1][:-1] - point[:-1])
            if dist < min_dist:
                assigned_track = id
                min_dist = dist
    print('Debug overlay assign with dist ', min_dist)
    return assigned_track


def try_assign_to_tracks(point):
    global tracks
    assigned_track = -1
    min_dist = pure_cloud_threshold
    for id, data in tracks.items():
        if data[2] == False:
            dist = np.linalg.norm(data[1][:-1] - point[:-1])
            if dist < min_dist:
                assigned_track = id
                min_dist = dist
    print('Debug assign with dist ', min_dist)
    return assigned_track


def fusion(det, clusters, args):
    global left_clusters
    # print('Entering fusion callback for region ', args[1])
    # print('seq numbers ', clusters.header.seq, det.header.seq)
    start_time = datetime.datetime.now()
    if len(left_clusters) > 0:
        lidar_cam_transf = args[0]
        for detection in det.detections:
            if len(left_clusters) > 0:
                pts = np.array(left_clusters)
                results = detection.results
                label, score = get_highest_score_id(results)
                # label = get_label_name(dataset_name, id)
                if label in [62, 72]:  # labels of interest
                    # print(' detection w label ', label, ' # clusters still left ', len(left_clusters))
                    pixels_from_lidar, cam_pts = convert_cloud_to_pixel_space(lidar_cam_transf, new_camera_matrix_north,
                                                                              new_dist_coeffs_north, pts)
                    indexes = get_lidar_bbox_overlay_workaround(pixels_from_lidar, cam_pts,
                                                                detection.bbox)
                    if indexes.shape[0] > 0:
                        # print('Debug ', indexes, left_clusters)
                        selected_clusters = pts[indexes]
                        # for idx in indexes:
                        #     left_clusters.pop(idx)
                        # print('Overlayed clusters', len(selected_clusters), ' label ', label)
                        idx, pt = get_closest_point_to_sensor(selected_clusters)
                        assigned_track = try_assign_overlay_to_tracks(pt, label)
                        if assigned_track == -1:
                            create_new_track(label, pt)
                        else:
                            print('Assigning overlay')
                            update_track(assigned_track, pt, 1)
                        print('debug overlay before pop ', len(left_clusters))
                        left_clusters.pop(indexes[idx])
                        print('debug overlay after pop ', len(left_clusters))
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()
    # print('time spend on fusion ', execution_time)

def pure_cloud(clusters):
    start_time = datetime.datetime.now()
    # print('Entering pure cloud callback ', clusters.header)
    # print('# left clusters for pure cloud assoc attempt ', len(left_clusters))
    global left_clusters
    global tracks
    global just_created_tracks
    for cluster in left_clusters:
        assigned_track = try_assign_to_tracks(cluster)
        if assigned_track != -1:
            print('Assigning point from pure cloud')
            update_track(assigned_track, cluster, 2)
    left_clusters = list()
    # Reset update flags of tracks to False
    # tracks with just created_tracks
    tracks.update(just_created_tracks)
    just_created_tracks.clear()

    # print('tracks', tracks)

    if save_report:
        save_track_states(int(clusters.header.seq)-1, tracks)
    for track_id, data in tracks.items():
        data[2] = False
        data[3] = 0

    for centroid in clusters.centroids:
        pt = ros_numpy.numpify(centroid)
        left_clusters.append(pt)
    # print('# New clusters ', len(left_clusters))
    # print('# tracks', tracks)

    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()
    # print('time spend on pure cloud ', execution_time)

    send_track_markers(tracks)
    update_path(tracks)

def save_track_states(seq, tracks):
    global report_df
    for id, data in tracks.items():
        new_row = pd.Series(data={'seq': seq, 'id': id, 'label': data[0], 'x': data[1][0], 'y': data[1][1], 'overlay_type':data[3]})
        report_df = report_df.append(new_row, ignore_index=True)


def send_debug_markers(moving, color=(200, 0, 0), sizes=None):
    msg = MarkerArray()
    del_marker = Marker()
    del_marker.header.frame_id = 'os_sensor'
    del_marker.id = 0
    del_marker.type = 0
    del_marker.action = 3
    del_marker.scale.x = 0.1
    del_marker.scale.y = 0.1
    del_marker.scale.z = 0.1
    msg.markers.append(del_marker)
    out_pub.publish(msg)
    msg = MarkerArray()
    for i, mov_track in enumerate(moving):
        obj_marker = Marker()
        obj_marker.type = 2
        obj_marker.id = i
        obj_marker.header.frame_id = 'os_sensor'
        obj_marker.color.r = color[0]
        obj_marker.color.g = color[1]
        obj_marker.color.b = color[2]
        obj_marker.color.a = 1.0
        if sizes is None:
            size = 10
        else:
            size = sizes[i]
        obj_marker.scale.x = 0.01 * size
        obj_marker.scale.y = 0.01 * size
        obj_marker.scale.z = 0.01 * size
        obj_marker.action = 0
        obj_marker.pose.position.x = mov_track[0]
        obj_marker.pose.position.y = mov_track[1]
        obj_marker.pose.position.z = mov_track[2]
        msg.markers.append(obj_marker)
    out_pub.publish(msg)


def send_track_markers(track_dict):
    msg = MarkerArray()
    for track_id, data in track_dict.items():
        # if overlay_flags is None:
        #     a = 1.0
        # else:
        #     if overlay_flags[i]:
        #         a = 1.0
        #     else:
        #         a = 0.8
        obj_marker = Marker()
        obj_marker.type = 9
        obj_marker.id = track_id
        obj_marker.header.frame_id = 'os_sensor'
        obj_marker.color.g = int(30 * track_id)
        obj_marker.color.b = int(200 - 20 * track_id)
        obj_marker.color.r = int(20*track_id)
        obj_marker.color.a = 1.0
        obj_marker.scale.x = 0.15
        obj_marker.scale.y = 0.15
        obj_marker.scale.z = 0.15
        obj_marker.pose.position.x = data[1][0]
        obj_marker.pose.position.y = data[1][1]
        obj_marker.pose.position.z = 0
        dist = np.linalg.norm(data[1][:-1])
        obj_marker.text = 'Track:' + str(track_id) + '\nLabel:' + str(data[0]) + '\ndist:' + str(round(dist, 2))
        # print('debug pub marker ', obj_marker.text)
        msg.markers.append(obj_marker)
    out_pub.publish(msg)


def update_path(tracks):
    for track_id, data in tracks.items():
        data[4].header.frame_id = 'os_sensor'
        if track_id < len(path_pub_list):
            path_pub_list[track_id].publish(data[4])


def send_cloud(data, colors=None):
    cloud_type = [
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('r', np.float32),
        ('g', np.float32),
        ('b', np.float32)]
    if colors is None:
        colors = np.tile(np.array([0.0, 0.0, 0.0]), (data.shape[0], 1))
    print('Debugging send cloud ', colors.shape, data.shape)
    complete_data = np.column_stack((data, colors))
    cloud_arr = np.array([tuple(i) for i in complete_data], dtype=cloud_type)
    cloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(cloud_arr, frame_id='os_sensor')
    filter_cloud_pub.publish(cloud_msg)


def receive_cluster(cl):
    print('cluster header : ', cl.header)
    pass


def receive_detection(det):
    print('Detection header : ', det.header)
    pass


def shutdown():
    print('Node interrrupted')
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    report_df.to_csv('/home/jose/surround_early_fusion/surround_view_ws/src/overlay/report_'+current_time+'.csv')
    print('wrote file')


if __name__ == '__main__':
    try:
        rospy.init_node('overlay')
        node_name = rospy.get_name()
        # raw_image_var = rospy.get_param(node_name + '/raw_image')
        # detection_var = rospy.get_param(node_name + '/det')
        # raw_image_var = "/cn/camera/image_raw"
        cluster_var = '/cloud_cluster/clusters'

        cluster_sub = message_filters.Subscriber('/cloud_cluster/clusters', CloudCluster)
        cluster_sub.registerCallback(pure_cloud)
        # cluster_sub.registerCallback(receive_cluster)
        # Initialize the node and name it.

        # rate = rospy.Rate(10.0)

        detect_sub_n = message_filters.Subscriber('/det_n/detections', Detection2DArray)
        ts_n = message_filters.ApproximateTimeSynchronizer([detect_sub_n, cluster_sub], 1, 3,
                                                           allow_headerless=True)
        ts_n.registerCallback(fusion, (new_lidar_2_cam_north, 'n'))

        detect_sub_e = message_filters.Subscriber('/det_e/detections', Detection2DArray)
        ts_e = message_filters.ApproximateTimeSynchronizer([detect_sub_e, cluster_sub], 1, 3,
                                                           allow_headerless=True)
        ts_e.registerCallback(fusion, (new_lidar_2_cam_east, 'e'))

        detect_sub_s = message_filters.Subscriber('/det_s/detections', Detection2DArray)
        ts_s = message_filters.ApproximateTimeSynchronizer([detect_sub_s, cluster_sub], 1, 3,
                                                           allow_headerless=True)
        ts_s.registerCallback(fusion, (new_lidar_2_cam_south, 's'))
        # detect_sub_s.registerCallback(receive_detection)

        detect_sub_w = message_filters.Subscriber('/det_w/detections', Detection2DArray)
        ts_w = message_filters.ApproximateTimeSynchronizer([detect_sub_w, cluster_sub], 1, 3,
                                                           allow_headerless=True)
        ts_w.registerCallback(fusion, (new_lidar_2_cam_west, 'w'))

        out_name = rospy.resolve_name(node_name + '/object')
        # out_pub = rospy.Publisher(out_name, Marker, queue_size=1)
        out_pub = rospy.Publisher(out_name, MarkerArray, queue_size=1)

        filter_cloud_name = rospy.resolve_name(node_name + '/filtered_cloud')

        # Publish the lidar overlay image
        filter_cloud_pub = rospy.Publisher(filter_cloud_name, PointCloud2, queue_size=1)

        path_pub_list = list()
        for i in range(4):
            name = rospy.resolve_name(node_name + '/track_path_' + str(i))

            # Publish the lidar overlay image
            pub = rospy.Publisher(name, Path, queue_size=1)
            path_pub_list.append(pub)
        print("its publishing")
        # test = transform.lookupTransform('os1_lidar', 'os1_sensor',rospy.Time(0))

        rospy.on_shutdown(shutdown)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass


