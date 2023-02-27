#!/usr/bin/env python2

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
    get_label_name, get_distance, get_highest_score_id, get_closest_point_to_sensor, get_lidar_bbox_overlay
from calibration import lidar_2_cam_north, camera_matrix_north, dist_coeffs_north, new_lidar_2_cam_north, \
    new_camera_matrix_north, new_dist_coeffs_north, new_lidar_2_cam_east, new_lidar_2_cam_west, new_lidar_2_cam_south
import tf2_ros
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf2_geometry_msgs
import tf
import time
from cloud_preprocess.msg import CloudCluster
import scipy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import pandas as pd
import datetime
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

cam0_matrix = np.array([[552.554261, 0.000000, 682.049453],
                        [0.000000, 552.554261, 238.769549],
                        [0.000000, 0.000000, 1.000000]], dtype=np.float64)

bridge = cv_bridge.CvBridge()
image_overlay_flag = False
filter_cloud_flag = False
# color_filter_cloud_flag = False

# last_det = None
# lidar_cam_transf = new_lidar_2_cam_north

tracks = dict()  # tuple with point, label, associated_flag, overlay_type(0:no update, 1:overlay, 2:purecloud)
last_overlaid_cloud_seq = -1

overlay_threshold = 5.0
pure_cloud_threshold = 5.0

left_clusters = list()
just_created_tracks = dict()

tf_buffer = None
global reset
global frame, last_det_msg, last_det_frame, last_clusters_frame, last_clusters_msg, det_dict
global inner_count
time_list = list()

inner_count = 0
reset = False
frame = -1
last_clusters_frame = -1
last_det_frame = -1
last_det_frame = -1
last_det_msg = None
last_clusters_msg = None
det_dict = dict()
save_report = True
report_df = pd.DataFrame(columns=['seq', 'id', 'label', 'x', 'y', 'z', 'overlay_type', 'exec_time'])
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
    pose.pose.position.z = point[2]
    tracks[0][4].poses.append(pose)


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
            print('######DIST TRYING TO ASSIGN OVERLAY', dist)
    return assigned_track


def try_assign_to_tracks(point):
    global tracks
    assigned_track = -1
    min_dist = pure_cloud_threshold
    for id, data in tracks.items():
        if data[2] == False:
            # dist = np.linalg.norm(data[1][:-1] - point[:-1])
            dist = np.linalg.norm(data[1] - point)
            if dist < min_dist:
                assigned_track = id
                min_dist = dist
            # print('Debug assign with dist ', min_dist)
            print('######DIST TRYING TO ASSIGN PURE CLOUD', dist)
    return assigned_track


def debug_projection(clusters):
    pts = np.array(clusters)
    if pts.shape != () and pts.shape[0] != 0:
        print('debug projection', pts.shape)
        transformation_msg = tf_buffer.lookup_transform('kitti360_cam_00',
                                                        'map', rospy.Time(0), rospy.Duration(0.1))
        transform = transformation_msg.transform
        global_2_cam0_matrix = tf.transformations.quaternion_matrix(
            (transform.rotation.x,
             transform.rotation.y,
             transform.rotation.z,
             transform.rotation.w))
        global_2_cam0_matrix[:3, 3] = (transform.translation.x,
                                       transform.translation.y,
                                       transform.translation.z)
        cam0_distortion_coeffs = None

        pixels_from_lidar, cam_pts = convert_cloud_to_pixel_space(global_2_cam0_matrix, cam0_matrix,
                                                                  cam0_distortion_coeffs, pts)
        send_cloud(cam_pts, output_frame='kitti360_cam_00')


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


def fusion(clusters):
    global time_list

    start_time = datetime.datetime.now()
    # print('Match of messages')
    global frame, left_clusters, last_det_frame, last_det_msg, last_clusters_msg, last_clusters_frame, tracks, det_dict
    print("!!!!!!SYNCHRONIZATION!!!!!", frame)
    if frame == 0:
        print("Deleting tracks")
        delete_track_markers(tracks)
        tracks = dict()

    # while last_clusters_frame < frame:
    #     time.sleep(0.01)
    # if last_clusters_frame == frame:
    if True:
        for centroid in clusters.centroids:
            pt = ros_numpy.numpify(centroid)
            left_clusters.append(pt)

        # print('debug fusion', transformation_msg)
        # print('DEBUG COORDINATE FRAME', left_clusters)

        # print('Debug inside fusion, left centroids', len(left_clusters))
        if len(left_clusters) > 0:
            if frame in det_dict.keys():
                for cam_id, det_msg in det_dict[frame]:

                    transformation_msg = tf_buffer.lookup_transform('kitti360_cam_' + cam_id,
                                                                    'map', rospy.Time(0), rospy.Duration(0.1))
                    for detection in det_msg.detections:
                        if len(left_clusters) > 0:
                            pts = np.array(left_clusters)
                            results = detection.results
                            label, score = get_highest_score_id(results)
                            # label = get_label_name(dataset_name, id)
                            # print('debug fusion', label, transformation_msg)
                            if label in [3, 6, 7, 8]:  # labels of interest #3-car
                                print('MATCHED DETECTION WITH PROPER LABEL', cam_id, det_msg.header.stamp.to_sec(),
                                      clusters.header.stamp.to_sec())
                                transform = transformation_msg.transform
                                global_2_cam0_matrix = tf.transformations.quaternion_matrix(
                                    (transform.rotation.x,
                                     transform.rotation.y,
                                     transform.rotation.z,
                                     transform.rotation.w))
                                global_2_cam0_matrix[:3, 3] = (transform.translation.x,
                                                               transform.translation.y,
                                                               transform.translation.z)
                                cam0_distortion_coeffs = None

                                pixels_from_lidar, cam_pts = convert_cloud_to_pixel_space(global_2_cam0_matrix,
                                                                                          cam0_matrix,
                                                                                          cam0_distortion_coeffs, pts)
                                print('PIXELS FROM LIDAR', pixels_from_lidar)
                                print('BBOX', detection.bbox)
                                indexes = get_lidar_bbox_overlay(pixels_from_lidar, cam_pts,
                                                                 detection.bbox)
                                print('OVERLAY', indexes)
                                if indexes.shape[0] > 0:
                                    # print('Debug ', indexes, left_clusters)
                                    selected_clusters = pts[indexes]
                                    # send_debug_markers(selected_clusters)
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
                                    # print('debug overlay before pop ', len(left_clusters))
                                    left_clusters.pop(indexes[idx])
                                    # print('debug overlay after pop ', len(left_clusters))
            for cluster in left_clusters:
                assigned_track = try_assign_to_tracks(cluster)
                if assigned_track != -1:
                    print('Assigning point from pure cloud')
                    send_debug_markers([cluster], color=(100, 100, 0))
                    update_track(assigned_track, cluster, 2)

            left_clusters = list()
            # Reset update flags of tracks to False
            # tracks with just created_tracks
            tracks.update(just_created_tracks)

    send_track_markers(tracks)
    print('DEBUG TRACKS', tracks)
    just_created_tracks.clear()

    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()
    time_list.append(execution_time)
    print('Saving tracks for frame', frame)
    save_track_states(frame, tracks, execution_time)
    for track_id, data in tracks.items():
        data[2] = False
        data[3] = 0
    allow_clock.publish("Ready")
    print('Ready')
    update_path(tracks)
    # print('time spend on fusion ', execution_time)


def save_track_states(seq, tracks, exec_time=0):
    global report_df
    for id, data in tracks.items():
        print('REPORT DEBUG', seq)
        new_row = pd.Series(
            data={'seq': seq, 'id': id, 'label': data[0], 'x': data[1][0], 'y': data[1][1], 'z': data[1][1],
                  'overlay_type': data[3], 'exec_time': exec_time})
        report_df = report_df.append(new_row, ignore_index=True)


def send_debug_markers(moving, color=(200, 0, 0), sizes=None):
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
    out_pub.publish(msg)
    print("Sending debug markers")
    msg = MarkerArray()
    for i, mov_track in enumerate(moving):
        obj_marker = Marker()
        obj_marker.type = 2
        obj_marker.id = i
        obj_marker.header.frame_id = 'map'
        obj_marker.color.r = color[0]
        obj_marker.color.g = color[1]
        obj_marker.color.b = color[2]
        obj_marker.color.a = 1.0
        if sizes is None:
            size = 100
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


def delete_track_markers(track_dict):
    msg = MarkerArray()
    for track_id, data in track_dict.items():
        msg = MarkerArray()
        del_marker = Marker()
        del_marker.header.frame_id = 'map'
        del_marker.id = track_id
        del_marker.action = 3
        msg.markers.append(del_marker)
        out_pub.publish(msg)
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
        obj_marker.type = 2
        # print("Sending tracks on frame", frame)
        # obj_marker.ns = str(frame)
        obj_marker.id = track_id
        obj_marker.header.frame_id = 'map'
        obj_marker.color.g = int(30 * track_id)
        obj_marker.color.b = int(200 - 20 * track_id)
        obj_marker.color.r = int(20 * track_id)
        obj_marker.color.a = 1
        obj_marker.scale.x = 1
        obj_marker.scale.y = 1
        obj_marker.scale.z = 1
        obj_marker.action = obj_marker.ADD
        obj_marker.pose.orientation.x = 0
        obj_marker.pose.orientation.y = 0
        obj_marker.pose.orientation.z = 0
        obj_marker.pose.orientation.w = 1
        obj_marker.pose.position.x = data[1][0]
        obj_marker.pose.position.y = data[1][1]
        obj_marker.pose.position.z = data[1][2]
        dist = np.linalg.norm(data[1][:-1])
        # obj_marker.text = 'Track:' + str(track_id) + '\nLabel:' + str(data[0]) + '\ndist:' + str(round(dist, 2))
        # print('debug pub marker ', obj_marker.text)
        msg.markers.append(obj_marker)
    out_pub.publish(msg)


def update_path(tracks):
    for track_id, data in tracks.items():
        data[4].header.frame_id = 'map'
        if track_id < len(path_pub_list):
            path_pub_list[track_id].publish(data[4])


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
    print('Debugging send cloud ', colors.shape, data.shape)
    complete_data = np.column_stack((data, colors))
    cloud_arr = np.array([tuple(i) for i in complete_data], dtype=cloud_type)
    cloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(cloud_arr, frame_id=output_frame)
    filter_cloud_pub.publish(cloud_msg)


def receive_cluster(cl):
    global frame, last_clusters_msg, last_clusters_frame
    last_clusters_msg = cl
    last_clusters_frame = frame
    print('cluster header : ', cl.header, last_clusters_frame)


def receive_cloud(cld):
    global time_list

    start_time = datetime.datetime.now()
    # print('Match of messages')
    global frame, left_clusters, last_det_frame, last_det_msg, last_clusters_msg, last_clusters_frame, tracks, det_dict
    print("!!!!!!BASELINE SYNCHRONIZATION!!!!!", frame)
    if frame == 0:
        print("Deleting tracks")
        delete_track_markers(tracks)
        tracks = dict()

    # while last_clusters_frame < frame:
    #     time.sleep(0.01)
    # if last_clusters_frame == frame:
    if True:

        transformation_msg = tf_buffer.lookup_transform('map',
                                                        'kitti360_velodyne', rospy.Time(0), rospy.Duration(0.1))
        cld = do_transform_cloud(cld, transformation_msg)
        points = ros_numpy.point_cloud2.pointcloud2_to_array(cld)  #
        aux = points.reshape(-1)
        xyz = np.array(aux[['x', 'y', 'z']].tolist())
        xyz.setflags(write=1)

        for point in xyz:
            left_clusters.append(point)
        # print('debug fusion', transformation_msg)
        # print('DEBUG COORDINATE FRAME', left_clusters)

        transformation_msg = tf_buffer.lookup_transform('kitti360_cam_00',
                                                        'map', rospy.Time(0), rospy.Duration(0.1))

        # print('Debug inside fusion, left centroids', len(left_clusters))
        if len(left_clusters) > 0:
            if frame in det_dict.keys():
                for detection in det_dict[frame].detections:
                    if len(left_clusters) > 0:
                        pts = np.array(left_clusters)
                        results = detection.results
                        label, score = get_highest_score_id(results)
                        # label = get_label_name(dataset_name, id)
                        # print('debug fusion', label, transformation_msg)
                        if label in [3, 6, 7, 8]:  # labels of interest #3-car
                            print('MATCHED DETECTION WITH PROPER LABEL', det_dict[frame].header.stamp.to_sec(),
                                  cld.header.stamp.to_sec())
                            transform = transformation_msg.transform
                            global_2_cam0_matrix = tf.transformations.quaternion_matrix(
                                (transform.rotation.x,
                                 transform.rotation.y,
                                 transform.rotation.z,
                                 transform.rotation.w))
                            global_2_cam0_matrix[:3, 3] = (transform.translation.x,
                                                           transform.translation.y,
                                                           transform.translation.z)
                            cam0_distortion_coeffs = None

                            pixels_from_lidar, cam_pts = convert_cloud_to_pixel_space(global_2_cam0_matrix,
                                                                                      cam0_matrix,
                                                                                      cam0_distortion_coeffs, pts)
                            print('PIXELS FROM LIDAR', pixels_from_lidar)
                            print('BBOX', detection.bbox)
                            indexes = get_lidar_bbox_overlay(pixels_from_lidar, cam_pts,
                                                             detection.bbox)
                            print('OVERLAY', indexes)
                            if indexes.shape[0] > 0:
                                # print('Debug ', indexes, left_clusters)
                                selected_clusters = pts[indexes]
                                # send_debug_markers(selected_clusters)
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
                                # print('debug overlay before pop ', len(left_clusters))
                                left_clusters.pop(indexes[idx])
                                # print('debug overlay after pop ', len(left_clusters))
            for cluster in left_clusters:
                assigned_track = try_assign_to_tracks(cluster)
                if assigned_track != -1:
                    print('Assigning point from pure cloud')
                    send_debug_markers([cluster], color=(100, 100, 0))
                    update_track(assigned_track, cluster, 2)

            left_clusters = list()
            # Reset update flags of tracks to False
            # tracks with just created_tracks
            tracks.update(just_created_tracks)

    send_track_markers(tracks)
    print('DEBUG TRACKS', tracks)
    just_created_tracks.clear()

    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()
    time_list.append(execution_time)
    print('Saving tracks for frame', frame)
    save_track_states(frame, tracks, execution_time)
    for track_id, data in tracks.items():
        data[2] = False
        data[3] = 0
    allow_clock.publish("Ready")
    print('Ready')
    # print('time spend on fusion ', execution_time)


def receive_detection(det, cam_id):
    global frame, last_det_msg, last_det_frame, det_dict
    last_det_msg = det
    last_det_frame = frame
    if frame not in det_dict.keys():
        det_dict[frame] = list()
    det_dict[frame].append((cam_id, det))
    # for detection in det.detections:
    #     label, score = get_highest_score_id(detection.results)
    #     print('#### DEBUG DETECTION AT FRAME COUNT', frame, label)
    print('!!!!DETECTION : ', cam_id, last_det_frame, det.header.stamp.to_sec())


def shutdown():
    global time_list
    print('Node interrrupted')
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    report_df.to_csv(
        '/home/jose/surround_early_fusion/surround_view_ws/src/overlay/kitti_report_' + current_time + '.csv')
    avg = np.mean(time_list)
    std = np.std(time_list)
    print('### FUSION PROCESSING TIME METRICS ###', avg, std)
    print('wrote file')


def reset_cbk(data):
    global reset
    global frame
    frame = int(data.data)
    # print("Received Reset signal")
    # reset = True


if __name__ == '__main__':
    try:
        rospy.init_node('overlay')
        node_name = rospy.get_name()
        # raw_image_var = rospy.get_param(node_name + '/raw_image')
        # detection_var = rospy.get_param(node_name + '/det')
        # raw_image_var = "/cn/camera/image_raw"
        cluster_var = '/cloud_cluster/clusters'

        tf_buffer = tf2_ros.Buffer(rospy.Duration(0.5))
        tf2_ros.TransformListener(tf_buffer)

        cluster_sub = rospy.Subscriber('/cloud_cluster/clusters', CloudCluster, fusion)
        # cluster_sub.registerCallback(receive_cluster,)

        # cloud_name = rospy.resolve_name('/kitti360/cloud')
        # print('Waiting for point cloud data from ' + cloud_name)
        #
        # cloud_sub = rospy.Subscriber(cloud_name, PointCloud2, receive_cloud)

        det_sub = rospy.Subscriber('/det_00/detections', Detection2DArray, receive_detection, ('00'))
        # det_sub = rospy.Subscriber('/det_02/detections', Detection2DArray, receive_detection, ('02'))
        # cluster_sub.registerCallback(receive_detection, ('00'))

        # cluster_sub = message_filters.Subscriber('/cloud_cluster/clusters', CloudCluster)
        # cluster_sub.registerCallback(pure_cloud)
        # # cluster_sub.registerCallback(receive_cluster)
        # # Initialize the node and name it.
        #
        # # rate = rospy.Rate(10.0)
        #
        # detect_sub_n = message_filters.Subscriber('/det_00/detections', Detection2DArray)
        # ts_n = message_filters.ApproximateTimeSynchronizer([detect_sub_n, cluster_sub], queue_size=2, slop=0.3,
        #                                                    allow_headerless=True)
        # ts_n.registerCallback(fusion, ('00'))

        out_name = rospy.resolve_name(node_name + '/object')
        # out_pub = rospy.Publisher(out_name, Marker, queue_size=1)
        out_pub = rospy.Publisher(out_name, MarkerArray, queue_size=1)

        filter_cloud_name = rospy.resolve_name(node_name + '/filtered_cloud')

        # Publish the lidar overlay image
        filter_cloud_pub = rospy.Publisher(filter_cloud_name, PointCloud2, queue_size=1)

        # Publish
        allow_clock = rospy.Publisher('allow_clock', String, queue_size=1)
        reset_sub = rospy.Subscriber("reset_tracks", String, reset_cbk)

        path_pub_list = list()
        for i in range(4):
            name = rospy.resolve_name(node_name + '/track_path_' + str(i))

            # Publish the lidar overlay image
            pub = rospy.Publisher(name, Path, queue_size=1)
            path_pub_list.append(pub)

        rospy.on_shutdown(shutdown)
        # for i in range(2):
        time.sleep(0.5)
        allow_clock.publish("Ready")
        print('Ready')
        rospy.spin()

    except rospy.ROSInterruptException:
        pass


