#!/usr/bin/env python
import numpy as np
import cv2
import ros_numpy
from vision_msgs.msg import ObjectHypothesisWithPose
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import datetime
import pcl

octree = None


def preprocess_cloud(cloud_from_ros):
	points = ros_numpy.point_cloud2.pointcloud2_to_array(cloud_from_ros)  #
	aux = points.reshape(-1)
	xyz = np.array(aux[['x', 'y', 'z']].tolist())
	xyz.setflags(write=1)
	# xyz[:, 0] = -xyz[:, 0]
	# xyz[:, 1] = -xyz[:, 1]
	return xyz


def cart2pol(x, y):
	rho = np.sqrt(x**2 + y**2)
	phi = np.arctan2(y, x)
	return np.array((rho, phi))


def custom_cylinder_metric(X1,X2):
	pol_a = cart2pol(X1[0],X1[1])
	pol_b = cart2pol(X2[0], X2[1])
	r_diff = np.abs(pol_a[0]-pol_b[0])
	a_diff = np.cos(pol_a[1]-pol_b[1])
	distance = np.linalg.norm([r_diff, a_diff])
	return distance


def mimic_minkowski(X1,X2):
	distance=0.0                            # Initialize distance
	distance=np.linalg.norm(X1-X2)           # Calculate final distance as sqrt of previous sum
	return distance


def octree_change_detector(points_a, points_b):
	# // Octree resolution - side length of octree voxels
	global octree
	resolution = 0.1
	cloudA = pcl.PointCloud()
	cloudA.from_array(points_a.astype(np.float32))
	if octree is None:
		octree = cloudA.make_octreeChangeDetector(resolution)
	octree.add_points_from_input_cloud()
	#
	octree.switchBuffers()
	cloudB = pcl.PointCloud()
	cloudB.from_array(points_b.astype(np.float32))
	octree.set_input_cloud(cloudB)
	octree.add_points_from_input_cloud()
	#
	newPointIdxVector = octree.get_PointIndicesFromNewVoxels()
	print('Output from getPointIndicesFromNewVoxels:')
	cloudB.extract(newPointIdxVector)
	octree.delete_tree()
	return points_b[newPointIdxVector, :]
	# return points_b[[0,1,2,3,4,5,6], :]

def cluster_dbscan(points, eps=0.05, min=10):
	start_time = datetime.datetime.now()
	label_colors = None
	if points is not None:
		clustering = DBSCAN(eps=eps, min_samples=min, algorithm='auto', n_jobs=1).fit(points[:, :])
		unique_labels = set(clustering.labels_)
		print('Debugging unique labels ', unique_labels)
		colors = [np.array(plt.cm.Spectral(each)[:-1])
				  for each in np.linspace(0, 1, len(unique_labels))]
		unique_colors = dict(zip(list(unique_labels), colors))
		label_colors = np.array([unique_colors[label] for label in clustering.labels_])
		cluster_sizes = list()
		centroids = list()
		group_list = list()
		for count, cluster_id in enumerate(unique_labels):
			if cluster_id != -1:
				class_member_mask = (clustering.labels_ == cluster_id)
				cluster_pts = points[class_member_mask]
				group_list.append(cluster_pts)
				centroid = np.mean(cluster_pts, axis=0)
				variance = np.var(cluster_pts, axis=0)
				if variance[2] > 0.01:
				# print('Centroid calculated :', centroid)
					cluster_sizes.append(cluster_pts.shape[0])
					centroids.append(centroid)
	end_time = datetime.datetime.now()
	time_diff = (end_time - start_time)
	execution_time = time_diff.total_seconds()
	print('time spend on clustering ', execution_time)
	print('cluster sizes : ', cluster_sizes)
	return label_colors, centroids, cluster_sizes


def remove_all_horizontal_planes(points):
	A = points[:,:]
	B = np.roll(points,1)
	vectors = A - B
	vectors_2d = vectors[:,:-1]
	diff = vectors_2d - np.roll(vectors_2d,1)
	angles = np.rad2deg(np.arctan2(diff[:,1], diff[:,0]))
	boolean = np.logical_and(np.abs(angles) > 30, np.abs(angles) < 120)
	print('  horiz debug ', points[boolean,:])
	return points[boolean,:]


def pcl_dbscan(points):
	start_time = datetime.datetime.now()
	# label_colors = np.zeros(shape=points.shape)
	label_colors = None
	if points is not None:
		pcl_inst = pcl.PointCloud(points.astype(np.float32))
		tree = pcl_inst.make_kdtree()
		ec = pcl_inst.make_EuclideanClusterExtraction()
		ec.set_ClusterTolerance(0.3)
		ec.set_MinClusterSize(10)
		ec.set_MaxClusterSize(25000)
		ec.set_SearchMethod(tree)
		cluster_indices = ec.Extract()
		# clustering = DBSCAN(eps=0.2).fit(points)
		# Black removed and is used for noise instead.
		# unique_labels = len(cluster_indices)
		# print('Debugging unique labels ', unique_labels)
		# colors = [np.array(plt.cm.Spectral(each)[:-1])
		# 		  for each in np.linspace(0, 1, len(cluster_indices))]
		# unique_colors = dict(zip(list(range(unique_labels)), colors))
		# for label in range(unique_labels):
		# 	indexes = cluster_indices[label]
		# 	label_colors[indexes] = unique_colors[label]
		cluster_sizes = list()
		centroids = list()
		group_list = list()
		for cluster in cluster_indices:
			cluster_pts = points[cluster]
			cluster_sizes.append(cluster_pts.shape[0])
			# group_list.append(cluster_pts)
			centroid = np.mean(cluster_pts, axis=0)
			# print('Centroid calculated :', centroid)
			centroids.append(centroid)
	end_time = datetime.datetime.now()
	time_diff = (end_time - start_time)
	execution_time = time_diff.total_seconds()
	print('time spend on clustering ', execution_time)
	return label_colors, centroids, cluster_sizes


def voxelize(points):
	start_time = datetime.datetime.now()
	if points is not None:
		pcl_inst = pcl.PointCloud(points.astype(np.float32))
		sor = pcl_inst.make_voxel_grid_filter()
		sor.set_leaf_size(0.1, 0.1, 0.1)
		cloud_filtered = sor.filter()
		end_time = datetime.datetime.now()
		time_diff = (end_time - start_time)
	execution_time = time_diff
	print(execution_time.total_seconds())
	return cloud_filtered.to_array()