#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
import hdbscan

class LidarDataFilter:
    def __init__(self):
        rospy.init_node('cone_node', anonymous=True)

        # 수평 각도의 최소 및 최대 값을 설정
        self.min_horizontal_angle = np.radians(-20)
        self.max_horizontal_angle = np.radians(20)

        self.pub_filtered = rospy.Publisher("/filtered_points", PointCloud2, queue_size=10)
        self.pub_clusters = rospy.Publisher("/point_clusters", PointCloud2, queue_size=10)
        self.pub_bounding_boxes = rospy.Publisher("/bounding_boxes", Marker, queue_size=10)
        self.pub_lines = rospy.Publisher("/bounding_boxes", Marker, queue_size=10)  # 이름 변경
        self.pub_control = rospy.Publisher("/erp42_ctrl_cmd", Twist, queue_size=10)  # 이름 변경

        rospy.Subscriber('/velodyne_points', PointCloud2, self.lidar_callback)

        self.object_positions = []
        self.angle_with_x_axis_deg = 0

    def lidar_callback(self, msg):
        # PointCloud2 메시지를 Numpy 배열로 변환
        pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pc_data))

        # 수평 각도를 추출하고 원하는 각도 범위 내의 포인트를 필터링
        horizontal_angles = np.arctan2(points[:, 1], points[:, 0])
        angle_filtered_points = points[(horizontal_angles >= self.min_horizontal_angle) & (horizontal_angles <= self.max_horizontal_angle)]

        # x 및 y 값을 사용하여 원점으로부터의 거리를 계산
        distances = np.sqrt(angle_filtered_points[:, 0]**2 + angle_filtered_points[:, 1]**2)
        desired_distance_min, desired_distance_max = 2.0, 4.0

        # 거리만으로 포인트를 필터링 (필요한 범위로 조정)
        filtered_points = angle_filtered_points[(distances >= desired_distance_min) & (distances <= desired_distance_max)]
        
        # HDBSCAN을 사용하여 필터링된 포인트를 클러스터링
        clustered_points = self.cluster_points(filtered_points)

        # 필터링된 포인트를 새로운 PointCloud2 메시지로 게시
        header = msg.header
        pc_msg_filtered = pc2.create_cloud_xyz32(header, filtered_points)
        self.pub_filtered.publish(pc_msg_filtered)

        # 클러스터링된 포인트를 하나의 배열로 합치기
        all_clustered_points = np.vstack(clustered_points)

        # 클러스터링된 포인트를 새로운 PointCloud2 메시지로 게시
        pc_msg_clusters = self.create_pointcloud_message(all_clustered_points, header)
        self.pub_clusters.publish(pc_msg_clusters)

        # 각 클러스터에 대한 3D 바운딩 박스를 생성
        bounding_boxes = self.generate_bounding_boxes(clustered_points, header)
        self.pub_bounding_boxes.publish(bounding_boxes)

        # 객체 위치를 업데이트하고 객체를 왼쪽 또는 오른쪽으로 분류
        left_objects, right_objects = self.classify_objects()

        # 왼쪽 및 오른쪽 객체의 가장 가까운 포인트에서 중앙까지 선을 그림
        self.draw_lines(left_objects, right_objects)

    def cluster_points(self, points):
        # HDBSCAN 클러스터링 적용
        clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=1, allow_single_cluster=True)
        labels = clusterer.fit_predict(points)

        # 각 클러스터에 대한 포인트를 추출
        unique_labels = np.unique(labels)
        clustered_points = [points[labels == label] for label in np.unique(labels) if label != -1]

        return clustered_points

    def create_pointcloud_message(self, points, header):
        pc_msg = PointCloud2()
        pc_msg.header = header
        pc_msg.height = 1
        pc_msg.width = len(points)
        pc_msg.is_dense = False
        pc_msg.is_bigendian = False
        pc_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        pc_msg.point_step = 12
        pc_msg.row_step = 12 * len(points)
        pc_msg.data = np.asarray(points, np.float32).tobytes()

        return pc_msg

    def generate_bounding_boxes(self, clustered_points, header):
        marker = Marker()
        marker.header = header
        marker.ns = "bounding_boxes"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.object_positions = []  # Reset the object positions list

        for cluster in clustered_points:
            min_point = np.min(cluster, axis=0)
            max_point = np.max(cluster, axis=0)
            self.object_positions.append(((min_point[0] + max_point[0]) / 2, (min_point[1] + max_point[1]) / 2))

            points = [
                (min_point[0], min_point[1], min_point[2]),
                (min_point[0], max_point[1], min_point[2]),
                (max_point[0], max_point[1], min_point[2]),
                (max_point[0], min_point[1], min_point[2]),
                (min_point[0], min_point[1], max_point[2]),
                (min_point[0], max_point[1], max_point[2]),
                (max_point[0], max_point[1], max_point[2]),
                (max_point[0], min_point[1], max_point[2])
            ]

            # Add edges to form the bounding box
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]

            for edge in edges:
                marker.points.append(Point(*points[edge[0]]))
                marker.points.append(Point(*points[edge[1]]))

        return marker

    def classify_objects(self):
        # 차량의 진행 방향을 기준으로 객체를 왼쪽 또는 오른쪽으로 분류
        left_objects = []
        right_objects = []

        for position in self.object_positions:
            if position[1] >= 0:
                left_objects.append(position)
            else:
                right_objects.append(position)

        return left_objects, right_objects

    def draw_lines(self, left_objects, right_objects):
        if not left_objects or not right_objects:
            return

        # 왼쪽 및 오른쪽 객체의 y축에서 가장 가까운 포인트를 찾음
        closest_left = min(left_objects, key=lambda x: abs(x[1]))
        closest_right = min(right_objects, key=lambda x: abs(x[1]))

        # 가장 가까운 포인트 사이의 중간점을 계산
        line_point = [(closest_left[0] + closest_right[0]) / 2.0, (closest_left[1] + closest_right[1]) / 2.0]

        # 중앙에서 중간점까지의 선 마커를 게시
        line_marker = Marker()
        line_marker.header.frame_id = "velodyne"
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.1
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        line_marker.points.append(Point(0, 0, 0))
        line_marker.points.append(Point(line_point[0], line_point[1], 0))

        # 방향 벡터를 계산
        direction_vector = np.array([line_point[0], line_point[1], 0.0])
        direction_vector /= np.linalg.norm(direction_vector)

        # x축에 대한 각도를 계산
        angle_with_x_axis = np.arctan2(direction_vector[1], direction_vector[0])
        self.angle_with_x_axis_deg = np.degrees(angle_with_x_axis)

        # 각도를 [-33, 33]도 범위로 조정
        if self.angle_with_x_axis_deg > 33:
            self.angle_with_x_axis_deg = 33
        elif self.angle_with_x_axis_deg < -33:
            self.angle_with_x_axis_deg = -33

        # 제어 명령 생성 및 게시
        msg = Twist()
        msg.linear.x = 0.5
        msg.angular.z = -self.angle_with_x_axis_deg * (np.pi / 180)

        self.pub_lines.publish(line_marker)
        self.pub_control.publish(msg)

if __name__ == '__main__':
    try:
        LidarDataFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
