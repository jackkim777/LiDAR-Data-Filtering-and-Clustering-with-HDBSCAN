# LiDAR Data Filtering and Clustering

This project implements a LiDAR data filtering and clustering system using ROS (Robot Operating System) to process point cloud data from a Velodyne LiDAR sensor. The system filters points based on their horizontal angles and distances, clusters them using the HDBSCAN algorithm, and visualizes the results with bounding boxes and control commands for an autonomous vehicle.

## Features

- Filters point cloud data based on a specified range of horizontal angles and distances.
- Clusters filtered points using HDBSCAN for effective object detection.
- Generates 3D bounding boxes around detected objects.
- Classifies objects into left and right based on their position relative to the vehicle's forward direction.
- Visualizes object positions and central line between the closest left and right objects.
- Publishes control commands for the autonomous vehicle based on the object's positions.

## Dependencies

- ROS (Robot Operating System)
- NumPy
- HDBSCAN
- sensor_msgs
- visualization_msgs
- geometry_msgs

## Usage

1. Ensure you have ROS installed and configured on your system.
2. Clone this repository and navigate to the project directory.
3. Build the package using `catkin_make`.
4. Launch the ROS node using:
   ```bash
   rosrun <your_package_name> <your_script_name>.py
   ```
5. Subscribe to the `/filtered_points`, `/point_clusters`, and `/bounding_boxes` topics to visualize the results.
