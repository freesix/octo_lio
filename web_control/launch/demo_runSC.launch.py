from launch import LaunchDescription 
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from  ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # lakibeam_dir = get_package_share_directory('lakibeam1')
    # serial_imu_dir = get_package_share_directory('serial_imu')
    # serial_imu_old_dir = get_package_share_directory('serial_imu_old')
    motor_dir = get_package_share_directory('motor_drive')
    teleop_joy_dir = get_package_share_directory('teleop_twist_joy')
    livox_dir = get_package_share_directory('livox_ros_driver2')
    lio_sam_dir = get_package_share_directory('lio_sam')

    # build octomap use terrain analysis result, that is, intensity field is used as height field
    octomap_server_node = Node(
    package='octomap_server',
        executable='octomap_server_node',
        output='screen',
        parameters=[{
            "frame_id": "map",#! MUST SET TO MAP
            "base_frame_id": "sensor_at_scan",#! THIE MUST BE SPECIFIED IF YOU WANT TO SAVE MAP
            # "use_height_map":False,
            # "colored_map":False,
            # "color_factor":0.8,
            # "point_cloud_min_x":-10.0,
            # "point_cloud_max_x":10.0,
            # "point_cloud_min_y":-10.0,
            # "point_cloud_max_y":10.0,
            "point_cloud_min_z":0.15,
            # "point_cloud_max_z":100.0,
            # "occupancy_min_z":1.0,
            # "occupancy_max_z":100.0,
            # "min_x_size":0.0,
            # "min_y_size":0.0,
            "filter_speckles":True,
            "filter_ground_plane":False,
            # "ground_filter.distance":0.5,
            # "ground_filter.angle":0.15,
            # "ground_filter.plane_distance": 0.07, 
            # "sensor_model.max_range": 5.0,
            # "sensor_model.hit": 0.7,
            # "sensor_model.miss": 0.4,
            # "sensor_model.min": 0.12,
            # "sensor_model.max": 0.97, 
            # "compress_map": True,
            # "incremental_2D_projection": True, 
            "resolution": 0.1,
            "latch": True,#! THIE MUST BE SPECIFIED IF YOU WANT TO SAVE MAP
            # "publish_free_space": False,
            # "octomap_path", ""
        }],
            remappings=[('/cloud_in', '/lio_sam/mapping/cloud_registered')]
    )


    return LaunchDescription([
        # IncludeLaunchDescription(
            # PythonLaunchDescriptionSource(os.path.join(
                # lakibeam_dir, 'launch', 'lakibeam1_scan.launch.py'))        
        # ),
        # IncludeLaunchDescription(
            # PythonLaunchDescriptionSource(os.path.join(
                # serial_imu_dir, 'launch', 'serial_imu.launch.py')) 
        # ),    
        # IncludeLaunchDescription(
            # PythonLaunchDescriptionSource(os.path.join(
                # serial_imu_old_dir, 'launch', 'serial_imu_old.launch.py'))
        # ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                motor_dir, 'launch', 'robot_motor_base.launch.py')) 
        ), 
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                teleop_joy_dir, 'launch', 'teleop-launch.py'))   
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                livox_dir, 'launch_ROS2', 'msg_MID360_launch.py'))     
        ),
        
        TimerAction(
            period=5.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(os.path.join(
                        lio_sam_dir, 'launch', 'runSCLoop.launch.py'))),
                octomap_server_node 
            ]
        ) 
    ])