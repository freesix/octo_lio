from launch import LaunchDescription 
from launch_ros.actions import Node 
from launch.actions import IncludeLaunchDescription
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
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                lio_sam_dir, 'launch', 'run.launch.py'))     
        )
    ])