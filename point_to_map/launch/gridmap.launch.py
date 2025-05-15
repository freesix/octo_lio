from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    share_path = get_package_share_directory('point_to_map')

    param_file = DeclareLaunchArgument(
        'param_file',
        default_value=os.path.join(share_path, 'config', 'params.yaml'),
        description="Path to the parameters file")
    
    gridmapNode = Node(
        package='point_to_map',
        executable='gridmap_node',
        name='gridmap',
        output='screen',
        parameters=[LaunchConfiguration('param_file')]     
    )

    return LaunchDescription([
        param_file,
        gridmapNode     
    ])