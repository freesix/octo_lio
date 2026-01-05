from launch import LaunchDescription 
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node 

def generate_launch_description():
    # 定义参数变量
    velocity_cmd_topic = LaunchConfiguration('velocity_cmd_topic')
    odom_publisher = LaunchConfiguration('odom_publisher')
    serial_port_name = LaunchConfiguration('serial_port_name')
    serial_baudrate = LaunchConfiguration('serial_baudrate')   
    right_wheel_radius = LaunchConfiguration('right_wheel_radius')    
    left_wheel_radius = LaunchConfiguration('left_wheel_radius')     
    wheel_distance = LaunchConfiguration('wheel_distance')
    encoder_resolution = LaunchConfiguration('encoder_resolution') 
    begin_lost_pose = LaunchConfiguration('begin_lost_pose')    
    odom_frame_id = LaunchConfiguration('odom_frame_id')
    base_frame_id = LaunchConfiguration('base_frame_id')

    # 声明启动参数
    velocity_cmd_topic_cmd = DeclareLaunchArgument('velocity_cmd_topic', default_value='/cmd_vel')
    odom_publisher_cmd = DeclareLaunchArgument('odom_publisher', default_value='/odom_1')
    serial_port_name_cmd = DeclareLaunchArgument('serial_port_name', default_value='/dev/ttymotor')
    serial_baudrate_cmd = DeclareLaunchArgument('serial_baudrate', default_value='115200')   
    right_wheel_radius_cmd = DeclareLaunchArgument('right_wheel_radius', default_value='0.0845')    
    left_wheel_radius_cmd = DeclareLaunchArgument('left_wheel_radius', default_value='0.0845')     
    wheel_distance_cmd = DeclareLaunchArgument('wheel_distance', default_value='0.385')
    encoder_resolution_cmd = DeclareLaunchArgument('encoder_resolution', default_value='5600') 
    begin_lost_pose_cmd = DeclareLaunchArgument('begin_lost_pose', default_value='false')    
    odom_frame_id_cmd = DeclareLaunchArgument('odom_frame_id', default_value='odom')
    base_frame_id_cmd = DeclareLaunchArgument('base_frame_id', default_value='base_link')

    # 创建机器人电机节点
    robot_motor_node = Node(
        package='motor_drive_v2',
        executable='motor_drive_v2_node',
        parameters=[{
            'velocity_cmd_topic': velocity_cmd_topic, 
            'odom_publisher': odom_publisher,
            'serial_port_name': serial_port_name,
            'serial_baudrate': serial_baudrate,
            'right_wheel_radius': right_wheel_radius,
            'left_wheel_radius': left_wheel_radius,
            'wheel_distance': wheel_distance,
            'encoder_resolution': encoder_resolution,
            'begin_lost_pose': begin_lost_pose,
            'odom_frame_id': odom_frame_id,
            'base_frame_id': base_frame_id
        }] 
    )

    return LaunchDescription([
        velocity_cmd_topic_cmd,
        odom_publisher_cmd,
        serial_port_name_cmd,
        serial_baudrate_cmd,
        right_wheel_radius_cmd,
        left_wheel_radius_cmd,
        wheel_distance_cmd,
        encoder_resolution_cmd,
        begin_lost_pose_cmd,
        odom_frame_id_cmd,
        base_frame_id_cmd,
        robot_motor_node     
    ])
