from launch import LaunchDescription 
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node 

def generate_launch_description():
    VOLECITY_TOPIC = LaunchConfiguration('velocity_cmd_topic')
    ENCODER_TOPIC = LaunchConfiguration('odom_publisher')
    serial_port_name = LaunchConfiguration('serial_port_name')
    serial_baudrate = LaunchConfiguration('serial_baudrate')   
    right_wheel_radius = LaunchConfiguration('right_wheel_radius')    
    left_wheel_radius = LaunchConfiguration('left_wheel_radius')     
    wheel_distance = LaunchConfiguration('wheel_distance')
    encoder_resolution = LaunchConfiguration('encoder_resolution') 
    begin_lost_pose = LaunchConfiguration('begin_lost_pose')    
    odom_frame_id = LaunchConfiguration('odom_frame_id')
    base_frame_id = LaunchConfiguration('base_frame_id')

    VOLECITY_TOPIC_cmd = DeclareLaunchArgument('velocity_cmd_topic', default_value='/cmd_vel')
    ENCODER_TOPIC_cmd = DeclareLaunchArgument('odom_publisher', default_value='/odom')
    serial_port_name_cmd = DeclareLaunchArgument('serial_port_name', default_value='/dev/ttyUSB0')
    serial_baudrate_cmd = DeclareLaunchArgument('serial_baudrate', default_value='19200')   
    right_wheel_radius_cmd = DeclareLaunchArgument('right_wheel_radius', default_value='0.0845')    
    left_wheel_radius_cmd = DeclareLaunchArgument('left_wheel_radius', default_value='0.0845')     
    wheel_distance_cmd = DeclareLaunchArgument('wheel_distance', default_value='0.489')
    encoder_resolution_cmd = DeclareLaunchArgument('encoder_resolution', default_value='5600') 
    begin_lost_pose_cmd = DeclareLaunchArgument('begin_lost_pose', default_value='false')    
    odom_frame_id_cmd = DeclareLaunchArgument('odom_frame_id', default_value='odom')
    base_frame_id_cmd = DeclareLaunchArgument('base_frame_id', default_value='base_link')

    robot_motor_node = Node(
        package='motor_drive',
        executable='motor_drive_node',
        parameters=[{
            'velocity_cmd_topic':VOLECITY_TOPIC, 
            'odom_publisher' : ENCODER_TOPIC,
            'serial_port_name':serial_port_name,
            'serial_baudrate':serial_baudrate,
            'right_wheel_radius':right_wheel_radius,
            'left_wheel_radius':left_wheel_radius,
            'wheel_distance':wheel_distance,
            'encoder_resolution':encoder_resolution,
            'begin_lost_pose':begin_lost_pose,
            'odom_frame_id':odom_frame_id,
            'base_frame_id':base_frame_id}] 
    )

    return LaunchDescription([
        VOLECITY_TOPIC_cmd,
        ENCODER_TOPIC_cmd,
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
