import geometry_msgs.msg
import launch
import launch_ros.actions
import launch_testing

import pytest

import test_joy_twist


@pytest.mark.rostest
def generate_test_description():
    teleop_node = launch_ros.actions.Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        parameters=[{
            'publish_stamped_twist': True,
            'axis_linear.x': 1,
            'axis_angular.yaw': 0,
            'scale_linear.x': 2.0,
            'scale_angular.yaw': 3.0,
            'enable_button': 0,
        }],
    )

    return launch.LaunchDescription([
            teleop_node,
            launch_testing.actions.ReadyToTest(),
        ]), locals()


class PublishTwistStamped(test_joy_twist.TestJoyTwist):

    def setUp(self):
        self.cmd_vel_msg_type = geometry_msgs.msg.TwistStamped
        super().setUp()
        self.joy_msg['axes'] = [0.3, 0.4]
        self.joy_msg['buttons'] = [1]
        self.expect_cmd_vel['linear']['x'] = 0.8
        self.expect_cmd_vel['angular']['z'] = 0.9
