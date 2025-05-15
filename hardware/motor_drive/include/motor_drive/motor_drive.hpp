#pragma once
#include <serial/serial.h>
#include <math.h>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// extern std::string m_SerialPortName;

class MotorDrive : public serial::Serial{
public:
	
	/** Robot velocity, including the velocity in x, y, and the angular velocity around z  */
	struct RobotVelocity
	{
		float linear_vel_x;
		float linear_vel_y;
		float angular_vel_z;
	};
	
	bool serial_init(std::string serial_port_name, const int serial_baudrate);

	bool motor_init(double RightRadius, double LeftRadius, double WheelDistance, int EncoderResolution);

	bool enable_motor();

	void disable_motor();

	void setRobotVelocity(RobotVelocity robot_vel);

	int getEncoderInformation(int &left_encoder, int &right_encoder);

	bool isVelocityModle();

	nav_msgs::msg::Odometry decoder(int leftcoder, int rightcoder);

	void cmdCallback(geometry_msgs::msg::Twist::SharedPtr msg);

private:

	double normalize_theta(double theta);

	int normalize_encoder_diff(int diff, int resolution);


};