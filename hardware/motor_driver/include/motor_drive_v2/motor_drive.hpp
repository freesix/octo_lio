#pragma once
#include <serial/serial.h>
#include <math.h>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <mutex>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// extern std::string m_SerialPortName;

class MotorDrive : public serial::Serial{
public:
	// Constructor
	MotorDrive() : last_velocity_cmd_time_(rclcpp::Time(0, 0, RCL_SYSTEM_TIME)), min_cmd_interval_(0.05) {}
	
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
	
	// Atomic serial communication functions
	bool atomicSerialWrite(const uint8_t* data, size_t length);
	bool atomicSerialRead(std::vector<uint8_t>& buffer, size_t expected_length, int timeout_ms = 50);

	nav_msgs::msg::Odometry decoder(int leftcoder, int rightcoder);

	// 基于速度指令积分计算里程计（新增重载）
	nav_msgs::msg::Odometry decoder(const geometry_msgs::msg::Twist& velocity_cmd);

	void cmdCallback(geometry_msgs::msg::Twist::SharedPtr msg);

	/** Reset odometry state to zero and reinitialize encoder baselines */
	void resetOdometry();
	
	/** Get current velocity message */
	bool getCurrentVelocity(geometry_msgs::msg::Twist& velocity_msg);

	/** Set calibration mode (0=odometry mode, 1=calibration mode) */
	void setCalibrationMode(int mode);

	/** Perform wheel radius and wheelbase calibration */
	void performCalibration(int leftcoder, int rightcoder);

private:

	double normalize_theta(double theta);

	int normalize_encoder_diff(int diff, int resolution);

	// Apply first-order low-pass filter
	// isAngular=true -> filter v_th, false -> filter vx
	double filterVelocity(double raw_value, double dt, bool isAngular);

	// Mutex to protect odometry state variables
	std::mutex odometry_mutex_;
	
	// Mutex to protect serial port access
	std::mutex serial_mutex_;

	// State for angular velocity low-pass filter
	bool vth_filter_initialized_ = false;
	double vth_prev_filtered_ = 0.0;
	double vth_cutoff_hz_ = 5.0; // cutoff frequency (Hz)

	// State for linear velocity low-pass filter
	bool vx_filter_initialized_ = false;
	double vx_prev_filtered_ = 0.0;
	double vx_cutoff_hz_ = 5.0; // cutoff frequency (Hz)
	
	// Rate limiting for velocity commands
	rclcpp::Time last_velocity_cmd_time_;
	double min_cmd_interval_; // 50ms minimum interval (20Hz max)
	
	// Store serial port parameters for reconnection
	std::string serial_port_name_;
	int serial_baudrate_;
	
	// 速度消息相关成员变量
	geometry_msgs::msg::Twist current_velocity_msg_;
	bool velocity_msg_available_ = false;

};