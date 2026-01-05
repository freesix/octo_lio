#include "motor_drive_v2/motor_drive.hpp"
#include "motor_drive_v2/crc.hpp"
#include "motor_drive_v2/type.hpp"
#include <cmath>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Transform.h>
#include <vector>
#include <chrono>

static rclcpp::Clock s_throttle_clock(RCL_SYSTEM_TIME);
uint8_t m_uVelocityModle[8] = {0x02, 0x43, 0x21, 0x02, 0x31, 0x02, 0x7b, 0x9b};

//serial command to enable the motor
uint8_t m_uEnableMotor[12] = {0x02,0x44,0x21,0x00,0x31,0x00,0x00,0x01,0x00,0x01,0x85,0x3b};

//serial command to disable the motor
uint8_t m_uDisableMotor[12] = {0x02,0x44,0x21,0x00,0x31,0x00,0x00,0x00,0x00,0x00,0x15,0x3b};

//serial command to read the speed of the motor, currentlt not used
uint8_t m_uReadMotorSpeed[8] = {0x02,0x43,0x50,0x00,0x51,0x00,0x68,0xa6};

//serial command to read the encoder
uint8_t m_uReadEncoder[8] = {0x02,0x43,0x50,0x04,0x51,0x04,0x28,0xa4};

//radius of right wheel
double m_dRightWheelRadius = 0.1;

//radius of left wheel
double m_dLeftWheelRadius = 0.1;

// distance between left and right
double m_dWheelDistance = 0.5;

//define the resolution of the encoder
int m_iEncoderResolution=5600;

//left wheel encoder reading
int m_iLeftEncoderReading=-1;

//right wheel encoder reading
int m_iRightEncoderReading=-1;

//right wheel encoder reading
int m_iEmgergencyStopStatus=-1;

int current_stop_status = -1;

bool lose_pose = false;
bool begin_lost_pose_ = false;

rclcpp::Time current_time;
rclcpp::Time last_time;
tf2::Quaternion odom_quat;
tf2::Quaternion current_quat(0, 0, 0, 1); // 初始化为单位四元数
//the current odom
double x = 0;
double y = 0;
double th = 0;
// 保存上一帧的角度，用于圆弧积分计算
double last_th = 0;

double MotorDrive::filterVelocity(double raw_value, double dt, bool isAngular){
	// 保护：无效参数直接透传并初始化对应状态
	if(!std::isfinite(raw_value) || !std::isfinite(dt) || dt <= 0.0){
		if(isAngular){
			vth_prev_filtered_ = raw_value;
			vth_filter_initialized_ = true;
		}else{
			vx_prev_filtered_ = raw_value;
			vx_filter_initialized_ = true;
		}
		return raw_value;
	}
	// 选择对应的截止频率与状态
	double cutoff_hz = isAngular ? vth_cutoff_hz_ : vx_cutoff_hz_;
	if(cutoff_hz <= 0.0){
		if(isAngular){
			vth_prev_filtered_ = raw_value;
			vth_filter_initialized_ = true;
		}else{
			vx_prev_filtered_ = raw_value;
			vx_filter_initialized_ = true;
		}
		return raw_value;
	}
	double tau = 1.0 / (2.0 * M_PI * cutoff_hz);
	double alpha = dt / (tau + dt);
	if(isAngular){
		if(!vth_filter_initialized_){
			vth_prev_filtered_ = raw_value;
			vth_filter_initialized_ = true;
			return vth_prev_filtered_;
		}
		vth_prev_filtered_ = vth_prev_filtered_ + alpha * (raw_value - vth_prev_filtered_);
		return vth_prev_filtered_;
	}else{
		if(!vx_filter_initialized_){
			vx_prev_filtered_ = raw_value;
			vx_filter_initialized_ = true;
			return vx_prev_filtered_;
		}
		vx_prev_filtered_ = vx_prev_filtered_ + alpha * (raw_value - vx_prev_filtered_);
		return vx_prev_filtered_;
	}
}

bool MotorDrive::serial_init(const std::string serial_port_name, const int serial_baudrate){
	if(serial_port_name.empty() || serial_baudrate != 115200){  // 115200 just
		RCLCPP_ERROR(rclcpp::get_logger("Serial_1"), "serial port or baudrate is error");
		return false;
	}

	// Store serial port parameters for reconnection
	serial_port_name_ = serial_port_name;
	serial_baudrate_ = serial_baudrate;

	RCLCPP_INFO(rclcpp::get_logger("Serial"), "Initializing serial port: %s at %d baud", 
		serial_port_name.c_str(), serial_baudrate);

	try{
		this->setPort(serial_port_name);
		this->setBaudrate(serial_baudrate);
		// Set more appropriate timeout for 115200 baud rate
		serial::Timeout to = serial::Timeout::simpleTimeout(2000); // 2 second timeout
		this->setTimeout(to);	
		this->open();
		
		RCLCPP_INFO(rclcpp::get_logger("Serial"), "Serial port opened successfully");
	}
	catch(serial::IOException& e){
		RCLCPP_ERROR(rclcpp::get_logger("Serial"), "Unable to open port: %s", e.what());
		return false;	
	}
	
	if(this->isOpen()){
		RCLCPP_INFO(rclcpp::get_logger("Serial"), "Serial Port initialized successfully");
		return true;	
	}
	else{
		RCLCPP_ERROR(rclcpp::get_logger("Serial"), "Serial port failed to open");
		return false;	
	}
}

bool MotorDrive::motor_init(double RightRadius, double LeftRadius, double WheelDistance, int EncoderResolution){
	m_dRightWheelRadius = RightRadius;
	m_dLeftWheelRadius = LeftRadius;
	m_dWheelDistance = WheelDistance;
	m_iEncoderResolution = EncoderResolution;

	return (m_dLeftWheelRadius && m_dRightWheelRadius && m_dWheelDistance && m_iEncoderResolution);
}

bool MotorDrive::enable_motor(){
	if(!atomicSerialWrite(m_uEnableMotor, 12)) {
		RCLCPP_ERROR(rclcpp::get_logger("motor"), "Failed to enable motor");
		return false;
	}
    rclcpp::sleep_for(std::chrono::milliseconds(20)); // 20ms
	return true;
}

void MotorDrive::disable_motor(){
	if(!atomicSerialWrite(m_uDisableMotor, 12)) {
		RCLCPP_ERROR(rclcpp::get_logger("motor"), "Failed to disable motor");
	}
	rclcpp::sleep_for(std::chrono::milliseconds(10));
}

void MotorDrive::setRobotVelocity(RobotVelocity robot_vel){
// --------------------------------------------------------------------------
	if(!this->isOpen())
		return ;

	uint8_t setting_speed[12] = {0x02,0x44,0x23,0x18,0x33,0x18};

	//left wheel speed
	float left_wheel_speed;
	
	//right wheel speed
	float right_wheel_speed;

	//differential drive, compute the left wheel velocity and right wheel velocity
	left_wheel_speed = robot_vel.linear_vel_x - robot_vel.angular_vel_z * m_dWheelDistance * 0.5;
	right_wheel_speed = robot_vel.linear_vel_x + robot_vel.angular_vel_z * m_dWheelDistance * 0.5;

	//convert it to motor speed
	float left_motor_speed_float = left_wheel_speed / (2 * 3.141592 * m_dLeftWheelRadius) * 60.0;
	float right_motor_speed_float = -right_wheel_speed / (2 * 3.141592 * m_dRightWheelRadius) * 60.0;
	
	
	//limit the speed within 500 rpm
	if(fabs((int)left_motor_speed_float) > 500)
	{
		if((int)left_motor_speed_float > 0)
			left_motor_speed_float = 500;
		if((int)left_motor_speed_float < 0)
			left_motor_speed_float = -500;
	}

	if(fabs((int)right_motor_speed_float) > 500)
	{
		if((int)right_motor_speed_float > 0)
			right_motor_speed_float = 500;
		if((int)right_motor_speed_float < 0)
			right_motor_speed_float = -500;
	}

	//convert the float speed to the interger speed
	int left_motor_speed_int=(int)left_motor_speed_float;
	int right_motor_speed_int=(int)right_motor_speed_float;
	
	//left motor
	if(left_motor_speed_int>= 0 || left_motor_speed_int == -0)
	{
		//for positive speed
		setting_speed[6] = 0x00;
		setting_speed[7] = left_motor_speed_int;
	}    
	else 
	{
		//for negative speed
		setting_speed[6] = 0xff;
		setting_speed[7] = (~((int)(fabs(left_motor_speed_int)) & 0xff))+1;
	}

	//right motor
	if(right_motor_speed_int >= 0 || right_motor_speed_int == -0)
	{
		//for positive speed
		setting_speed[8] = 0x00;
		setting_speed[9] = (int)right_motor_speed_int;
	}    
	else 
	{
		//for negative speed
		setting_speed[8] = 0xff;
		setting_speed[9] = (~((int)(fabs(right_motor_speed_int)) & 0xff))+1;
	}

	//CRC computation
	uint16_t crc = crc16(setting_speed, 10, crc_16_MODBUS);
	setting_speed[10] = crc & 0xff;       //Low
	setting_speed[11] = (crc >> 8) & 0xff;   //High

	// Use atomic serial write operation
	if(!atomicSerialWrite(setting_speed, 12)) {
		RCLCPP_WARN_THROTTLE(rclcpp::get_logger("motor"), 
			s_throttle_clock, 2000,
			"Failed to send velocity command to motor");
	}
	
	// Small delay to ensure command is processed
	rclcpp::sleep_for(std::chrono::milliseconds(5));
}

int MotorDrive::getEncoderInformation(int &left_encoder, int &right_encoder){
	// Use atomic serial write operation
	if(!atomicSerialWrite(m_uReadEncoder, 8)) {
		RCLCPP_WARN_THROTTLE(rclcpp::get_logger("motor"), 
			s_throttle_clock, 2000,
			"Failed to send encoder read command");
		return -1;
	}
	
	// Wait for response with timeout
	rclcpp::sleep_for(std::chrono::milliseconds(9));
	
	// Use atomic serial read operation
	std::vector<uint8_t> response_buffer;
	if(!atomicSerialRead(response_buffer, 10, 15)) { // Expect at least 10 bytes, timeout 50ms
		RCLCPP_WARN_THROTTLE(rclcpp::get_logger("motor"), 
			s_throttle_clock, 2000,
			"Failed to read encoder response from serial port");
		return -1;
	}
	
	int data_size = response_buffer.size();
	/* RCLCPP_DEBUG_THROTTLE(rclcpp::get_logger("motor"), 
		s_throttle_clock, 
		2000, // 2000毫秒 = 2秒
		"getEncoderInformation: read data size = %d", data_size); */
	
	// 打印接收到的原始数据（十六进制格式）
	std::string hex_data = "";
	for(int j = 0; j < std::min(data_size, 20); j++) { // 只打印前20个字节避免日志过长
		char hex[8];
		snprintf(hex, sizeof(hex), "0x%02X ", response_buffer[j]);
		hex_data += hex;
	}
	/* RCLCPP_DEBUG_THROTTLE(rclcpp::get_logger("motor"), 
		s_throttle_clock, 
		2000, // 2000毫秒 = 2秒
		"getEncoderInformation: raw data = %s", hex_data.c_str()); */
	
	int returned_value = -1;
	for(int i = 0; i < data_size; i++) {
		if(i + 9 < data_size) {
			if(response_buffer[i+0] == 0x02 && response_buffer[i+1] == 0x43 && 
			   response_buffer[i+2] == 0x50 && response_buffer[i+3] == 0x04 && 
			   response_buffer[i+4] == 0x51 && response_buffer[i+5] == 0x04) {
				// 寻找特定的帧头
				left_encoder = response_buffer[i+6] * 256 + response_buffer[i+7];
				right_encoder = response_buffer[i+8] * 256 + response_buffer[i+9];
				
				RCLCPP_DEBUG_THROTTLE(rclcpp::get_logger("motor"), 
					s_throttle_clock, 
					2000, // 2000毫秒 = 2秒
					"getEncoderInformation: found encoder data at index %d, left=%d, right=%d", 
					i, left_encoder, right_encoder);
	
				returned_value = returned_value + 1;
			}
		}
	}
	
	if(returned_value == -1) {
		/* RCLCPP_WARN_THROTTLE(rclcpp::get_logger("motor"), 
			s_throttle_clock, 
			2000, // 2000毫秒 = 2秒
			"读取失败"); */
	}
	
	return returned_value;
}

bool MotorDrive::isVelocityModle(){
	this->write(m_uVelocityModle, 8);
	rclcpp::sleep_for(std::chrono::milliseconds(10));
	int p = this->available();
	std_msgs::msg::UInt8MultiArray msg;
	if(p){
		this->read(msg.data, p);
		// TODO
		for(auto it:msg.data){
			RCLCPP_INFO(rclcpp::get_logger("motor"), "modle data: 0x%02X,", it);
		}
		return true;
	}
	return false;
}

bool MotorDrive::atomicSerialWrite(const uint8_t* data, size_t length){
	// Lock to protect serial port access
	std::lock_guard<std::mutex> lock(serial_mutex_);
	
	// Check if serial port is open before attempting write
	if(!this->isOpen()) {
		RCLCPP_WARN_THROTTLE(rclcpp::get_logger("motor"), 
			s_throttle_clock, 2000,
			"Serial port not open for write operation. Attempting to reopen...");
		
		// Try to reopen the serial port
		try {
			// Check if we have stored serial parameters
			if(serial_port_name_.empty()) {
				RCLCPP_ERROR(rclcpp::get_logger("motor"), "No stored serial port parameters for reconnection");
				return false;
			}
			
			// Reconfigure and reopen the serial port
			this->setPort(serial_port_name_);
			this->setBaudrate(serial_baudrate_);
			serial::Timeout to = serial::Timeout::simpleTimeout(2000);
			this->setTimeout(to);
			this->open();
			
			if(!this->isOpen()) {
				RCLCPP_ERROR(rclcpp::get_logger("motor"), "Failed to reopen serial port");
				return false;
			}
			RCLCPP_INFO(rclcpp::get_logger("motor"), "Serial port reopened successfully");
		} catch(const std::exception& e) {
			RCLCPP_ERROR(rclcpp::get_logger("motor"), "Failed to reopen serial port: %s", e.what());
			return false;
		}
	}
	try {
		// Clear any pending data in input buffer before writing
		this->flushInput();
		
		// Perform atomic write operation
		size_t bytes_written = this->write(data, length);
		this->flush(); // Ensure data is sent immediately
		
		if(bytes_written != length) {
			RCLCPP_WARN(rclcpp::get_logger("motor"), 
				"Serial write incomplete: expected %zu bytes, wrote %zu bytes", 
				length, bytes_written);
			return false;
		}
		
		return true;
	} catch(const std::exception& e) {
		RCLCPP_ERROR(rclcpp::get_logger("motor"), 
			"Serial write error: %s", e.what());
		// Mark port as closed if write fails
		try {
			this->close();
		} catch(...) {
			// Ignore close errors
		}
		return false;
	}
}

bool MotorDrive::atomicSerialRead(std::vector<uint8_t>& buffer, size_t expected_length, int timeout_ms){
	// Lock to protect serial port access
	std::lock_guard<std::mutex> lock(serial_mutex_);
	try {
		buffer.clear();
		buffer.reserve(expected_length);
		
		// Wait for data with timeout
		auto start_time = std::chrono::steady_clock::now();
		int available_bytes = 0;
		
		while(rclcpp::ok()) {
			available_bytes = this->available();
			if(available_bytes > 0) {
				break;
			}
			
			auto current_time = std::chrono::steady_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
				current_time - start_time).count();
			
			if(elapsed >= timeout_ms) {
				RCLCPP_WARN_THROTTLE(rclcpp::get_logger("motor"), 
					s_throttle_clock, 2000,
					"Serial read timeout: no data available after %d ms", timeout_ms);
				return false;
			}
			
			// Small sleep to avoid busy waiting
			rclcpp::sleep_for(std::chrono::milliseconds(1));
		}
		
		// Check if we exited due to shutdown signal
		if(!rclcpp::ok()) {
			RCLCPP_INFO(rclcpp::get_logger("motor"), "Serial read interrupted by shutdown signal");
			return false;
		}
		
		// Read all available data atomically
		buffer.resize(available_bytes);
		size_t bytes_read = this->read(buffer.data(), available_bytes);
		
		if(bytes_read != static_cast<size_t>(available_bytes)) {
			RCLCPP_WARN(rclcpp::get_logger("motor"), 
				"Serial read incomplete: expected %d bytes, read %zu bytes", 
				available_bytes, bytes_read);
			return false;
		}
		
		return true;
	} catch(const std::exception& e) {
		RCLCPP_ERROR(rclcpp::get_logger("motor"), 
			"Serial read error: %s", e.what());
		return false;
	}
}

/** Normalization of theta, for any given theta, return a theta value between -pi and pi  */
double MotorDrive::normalize_theta(double theta){
	int multiplier;
	if(theta>=-M_PI&&theta<M_PI)
	{
		return theta;
	}
	multiplier=(int)(theta/(2*M_PI));
	theta=theta-multiplier*2*M_PI;
	if(theta>=M_PI)
	{
		theta=theta-2*M_PI;
	}
	if(theta<-M_PI)
	{
		theta=theta+2*M_PI;
	}
	return theta;
}


/** Normalization of encoder difference. Given a encoder difference, return a value between -resolution/2 and -resolution/2 */
// --------------------------------------------------------------------------
int MotorDrive::normalize_encoder_diff(int diff, int resolution){
	int multiplier;
	if(diff>=-resolution/2&&diff<resolution/2)
	{
		return diff;
	}
	multiplier=(int)(diff/resolution);
	diff=diff-multiplier*resolution;
	if(diff>=resolution/2)
	{
		diff=diff-resolution;
	}
	if(diff<-resolution/2)
	{
		diff=diff+resolution;
	}
	return diff;
}

nav_msgs::msg::Odometry MotorDrive::decoder(int leftcoder, int rightcoder){//轮式里程计功能
	nav_msgs::msg::Odometry odom_msg;
	
	// Lock to protect odometry state variables
	std::lock_guard<std::mutex> lock(odometry_mutex_);
	
	if(m_iLeftEncoderReading<0 || m_iRightEncoderReading <0){//第一次运行
		m_iLeftEncoderReading = leftcoder;
		m_iRightEncoderReading = rightcoder;
		last_time = s_throttle_clock.now();
		// 第一次运行时，last_th应该等于当前th（都是0）
		last_th = th;
	}
	else{
		current_time = s_throttle_clock.now();
		double dt = (current_time - last_time).nanoseconds() * 1e-9;//计算时间间隔
		last_time = current_time;

		// 保护：若 dt<=0，跳过本次积分
		if(dt <= 0.0){
			RCLCPP_WARN_THROTTLE(rclcpp::get_logger("motor"), s_throttle_clock, 2000,
				"decoder: dt<=0 (dt=%.9f), skip this update", dt);
			return odom_msg;
		}
		
		// 添加最大时间间隔限制，防止dt过大导致数值不稳定
		const double MAX_DT = 0.1; // 100ms
		if(dt > MAX_DT) {
			RCLCPP_WARN_THROTTLE(rclcpp::get_logger("motor"), s_throttle_clock, 2000,
				"decoder: dt too large (%.6f), limiting to %.6f", dt, MAX_DT);
			dt = MAX_DT;
		}

		// get the encoder difference
		int left_diff = leftcoder-m_iLeftEncoderReading;//编码器差值计算
		int right_diff = rightcoder - m_iRightEncoderReading;
		//left_diff = -left_diff;  // reversal
		right_diff = -right_diff;
		// normalize the encoder difference
		left_diff = normalize_encoder_diff(left_diff, m_iEncoderResolution);//归一化
		right_diff = normalize_encoder_diff(right_diff, m_iEncoderResolution);
		m_iLeftEncoderReading = leftcoder;
		m_iRightEncoderReading = rightcoder;
		
		//get the distance and rotation inferred by the encoder
		//距离和角度计算
		//角速度 = (右轮速度 - 左轮速度) / 轮距
		double linear = 0.5*(2.0*M_PI*m_dLeftWheelRadius*left_diff/m_iEncoderResolution) + 
						0.5*(2.0*M_PI*m_dRightWheelRadius*right_diff/m_iEncoderResolution);
		double angular = -(2.0*M_PI*m_dRightWheelRadius*right_diff/m_iEncoderResolution - 
						2.0*M_PI*m_dLeftWheelRadius*left_diff/m_iEncoderResolution)/m_dWheelDistance;


		const int sum_tick_deadband = 1; // 可根据编码器分辨率调整
		if (abs(left_diff + right_diff) <= sum_tick_deadband) {
			linear = 0.0;
		}
		// 直线死区抑制直线运动的极小角位移
		const int encoder_tick_deadband = 1; // 可视实际分辨率调整
		if (abs(left_diff) <= encoder_tick_deadband &&
			abs(right_diff) <= encoder_tick_deadband &&
			abs(left_diff - right_diff) <= encoder_tick_deadband) {
			angular = 0.0;
		}

		/* double last_x = x;
		double last_y = y;
		double last_th = th; */

		//get the velocity
		double vx=linear/dt;
		// 对线速度添加一阶低通滤波
		//vx = filterVelocity(vx, dt, false);
		double vy=0;
		double v_th=angular/dt;
		const double vth_deadband = 0.01; 
		if (fabs(v_th) < vth_deadband) {
			v_th = 0.0;
		}
		// 对角速度添加一阶低通滤波
		//v_th = filterVelocity(v_th, dt, true);
		if(fabs(angular) > 1e-8) { // 只有非零旋转才进行四元数更新
			// 创建增量旋转四元数
			tf2::Quaternion delta_quat;
			delta_quat.setRPY(0, 0, angular);
			
			// 四元数乘法进行旋转积分
			current_quat = current_quat * delta_quat;
			current_quat.normalize(); // 保持单位四元数
		}

		tf2::Matrix3x3 rot_matrix(current_quat);
		double roll, pitch, yaw;
		rot_matrix.getRPY(roll, pitch, yaw);
		th = yaw;
		
		// 改进的位置积分方法
		if(fabs(angular) > 1e-6 && fabs(v_th) > 1e-3) {
			// 使用瞬时旋转中心(ICR)方法进行精确积分
			double r = vx / v_th; // 瞬时旋转半径
			
			// 检查r是否为有限值，避免无穷大传播
			if(std::isfinite(r) && fabs(r) < 1e6) {
				double start_angle = last_th;  // 上一帧的角度
				double end_angle = th;         // 当前帧的角度
				start_angle = normalize_theta(start_angle);
				end_angle = normalize_theta(end_angle);
				
				// 使用圆弧积分公式
				if(fabs(angular) > 1e-6) {
					// 对于小角度，使用泰勒展开提高精度
					if(fabs(angular) < 0.1) {
						double sin_theta = angular - angular*angular*angular/6.0 + angular*angular*angular*angular*angular/120.0;
						double cos_theta = 1.0 - angular*angular/2.0 + angular*angular*angular*angular/24.0;
						
						// 使用泰勒展开的圆弧积分
						x += r * (sin_theta * cos(start_angle) + cos_theta * sin(start_angle) - sin(start_angle));
						y += r * (cos(start_angle) - cos_theta * cos(start_angle) + sin_theta * sin(start_angle));
					} else {
						// 大角度使用精确三角函数
						double sin_start = sin(start_angle);
						double cos_start = cos(start_angle);
						double sin_end = sin(end_angle);
						double cos_end = cos(end_angle);
						
						// 圆弧积分：x += r * (sin(end) - sin(start)), y += r * (cos(start) - cos(end))
						x += r * (sin_end - sin_start);
						y += r * (cos_start - cos_end);
					}
				} else {
					// 极小角度近似
					double direction = th + 0.5 * angular;
					direction = normalize_theta(direction);
					x += linear * cos(direction);
					y += linear * sin(direction);
				}
			} else {
				// 回退到线性近似
				double direction = th + 0.5 * angular;
				direction = normalize_theta(direction);
				x += linear * cos(direction);
				y += linear * sin(direction);
			}
		} else {
			// 纯直线运动，使用简单的线性积分
			double direction = th + 0.5 * angular;
			direction = normalize_theta(direction);
			x += linear * cos(direction);
			y += linear * sin(direction);
		}
		
		odom_quat = current_quat;
		
		// 保存当前角度作为下一帧的last_th
		last_th = th;

		// odom_msg.header.stamp=rclcpp::Time(RCL_SYSTEM_TIME);
		odom_msg.pose.pose.position.x = x;
		odom_msg.pose.pose.position.y = y;
		odom_msg.pose.pose.position.z = 0.0;
		odom_msg.pose.pose.orientation.x = odom_quat.x();
		odom_msg.pose.pose.orientation.y = odom_quat.y();
		odom_msg.pose.pose.orientation.z = odom_quat.z();
		odom_msg.pose.pose.orientation.w = odom_quat.w();		
		odom_msg.pose.covariance[0] = 0.01;
		odom_msg.pose.covariance[7] = 0.01;
		odom_msg.pose.covariance[14] = 1000000;
		odom_msg.pose.covariance[21] = 1000000;
		odom_msg.pose.covariance[28] = 1000000;
		odom_msg.pose.covariance[35] = 0.001;

		odom_msg.twist.twist.linear.x = vx;
		odom_msg.twist.twist.linear.y = vy;
		odom_msg.twist.twist.linear.z = 0.0;
		odom_msg.twist.twist.angular.x = 0.0;
		odom_msg.twist.twist.angular.y = 0.0;
		odom_msg.twist.twist.angular.z = v_th;
		odom_msg.twist.covariance[0] = 0.0001;
		odom_msg.twist.covariance[7] = 0.0001;
		odom_msg.twist.covariance[35] = 1000;
		odom_msg.twist.covariance[14] = 1000000;
		odom_msg.twist.covariance[21] = 1000000;
		odom_msg.twist.covariance[28] = 1000000;
		
		// 包装速度消息用于发布
		current_velocity_msg_.linear.x = vx;
		current_velocity_msg_.linear.y = vy;
		current_velocity_msg_.linear.z = 0.0;
		current_velocity_msg_.angular.x = 0.0;
		current_velocity_msg_.angular.y = 0.0;
		current_velocity_msg_.angular.z = v_th;
		velocity_msg_available_ = true;
	}
	return odom_msg;
}		

                    					
void MotorDrive::resetOdometry(){
    // Lock to protect odometry state variables
    std::lock_guard<std::mutex> lock(odometry_mutex_);
    
    // Zero pose and twist state
    x = 0.0;
    y = 0.0;
    th = 0.0;
    last_th = 0.0;  // 重置上一帧角度
    odom_quat.setRPY(0,0,0);
    // 重置四元数状态为单位四元数
    current_quat = tf2::Quaternion(0, 0, 0, 1);
    // Force next decoder call to treat upcoming encoder values as baseline
    m_iLeftEncoderReading = -1;
    m_iRightEncoderReading = -1;
    last_time = s_throttle_clock.now();
	// 重置角速度滤波状态
	vth_filter_initialized_ = false;
	vth_prev_filtered_ = 0.0;
	// 重置线速度滤波状态
	vx_filter_initialized_ = false;
	vx_prev_filtered_ = 0.0;
	// 重置速度消息状态
	velocity_msg_available_ = false;
}

bool MotorDrive::getCurrentVelocity(geometry_msgs::msg::Twist& velocity_msg){
    std::lock_guard<std::mutex> lock(odometry_mutex_);
    if(velocity_msg_available_){
        velocity_msg = current_velocity_msg_;
        return true;
    }
    return false;
}

// 基于速度指令积分计算里程计：使用最近一次dt，将线速度与角速度积分
nav_msgs::msg::Odometry MotorDrive::decoder(const geometry_msgs::msg::Twist& velocity_cmd){
    nav_msgs::msg::Odometry odom_msg;

    // 保护与时间步长
    rclcpp::Time now = s_throttle_clock.now();
    static rclcpp::Time last = now;
    double dt = (now - last).nanoseconds() * 1e-9;
    last = now;
    if(dt <= 0.0){
        return odom_msg;
    }

    // 线速度与角速度（与cmdCallback一致，使用x和z）
    double vx = velocity_cmd.linear.x;
    double vth = velocity_cmd.angular.z;

    // 与编码器路径一致的积分与状态
    // 复用全局状态变量 x, y, th, current_quat
    // 小角度处理与死区
    const double vth_deadband = 0.01;
    if (fabs(vth) < vth_deadband) {
        vth = 0.0;
    }

    double linear = vx * dt;
    double angular = vth * dt;

    // 四元数更新
    if(fabs(angular) > 1e-8){
        tf2::Quaternion delta_quat;
        delta_quat.setRPY(0, 0, angular);
        current_quat = current_quat * delta_quat;
        current_quat.normalize();
    }

    // 提取姿态角
    tf2::Matrix3x3 rot_matrix(current_quat);
    double roll, pitch, yaw;
    rot_matrix.getRPY(roll, pitch, yaw);
    th = yaw;

    // 位置积分（考虑旋转影响）
    double direction = th + 0.5 * angular;
    direction = normalize_theta(direction);
    x += linear * cos(direction);
    y += linear * sin(direction);

    // 组织里程计消息
    odom_quat = current_quat;
    odom_msg.pose.pose.position.x = x;
    odom_msg.pose.pose.position.y = y;
    odom_msg.pose.pose.position.z = 0.0;
    odom_msg.pose.pose.orientation.x = odom_quat.x();
    odom_msg.pose.pose.orientation.y = odom_quat.y();
    odom_msg.pose.pose.orientation.z = odom_quat.z();
    odom_msg.pose.pose.orientation.w = odom_quat.w();

    odom_msg.twist.twist.linear.x = vx;
    odom_msg.twist.twist.linear.y = 0.0;
    odom_msg.twist.twist.angular.z = vth;

    return odom_msg;
}

void MotorDrive::cmdCallback(geometry_msgs::msg::Twist::SharedPtr msg){
	// Rate limiting to prevent overwhelming the system
	rclcpp::Time current_time = s_throttle_clock.now();
	if(last_velocity_cmd_time_.nanoseconds() > 0) {
		double dt = (current_time - last_velocity_cmd_time_).nanoseconds() * 1e-9;
		if(dt < min_cmd_interval_) {
			// Skip this command if too frequent
			return;
		}
	}
	last_velocity_cmd_time_ = current_time;
	
	RobotVelocity robot_vel;
	if(lose_pose == true && begin_lost_pose_ == true){
		robot_vel.linear_vel_x = 0.0;
		robot_vel.angular_vel_z = 0.0;   
	}
	else if(current_stop_status == 1){
		robot_vel.linear_vel_x = 0.0;
		robot_vel.angular_vel_z = 0.0;   
	}
	else{	
		robot_vel.linear_vel_x = msg->linear.x;
		robot_vel.angular_vel_z = msg->angular.z;   
	}
	setRobotVelocity(robot_vel);
}

