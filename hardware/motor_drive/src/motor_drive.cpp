#include "motor_drive.hpp"
#include "crc.hpp"
#include "type.hpp"
//确定是否为速度模式
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
//the current odom
double x = 0;
double y = 0;
double th = 0;

bool MotorDrive::serial_init(const std::string serial_port_name, const int serial_baudrate){
	if(serial_port_name.empty() || serial_baudrate != 19200){  // 115200 just
		RCLCPP_ERROR(rclcpp::get_logger("Serial"), "serial port or baudrate is error");
		return false;
	}

	try{
		this->setPort(serial_port_name);
		this->setBaudrate(serial_baudrate);
		serial::Timeout to = serial::Timeout::simpleTimeout(1000); // timeout
		this->setTimeout(to);	
		this->open();
	}
	catch(serial::IOException& e){
		RCLCPP_ERROR(rclcpp::get_logger("Serial"), "Unable to open port");
		return false;	
	}
	
	if(this->isOpen()){
		RCLCPP_INFO(rclcpp::get_logger("Serial"), "Serial Port initialized");
		return true;	
	}
	else{
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
	this->write(m_uEnableMotor,12);                     
    rclcpp::sleep_for(std::chrono::milliseconds(20)); // 10ms
	return true;
}

void MotorDrive::disable_motor(){
	this->write(m_uDisableMotor, 12);
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

	this->write(setting_speed,12);
	rclcpp::sleep_for(std::chrono::milliseconds(10));
}

int MotorDrive::getEncoderInformation(int &left_encoder, int &right_encoder){
	this->write(m_uReadEncoder,8);
	// there maybe need to sleep some time ....
	rclcpp::sleep_for(std::chrono::milliseconds(10));
	int p = this->available();
	std_msgs::msg::UInt8MultiArray msg;
	// RCLCPP_INFO(rclcpp::get_logger("motor"), "encoderinformation: %d,", p);
	if(p){
		this->read(msg.data, p);
		
		int data_size=msg.data.size();
		int returned_value=-1;
		for(int i=0;i<data_size;i++)
		{
			if(i+9<data_size){
				if(msg.data[i+0]==0x02 && msg.data[i+1]==0x43 && msg.data[i+2]==0x50 && msg.data[i+3]==0x04 && msg.data[i+4]==0x51 && msg.data[i+5]==0x04)
				{
					left_encoder=msg.data[i+6]*256+msg.data[i+7];
					right_encoder=msg.data[i+8]*256+msg.data[i+9];
		
					returned_value=returned_value+1;
				}
			}
		}
		return returned_value;
	}
	else{
		return -1;	
	}
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

nav_msgs::msg::Odometry MotorDrive::decoder(int leftcoder, int rightcoder){
	nav_msgs::msg::Odometry odom_msg;
	if(m_iLeftEncoderReading<0 || m_iRightEncoderReading <0){
		m_iLeftEncoderReading = leftcoder;
		m_iRightEncoderReading = rightcoder;
		last_time = rclcpp::Time(RCL_SYSTEM_TIME);
	}
	else{
		current_time = rclcpp::Time(RCL_SYSTEM_TIME);
		double dt = (current_time - last_time).nanoseconds() * 1e-9;
		last_time = current_time;

		// get the encoder difference
		int left_diff = leftcoder-m_iLeftEncoderReading;
		int right_diff = rightcoder - m_iRightEncoderReading;
		left_diff = -left_diff;  // reversal

		// normalize the encoder difference
		left_diff = normalize_encoder_diff(left_diff, m_iEncoderResolution);
		right_diff = normalize_encoder_diff(right_diff, m_iEncoderResolution);
		m_iLeftEncoderReading = leftcoder;
		m_iRightEncoderReading = rightcoder;
		
		//get the distance and rotation inferred by the encoder
		double linear = 0.5*(2.0*M_PI*m_dLeftWheelRadius*left_diff/m_iEncoderResolution) + 
						0.5*(2.0*M_PI*m_dRightWheelRadius*right_diff/m_iEncoderResolution);
		double angular = (2.0*M_PI*m_dRightWheelRadius*right_diff/m_iEncoderResolution - 
						2.0*M_PI*m_dLeftWheelRadius*left_diff/m_iEncoderResolution)/m_dWheelDistance;

		/* double last_x = x;
		double last_y = y;
		double last_th = th; */

		//get the velocity
		double vx=linear/dt;
		double vy=0;
		double v_th=angular/dt;
		bool use_rad = true;
		if(fabs(v_th) > 0.065 && use_rad==true){
			double r = vx/v_th;
			double direction_yaw = th+angular;
			direction_yaw = normalize_theta(direction_yaw);
			x = x+r*(sin(direction_yaw) - sin(th));
			y = y-r*(cos(direction_yaw) - cos(th));
		}
		else{
			double direction = th+0.5*angular;
			direction = normalize_theta(direction);
			x = x+linear*cos(direction);
			y = y+linear*sin(direction);	
		}
		th = th+angular;
		th = normalize_theta(th);

		odom_quat.setRPY(0,0,th);

		// odom_msg.header.stamp=rclcpp::Time(RCL_SYSTEM_TIME);
		odom_msg.pose.pose.position.x = x;
		odom_msg.pose.pose.position.y = y;
		odom_msg.pose.pose.position.z = 0.0;
		odom_msg.pose.pose.orientation.x = odom_quat.x();
		odom_msg.pose.pose.orientation.y = odom_quat.y();
		odom_msg.pose.pose.orientation.z = odom_quat.z();
		odom_msg.pose.pose.orientation.w = odom_quat.w();		
		odom_msg.pose.covariance[0] = 0.001;
		odom_msg.pose.covariance[7] = 0.001;
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
	}
	return odom_msg;
}	

                    					
void MotorDrive::cmdCallback(geometry_msgs::msg::Twist::SharedPtr msg){
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
		robot_vel.linear_vel_x = -msg->linear.x;
		robot_vel.angular_vel_z = msg->angular.z;   
	}
	setRobotVelocity(robot_vel);
}

