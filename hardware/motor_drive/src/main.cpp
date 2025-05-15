#include "motor_drive.hpp"
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>

MotorDrive motordrive = MotorDrive();

std::string VOLECITY_TOPIC;
std::string ODOM_TOPIC;


class MotorNode : public rclcpp::Node{
public:
    MotorNode() : Node("robot_motor"){
        this->declare_parameter<std::string>("serial_port_name", "/dev/ttyUSB1");
        this->declare_parameter<int>("serial_baudrate",  115200);
        this->declare_parameter<double>("right_wheel_radius", 0.0845);
        this->declare_parameter<double>("left_wheel_radius", 0.0845);
        this->declare_parameter<double>("wheel_distance", 0.3848);
        this->declare_parameter<int>("encoder_resolution", 5600);
        this->declare_parameter<bool>("begin_lost_pose", false);
        this->declare_parameter<std::string>("odom_frame_id", "odom");
        this->declare_parameter<std::string>("base_frame_id", "base_link");

        this->declare_parameter<std::string>("velocity_cmd_topic", "velocity_cmd");
        VOLECITY_TOPIC = this->get_parameter("velocity_cmd_topic").as_string();
        sub_vel = this->create_subscription<geometry_msgs::msg::Twist>("/cmd_vel", 10,
            [this](const geometry_msgs::msg::Twist::SharedPtr msg){motordrive.cmdCallback(msg);});
        // sub_vel = this->create_subscription<geometry_msgs::msg::Twist>("/cmd_vel", 10,
            // std::bind(&MotorNode::cmdCallback, this, std::placeholders::_1));
        this->declare_parameter<std::string>("odom_publisher", "odom_diff"); 
        ODOM_TOPIC = this->get_parameter("odom_publisher").as_string(); 
        // odom_pub = this->create_publisher<nav_msgs::msg::Odometry>(ODOM_TOPIC, 10);
    }


    void Run(){
        //Serial initional
        motordrive.serial_init(this->get_parameter("serial_port_name").as_string(),
            this->get_parameter("serial_baudrate").as_int());
        motordrive.motor_init(this->get_parameter("right_wheel_radius").as_double(),
            this->get_parameter("left_wheel_radius").as_double(),
            this->get_parameter("wheel_distance").as_double(),
            this->get_parameter("encoder_resolution").as_int());
        std::string odom_link = this->get_parameter("odom_frame_id").as_string();
        std::string base_link = this->get_parameter("base_frame_id").as_string();

        nav_msgs::msg::Odometry odom_msg;
        //Enable motor
        MotorDrive::RobotVelocity ZeroVel;
        ZeroVel.linear_vel_x=0.0;
        ZeroVel.angular_vel_z=0.0;
        motordrive.enable_motor();
        motordrive.setRobotVelocity(ZeroVel);

        rclcpp::Rate rate(10);  // 设置循环频率（10Hz）
        while(rclcpp::ok()){
            int left_encoder = 0;
            int right_encoder = 0;
            int p = motordrive.getEncoderInformation(left_encoder, right_encoder);
            if(p>=0){
                odom_msg = motordrive.decoder(left_encoder, right_encoder);     
            }
            odom_msg.header.stamp = this->now();
            odom_msg.header.frame_id = odom_link;
            odom_msg.child_frame_id = base_link;
            // RCLCPP_INFO_STREAM(this->get_logger(), "v: " << odom_msg.twist.twist.linear.x << " " <<
                // odom_msg.twist.twist.linear.y << " " << odom_msg.twist.twist.linear.z); 
            
            // odom_pub->publish(odom_msg);

            rate.sleep();  // 控制循环频率
        }
        motordrive.disable_motor();
    }
 
private:
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_vel;
};


int main(int argc, char** argv){    
    rclcpp::init(argc, argv);

    auto node = std::make_shared<MotorNode>();

    // 将 Run 函数放到另一个线程中运行
    std::thread run_thread([&node](){
        node->Run();
    });

    rclcpp::spin(node);  // 主线程处理回调

    run_thread.join();  // 等待 Run 线程完成
    rclcpp::shutdown();
    return 0;
}