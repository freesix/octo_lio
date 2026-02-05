# 前言
这是在实机上做工程化实验，所以没有仿真环境，使用的硬件环境也各不相同，各自适配驱动和urdf。
# 编译
ROS2 package编译方式，缺依赖自行下载，ubuntu22.04和humble自带源下载的依赖就行，没有特殊要求。
# 运行
* 建图
```
ros2 launch web_control demo_runSC.launch.py
```
会自动在主目录下新建一个LOAM文件夹保存点云地图、轨迹、平面点云、角点点云、SC描述子等等。
```
ros2 run nav2_map_saver map_saver_cli -t <octomap投影栅格地图话题名字> -f <想要保存的用于导航的pgm地图> --occ 0.55 --free 0.49
```
将保存的pgm地图图片和yaml文件保存到nav2_bringup的maps文件夹下，然后配置launch文件中查找路径
*导航
```
ros2 launch web_control demo_runLocalize.launch.py
``` 
此时便开始重定位，看打印的输出，不成功就遥控或者手动推动底盘去不停重定位，直到成功
```
ros2 launch nav2_bringup fd_robot.launch.py
```
启动导航栈，当然如果传感器话题和硬件不同，自行配置fd_params.yaml中参数
```
ros2 launch nav2_bringup fd_map_server.launch.py
```
启动nav2的地图服务，这个是做2d工程落地时地图续建、建图和导航模式动态切换遗留的写法，完全可以合在前一步中一起启动
```
ros2 launch nav2_bringup rviz_launch.py
```
启动rviz，在rviz上点击就可以开始导航了

# 底盘算法实现
* 定位建图：lio-sam，适配mid360
* 重定位：采用SC算法
* 地图保存：在可视化地图线程中保存，包含SC描述子、全局地图、关键帧点云、位姿图、角点点云、平面点点云、轨迹图、各时刻转换关系等
* 导航地图生成：采用Octomap投影生成
* 导航栈：navigation2
# 2026-2-5
## 已经完成
理想环境下完全可以全流程的建图、重定位、导航，是一套完善但不完美的移动底盘算法流程。
## 存在问题
1、lio_sam运行效率问题，其实lio_sam占用计算资源不算夸张，但内存分配和代码逻辑属实垃圾（虽然我也垃圾），是一个相当不错的学术产物。总比一些论文发表了“代码延期”。[MARG : MAstering Risky Gap Terrains for Legged Robots with Elevation Mapping](https://astrorix.github.io/MARG/)，催更一下。

2、mid360的非重复扫描方式，便宜是便宜，没有多线重复式扫描投影生成的地图好，特别是遇到实际动态场景，比如商场这种人流量大的地方，建的图能用才有鬼，所以强调理想环境。

3、算法本身，SC重定位有一点延迟，重定位成功位姿和当前位姿有时有一点偏差。（但可以表明已经在一个SC位姿附近，重新单独启动ros2 launch web_control demo_runLocalize.launch.py就可以了）

4、重定位成功后，按流程应该slam前端匹配、后端优化、回环检测、全局匹配定位这几者之间做成模块式互相配合，但全局匹配定位这个lio-sam中没有的，本代码中做了，但经常会出现错误，导致odom->map这个关系出错，故暂时屏蔽，仅依靠重定位来一次寻找odom->map的关系，后续导航位姿仅依靠slam。

# 2025-5-29
## 存在的问题 
1、动态点云会导致Octomap在投影生成导航栅格地图上存在伪影，nav2全局路径规划时会受到这些伪影影响，导致全局路径规划不是最优，或者规划失败无法找到到达目标点的路径

2、重定位效率不高，SC匹配算法，只有当机器人当前帧扫描和以有SC算子的扫描帧的位姿在附近才能匹配到

3、避障效果不行：不加入避障逻辑可以正常导航，加入避障逻辑会有‘摇头’、避障不流畅、狭小过道无法通过、障碍物误检测等问题，需要仔细调参
