# slam-tutorial

## Install ROS1::Noetic in Ubuntu 20.04
https://velog.io/@deep-of-machine/ROS-ROS1-%EC%84%A4%EC%B9%98-Ubuntu20.04-ROS-Noetic

## Dataset
LiDAR: hdl [LINK](https://zenodo.org/record/6960371).
CAMERA: HILTI[LINK](https://hilti-challenge.com/dataset-2021.html)
```
ros2 bag play -s rosbag_v2   uzh_tracking_area_run2.bag
```

## ROS2 archive
https://github.com/stars/Taeyoung96/lists/ros2

## install rosbag2
```
sudo apt install ros-foxy-rosbag2*
```
## RUN bag file (https://storage.googleapis.com/hilti_challenge/uzh_tracking_area_run2.bag)
```
ros2 bag  play uzh_tracking_area_run2.bag -s rosbag_v2 --read-ahead-queue-size 1000  --topics /alphasense/cam0/image_raw /alphasense/cam1/image_raw
```

## 사전 준비물
1. Rosfoxy, Rosnoetic 설치
2. `bashrc`에 아래 명령어 추가
```
alias rosnoetic='source /opt/ros/noetic/setup.bash'
alias rosfoxy='source /opt/ros/foxy/setup.bash && sour    ce ~/gcamp_ros2_ws/install/local_setup.bash && source     ~/slam-tutorial_ws/install/local_setup.bash'

alias cba='colcon build --symlink-install'
alias cbp='colcon build --symlink-install --packages-s    elect'
alias killg='killall -9 gzserver && killall -9 gzclien    t && killall -9 rosmaster'
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=~/gcamp_ros2_ws
```
3. Terminal 1: `rosnoetic` 실행 후 `rosfoxy` 실행
4. Terminal 1: bag 파일 play
5. Terminal 2: `rosfoxy` 입력 후 `slam-tutorial` build
```
cbp slam-tutorial
```
6. `slam-tutorial` 실행
```
ros2 run slam-tutorial main_node
```

## To-do
1. Convert KITTI stereo to `.bag` format.
2. Implement `PnP` in `call_back` function to obtain R, t, and publish R, t.
3. Visualize published R, t via rviz
