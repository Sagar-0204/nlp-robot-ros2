#!/usr/bin/env bash
# Stop every leftover Gazebo / ROS 2 process from earlier runs (this project or others),
# so a new launch starts from a clean slate. Safe to run any time.
#   bash ~/nlp_robot_ws/src/nlp_robot/tools/clean.sh

patterns=(
  "gz sim" "gzserver" "gzclient" "ros_gz_bridge" "ros_gz_image" "parameter_bridge" "image_bridge"
  "ros_gz_sim/create" "robot_state_publisher" "component_container" "nav2_" "rviz2"
  "slam_toolbox" "cartographer" "map_server" "amcl"
  "nlp_robot/lib/nlp_robot" "ros2 launch" "ros2 run"
)

echo "Stopping leftover processes..."
for p in "${patterns[@]}"; do
  pids=$(pgrep -f -- "$p" | grep -v "^$$\$" || true)
  if [ -n "$pids" ]; then
    echo "  $p: $(echo $pids | wc -w) process(es)"
    kill $pids 2>/dev/null
  fi
done
sleep 2
# Anything that ignored SIGTERM
for p in "${patterns[@]}"; do
  pids=$(pgrep -f -- "$p" | grep -v "^$$\$" || true)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
done

if command -v ros2 >/dev/null; then
  ros2 daemon stop >/dev/null 2>&1
  ros2 daemon start >/dev/null 2>&1
  echo "ROS 2 daemon restarted."
fi
echo "Clean. Now: source install/setup.bash && ros2 launch nlp_robot bringup.launch.py"
