#!/usr/bin/env bash
# Pre-flight check. Run it from a sourced terminal before launching:
#   cd ~/nlp_robot_ws && source install/setup.bash && bash src/nlp_robot/tools/check_setup.sh
# Every line is OK / WARN / FAIL with a fix.

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=1; }
FAILED=0

echo "== ROS environment"
if [ "$ROS_DISTRO" = "jazzy" ]; then ok "ROS 2 Jazzy sourced"
else fail "ROS 2 Jazzy not sourced (ROS_DISTRO='$ROS_DISTRO') -> source /opt/ros/jazzy/setup.bash"; fi
if ros2 pkg prefix nlp_robot >/dev/null 2>&1; then ok "workspace sourced (nlp_robot found)"
else fail "nlp_robot not found -> cd ~/nlp_robot_ws && colcon build --symlink-install && source install/setup.bash"; fi
if ros2 pkg prefix nl_robot >/dev/null 2>&1; then
  warn "the OLD nl_robot workspace is also sourced in this terminal -> open a fresh terminal"
fi
echo "  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}  (use the same value in every terminal)"

echo "== Simulation packages (stage 2)"
for p in turtlebot3_gazebo turtlebot3_description turtlebot3_teleop ros_gz_sim ros_gz_bridge \
         ros_gz_image nlp_robot_interfaces; do
  if ros2 pkg prefix "$p" >/dev/null 2>&1; then ok "$p"
  else fail "$p missing -> sudo apt install ros-jazzy-turtlebot3 ros-jazzy-turtlebot3-simulations ros-jazzy-ros-gz"; fi
done
echo "== Navigation packages (stage 3)"
for p in nav2_bringup nav2_common turtlebot3_navigation2 rviz2; do
  if ros2 pkg prefix "$p" >/dev/null 2>&1; then ok "$p"
  else warn "$p missing (needed from stage 3) -> sudo apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup"; fi
done

echo "== Leftover processes (cause a blank Gazebo, two maps, jumping localisation)"
check_proc() {  # pattern, description
  local n; n=$(pgrep -fc -- "$1" 2>/dev/null); n=${n:-0}
  if [ "$n" -gt 0 ]; then warn "$n '$2' process(es) already running -> bash src/nlp_robot/tools/clean.sh"
  else ok "no running $2"; fi
}
check_proc "gz sim" "Gazebo"
check_proc "slam_toolbox" "slam_toolbox"
check_proc "component_container" "Nav2 container"
check_proc "rviz2" "RViz"

echo "== Gazebo"
if command -v gz >/dev/null; then ok "gz CLI: $(gz sim --versions 2>/dev/null | head -1)"
else fail "gz CLI not found -> sudo apt install ros-jazzy-ros-gz"; fi

echo "== Perception (stage 7)"
for m in cv2 numpy; do
  if python3 -c "import $m" 2>/dev/null; then ok "python3 module $m"
  else fail "python3 module $m missing -> sudo apt install python3-opencv python3-numpy"; fi
done

echo "== Language backend (stage 4)"
if python3 -c "import requests" 2>/dev/null; then ok "python3 module requests"
else fail "python3 module requests missing -> sudo apt install python3-requests"; fi
MODEL=${NLP_MODEL:-qwen2.5:7b-instruct}
if curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
  if curl -s --max-time 2 http://localhost:11434/api/tags | grep -q "\"$MODEL\""; then
    ok "Ollama running, $MODEL pulled"
  else warn "Ollama running but $MODEL not pulled -> ollama pull $MODEL"; fi
  if command -v nvidia-smi >/dev/null; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$free" ] && echo "  GPU memory free: ${free} MiB (a 7B model needs ~5000 MiB to run fully on the GPU)"
  fi
else warn "Ollama not reachable -> start it (ollama serve), or use the rules backend"; fi
[ -n "$ANTHROPIC_API_KEY" ] && ok "ANTHROPIC_API_KEY set (anthropic backend available)"

echo
if [ "$FAILED" = 1 ]; then echo "Fix the FAIL lines above first."; exit 1; fi
echo "Ready: ros2 launch nlp_robot bringup.launch.py"
