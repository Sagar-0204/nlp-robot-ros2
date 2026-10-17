# One object found by the robot's perception, in the map frame.

# Label used in commands, e.g. "red_box"
string label
string color
# Estimated object centre in the global (map) frame
geometry_msgs/PoseStamped pose
# Number of sightings fused into this estimate
int32 observations
# How the latest distance was measured: "lidar" (precise, up to 3.5 m on the TurtleBot3)
# or "camera" (from the box's size in the image; coarser, but works further away)
string range_source
builtin_interfaces/Time last_seen
