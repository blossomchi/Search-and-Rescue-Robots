import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
def generate_launch_description():
    share = get_package_share_directory("sar_gazebo"); world = os.path.join(share, "worlds", "sar_arena.sdf"); plan = os.path.join(share, "worlds", "plan.json")
    return LaunchDescription([
        ExecuteProcess(cmd=["gz", "sim", "-r", world], output="screen"),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             arguments=["/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist", "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry", "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"]),
        Node(package="sar_gazebo", executable="replay_node", output="screen", parameters=[{"plan": plan, "use_sim_time": True}]),
    ])
