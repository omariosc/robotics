# Utility script to help run worlds more easily.
# This assumes you have already run the ros alias command from a *linux* system.

# Copy and Paste the following in the shell.

source $HOME/.bashrc

export TURTLEBOT_GAZEBO_WORLD_FILE=$HOME/catkin_ws/src/group_project/world/project.world

# Launch gazebo
roslaunch turtlebot_gazebo turtlebot_world.launch