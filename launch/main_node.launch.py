import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import xacro

# this is the function launch  system will look for
def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    package_name = "slam-tutorial"

    pkg_path = os.path.join(get_package_share_directory(package_name))



    main_node = Node(
        package='slam-tutorial',
        executable='main',
    )


    # create and return launch description object
    return LaunchDescription(
        [
            main_node
        ]

    )



