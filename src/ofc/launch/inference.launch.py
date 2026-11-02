from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ofc',
            executable='tln_inference',
            name='tln_inference',
            output='screen',
            parameters=[{
                'scan_topic': '/scan',
                'drive_topic': '/ackermann_cmd',
                'drive_publish_enabled': False,
            }],
        ),
    ])
