import os

# Every test runs away from the operator's robot domain, including colcon test.
os.environ['ROS_DOMAIN_ID'] = '187'
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
os.environ.pop('ROS_DISCOVERY_SERVER', None)
os.environ.pop('ROS_STATIC_PEERS', None)
