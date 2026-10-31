"""Wait until the arm, guard and wrist camera are usable, then exit 0.

bringup.launch.py starts the game only after this succeeds. Read-only: lists
controllers, the node graph and one frame per camera stream; commands nothing.
"""
import time

import rclpy
from controller_manager_msgs.srv import ListControllers
from rcl_interfaces.srv import GetParameters
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from .common import param
from .controller_modes import observed, controller_names, KNOWN

# Reuse any verified controller chain: the reset button switches it to position.
ACTIVE = ('joint_state_broadcaster', 'zephyr_arm_mit_controller', 'zephyr_arm_gripper_controller')
LOADED = [controller_names('zephyr_arm')[name] for name in KNOWN]
GUARD = '/zephyr_arm_impedance_trajectory_guard'
COLOR, DEPTH = '/camera/color/image_raw/compressed', '/camera/depth/image_raw/compressedDepth'
IMPEDANCE = '/zephyr_arm_impedance_controller'


def missing(controllers, nodes, frames):
    state = {c.name: c.state for c in controllers}
    out = [f'{name} active' for name in ACTIVE if state.get(name) != 'active']
    out += [f'{name} loaded' for name in LOADED if state.get(name) not in ('active','inactive')]
    try:observed(controllers,'zephyr_arm')
    except ValueError as exc:out.append('controller chain: '+str(exc))
    out += [GUARD] * (GUARD not in nodes)
    return out + [f'{topic} frame' for topic in (COLOR, DEPTH) if topic not in frames]


def main():
    rclpy.init()
    node = rclpy.create_node('cyberdog_startup_check')
    log = node.get_logger()
    try:
        deadline = time.monotonic() + param(node, 'timeout_sec', 120.)
        frames = set()
        for topic in (COLOR, DEPTH):
            node.create_subscription(CompressedImage, topic, lambda _, t=topic: frames.add(t), qos_profile_sensor_data)
        lister = node.create_client(ListControllers, '/controller_manager/list_controllers')
        controllers, pending, todo = [], None, ['startup']
        while todo:
            if time.monotonic() > deadline:
                raise RuntimeError('not ready: ' + ', '.join(todo))
            if pending is None and lister.service_is_ready():
                pending = lister.call_async(ListControllers.Request())
            rclpy.spin_once(node, timeout_sec=.5)
            if pending is not None and pending.done():
                controllers, pending = pending.result().controller, None
            nodes = {f'{ns.rstrip("/")}/{name}' for name, ns in node.get_node_names_and_namespaces()}
            todo = missing(controllers, nodes, frames)
        params = node.create_client(GetParameters, f'{IMPEDANCE}/get_parameters')
        names = ['stiffness', 'reference.position_rate_limit']
        if params.wait_for_service(timeout_sec=5.):
            future = params.call_async(GetParameters.Request(names=names))
            rclpy.spin_until_future_complete(node, future, timeout_sec=5.)
            if future.done():
                for name, value in zip(names, future.result().values):
                    log.info(f'{IMPEDANCE} {name} = {list(value.double_array_value)}')
        log.info('arm controllers, trajectory guard and wrist camera ready')
    except Exception as exc:
        log.error(str(exc))
        raise SystemExit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
