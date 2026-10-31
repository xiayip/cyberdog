import json
import math

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

EXPRESSIONS = {'idle', 'happy', 'curious', 'thinking', 'sleepy', 'confused', 'stopped'}
STATE_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL)


def param(node, name, default):
    # Restart to change transport or motion policy; do not pretend live updates apply.
    return node.declare_parameter(
        name, default, ParameterDescriptor(read_only=True)).value


def publish(pub, value):
    pub.publish(String(data=json.dumps(value, ensure_ascii=False, allow_nan=False)))


def finite(value, minimum, maximum, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(label)
    return float(value)


def run(factory):
    import rclpy
    import signal
    from rclpy.executors import ExternalShutdownException
    from rclpy.signals import SignalHandlerOptions
    # Keep the ROS context alive long enough to cancel this client's own goal.
    # launch can forward a second SIGINT while cleanup is already running.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    node = None
    try:
        node = factory()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        if node is not None:
            if rclpy.ok() and hasattr(node, 'prepare_shutdown'):
                node.prepare_shutdown()
                import time
                until = time.monotonic() + 2.
                while node.motion.busy and rclpy.ok() and time.monotonic() < until:
                    rclpy.spin_once(node, timeout_sec=.05)
                if node.motion.busy:
                    node.get_logger().error('Motion result unresolved at shutdown; verify controller state')
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
