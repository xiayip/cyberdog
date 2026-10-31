"""Add the measured table to MoveIt as a collision box.

Planning scene only: no controller, action or motor command.
"""
import math

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive

from .common import param


def table_box(plane, lo, hi, object_id):
    if (len(plane) != 3 or len(lo) != 2 or len(hi) != 2
            or not all(math.isfinite(v) for v in [*plane, *lo, *hi])
            or not all(a < b for a, b in zip(lo, hi))):
        raise ValueError('table plane and xy bounds not configured')
    a, b, c = plane
    # Same box as the grasp trials: 10 cm thick, top 1 cm above the highest
    # corner of the fitted plane, covering only the observed xy patch.
    top = max(a*x + b*y + c for x in (lo[0], hi[0]) for y in (lo[1], hi[1])) + .01
    box = CollisionObject(id=object_id, operation=CollisionObject.ADD)
    box.header.frame_id = 'base_link'
    box.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[hi[0]-lo[0], hi[1]-lo[1], .10])]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (lo[0]+hi[0])/2, (lo[1]+hi[1])/2, top-.05
    pose.orientation.w = 1.
    box.primitive_poses = [pose]
    return box


def main():
    rclpy.init()
    node = rclpy.create_node('cyberdog_table_scene')
    try:
        box = table_box(param(node, 'plane_z_equals_ax_by_c', [0., 0., 0.]),
                        param(node, 'xy_min', [0., 0.]), param(node, 'xy_max', [0., 0.]),
                        param(node, 'object_id', 'cyberdog_observed_table'))
        client = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        if not client.wait_for_service(timeout_sec=param(node, 'timeout_sec', 30.)):
            raise RuntimeError('/apply_planning_scene unavailable')
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        request.scene.world.collision_objects = [box]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.)
        if not future.done() or not future.result().success:
            raise RuntimeError('planning scene rejected the table box')
        node.get_logger().info(f'added {box.id}: top z={box.primitive_poses[0].position.z+.05:.3f} m')
    except Exception as exc:
        node.get_logger().error(str(exc))
        raise SystemExit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
