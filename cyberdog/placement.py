"""Taught JTC placement and optional collision-planned Cartesian placement."""
import math

from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, Constraints, OrientationConstraint, PositionConstraint
from moveit_msgs.srv import GetMotionPlan
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint

from .common import finite
from .game_core import path_points
from .motion import MAX_PATH_SPEED, MAX_PATH_ACCELERATION, bounded_start

DEFAULT_PLACE_JOINTS = [0.2, 0.635, 0.199, -0.599, -0.285, 0.273]


def joint_placement_points(start,target,speed,acceleration):
    """Bounded JTC interpolation to the operator-confirmed taught release pose."""
    finite(speed,.01,MAX_PATH_SPEED,'placement speed')
    finite(acceleration,.05,MAX_PATH_ACCELERATION,'placement acceleration')
    points=[]
    for at,q in path_points(bounded_start(start),[target],speed,1.,acceleration):
        point=JointTrajectoryPoint(positions=q,velocities=[0.]*6,accelerations=[0.]*6)
        point.time_from_start.sec,point.time_from_start.nanosec=divmod(round(at*1e9),1_000_000_000)
        points.append(point)
    return points


def waypoint_pose(waypoint):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = waypoint['position']
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = waypoint['orientation']
    return pose


def placement_waypoints(table, xy, radius, clearance=.01, approach=.16):
    """Place inside the measured patch, above its conservative collision top."""
    finite(radius, .015, .085, 'ball radius')
    finite(clearance, .005, .04, 'placement clearance')
    finite(approach, .08, .25, 'placement approach')
    if (table.header.frame_id != 'base_link' or len(table.primitives) != 1
            or len(table.primitive_poses) != 1 or table.meshes or table.planes):
        raise ValueError('放置需要 base_link 下的桌面盒模型')
    box = table.primitives[0]
    if box.type != SolidPrimitive.BOX or len(box.dimensions) != 3:
        raise ValueError('桌面模型不是有效盒体')
    p, origin = table.primitive_poses[0], table.pose
    for pose in (p, origin):
        q = pose.orientation
        if (max(abs(q.x), abs(q.y), abs(q.z)) > 1e-6
                or not (abs(q.w) < 1e-6 or abs(abs(q.w)-1.) < 1e-6)):
            raise ValueError('不支持旋转的桌面盒模型')
    center = [getattr(p.position, k)+getattr(origin.position, k) for k in ('x', 'y', 'z')]
    if (len(xy) != 2 or not all(math.isfinite(v) for v in [*xy, *center, *box.dimensions,
            p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w,
            origin.orientation.x,origin.orientation.y,origin.orientation.z,origin.orientation.w])
            or min(box.dimensions) <= 0.):
        raise ValueError('无效的桌面放置坐标')
    margin = radius+.02
    if any(abs(v-c)+margin > d/2 for v,c,d in zip(xy, center, box.dimensions)):
        raise ValueError('放置点太靠近桌面边缘或在已测桌面之外')
    top = center[2]+box.dimensions[2]/2
    z = top+radius+clearance
    yaw = math.atan2(xy[1], xy[0])
    # A vertical tool at the raised pose exceeds J4's range on this arm.
    # Tilt outwards 0.7 rad from vertical; radial yaw keeps J6 near zero.
    a,b = yaw/2,(math.pi-.7)/2
    q = [-math.sin(a)*math.sin(b), math.cos(a)*math.sin(b),
         math.sin(a)*math.cos(b), math.cos(a)*math.cos(b)]
    return [dict(stage=stage, position=[*xy, z+dz], orientation=q[:], gripper=grip)
            for stage,dz,grip in [('transfer', approach, 'hold'), ('place', 0., 'open')]]


def held_ball(radius):
    """Local planning attachment only; never alters the shared planning scene."""
    finite(radius, .015, .085, 'ball radius')
    attached = AttachedCollisionObject(link_name='omni_picker_tcp')
    attached.object = CollisionObject(id='cyberdog_held_ball', operation=CollisionObject.ADD)
    attached.object.header.frame_id = 'omni_picker_tcp'
    attached.object.pose.orientation.w = 1.
    attached.object.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[radius])]
    pose = Pose(); pose.orientation.w = 1.
    attached.object.primitive_poses = [pose]
    attached.touch_links = ['omni_picker_tcp']+[
        'omni_picker_hand_'+side+part+'_Link'
        for side in ('narrow', 'wide') for part in ('1', '2', '3', '4', '_loop')]
    return attached


def transfer_request(state, waypoint, floor):
    """Turn above the table before the separate vertical descent."""
    request = GetMotionPlan.Request(); req = request.motion_plan_request
    req.group_name = 'zephyr_arm_tcp'; req.pipeline_id = 'ompl'
    req.start_state = state; req.start_state.is_diff = True
    req.num_planning_attempts = 3; req.allowed_planning_time = 4.
    req.max_velocity_scaling_factor = .5; req.max_acceleration_scaling_factor = .5
    pose = waypoint_pose(waypoint)
    position = PositionConstraint(link_name='omni_picker_tcp', weight=1.)
    position.header.frame_id = 'base_link'
    position.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.004]*3)]
    region_pose = Pose(); region_pose.position = pose.position; region_pose.orientation.w = 1.
    position.constraint_region.primitive_poses = [region_pose]
    orientation = OrientationConstraint(link_name='omni_picker_tcp', orientation=pose.orientation,
        absolute_x_axis_tolerance=.01, absolute_y_axis_tolerance=.01, absolute_z_axis_tolerance=.01, weight=1.)
    orientation.header.frame_id = 'base_link'
    req.goal_constraints = [Constraints(position_constraints=[position], orientation_constraints=[orientation])]
    # The carried ball and robot still undergo normal collision checking.
    clearance = PositionConstraint(link_name='omni_picker_tcp', weight=1.)
    clearance.header.frame_id = 'base_link'
    clearance.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[2., 2., 1.5-floor])]
    region_pose = Pose(); region_pose.position.z = (1.5+floor)/2; region_pose.orientation.w = 1.
    clearance.constraint_region.primitive_poses = [region_pose]
    req.path_constraints.position_constraints = [clearance]
    return request
