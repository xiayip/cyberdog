"""Observed, verified Rebot mode transitions; existing controller settings untouched."""
import time
from .settling import SettlingWindow
from copy import deepcopy
from action_msgs.msg import GoalStatusArray
from control_msgs.msg import JointTrajectoryControllerState
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.qos import qos_profile_action_status_default, qos_profile_sensor_data
from rebot_arm_hmi_bridge.core import BASE, CHAINS, KNOWN, inspect, plan
from rebot_arm_hmi_bridge.protocol import Mode

IMP = 'rebot_arm_rs_impedance_controller'
JTC = 'rebot_arm_rs_impedance_trajectory_controller'
POS = 'rebot_arm_rs_trajectory_controller'

# These preflight failures describe transient settling, not an ownership,
# active-action or controller-layout fault. Callers may wait within a deadline.
JOINTS_SETTLING = 'fresh stationary joints required before mode switching'
JTC_SETTLING = 'active JTC is not observably settled'
SETTLING_REASONS = frozenset((JOINTS_SETTLING, JTC_SETTLING))


def controller_names(prefix):
    if prefix not in ('rebot_arm_rs', 'zephyr_arm'):
        raise ValueError('unsupported controller_prefix')
    return {name: name.replace('rebot_arm_rs', prefix, 1) for name in {BASE, *KNOWN}}


def canonical_controllers(controllers, prefix):
    mapping = controller_names(prefix)
    if prefix == 'rebot_arm_rs':
        return controllers
    reverse = {actual: canonical for canonical, actual in mapping.items()}
    result = deepcopy(controllers)
    for c in result:
        if c.name in mapping:
            raise ValueError('mixed Rebot and Zephyr controller profiles')
        c.name = reverse.get(c.name, c.name)
        c.claimed_interfaces = [reverse.get(i.split('/')[0], i.split('/')[0]) +
                                ('/' + i.split('/', 1)[1] if '/' in i else '')
                                for i in c.claimed_interfaces]
    return result


def observed(controllers, prefix='rebot_arm_rs'):
    controllers = canonical_controllers(controllers, prefix)
    active={c.name for c in controllers if c.state=='active' and c.name in KNOWN}
    if active=={IMP}:
        # Reuse the original unknown-owner/base-controller inspection by presenting
        # the direct-servo chain as a complete impedance chain for validation only.
        from copy import deepcopy
        checked=deepcopy(controllers)
        jtc=next((c for c in checked if c.name==JTC),None)
        if jtc is None or jtc.state!='inactive':
            raise ValueError('servo requires configured inactive impedance JTC')
        jtc.state='active'
        inspect(checked)
        return 'servo'
    mode,_=inspect(controllers)
    return {Mode.IMPEDANCE:'impedance',Mode.POSITION:'position',Mode.DAMPING:'damping'}[mode]


def switch_plan(controllers, target, prefix='rebot_arm_rs'):
    mapping = controller_names(prefix)
    activate, deactivate = _canonical_switch_plan(canonical_controllers(controllers, prefix), target)
    return [mapping[n] for n in activate], [mapping[n] for n in deactivate]


def _canonical_switch_plan(controllers,target):
    current=observed(controllers)
    targets={'position':list(CHAINS[Mode.POSITION]),'impedance':list(CHAINS[Mode.IMPEDANCE]),'damping':list(CHAINS[Mode.DAMPING]),'servo':[IMP]}
    if target not in targets:
        raise ValueError('unsupported target mode')
    if current==target:
        return [],[]
    if current!='servo' and target!='servo':
        return plan(controllers,{'position':Mode.POSITION,'damping':Mode.DAMPING,'impedance':Mode.IMPEDANCE}[target])
    wanted=targets[target]
    old={'servo':[IMP],'position':list(CHAINS[Mode.POSITION]),
         'damping':list(CHAINS[Mode.DAMPING]),'impedance':list(CHAINS[Mode.IMPEDANCE])}[current]
    states={c.name:c.state for c in controllers}
    if any(states.get(n) not in ('active','inactive') for n in wanted):
        raise ValueError('target chain unavailable')
    return [n for n in wanted if n not in old], [n for n in reversed(old) if n not in wanted]


class ControllerModes:
    def __init__(self,node,motion):
        self.node,self.motion=node,motion
        self.prefix=motion.prefix
        names=controller_names(self.prefix)
        self.imp,self.jtc,self.pos=names[IMP],names[JTC],names[POS]
        self.mock=motion.backend=='mock'
        self.mode='impedance' if self.mock else 'unknown'
        self.busy,self.phase,self.cancelled=False,'idle',False
        self.refused=False  # last request failed its preflight: nothing was switched
        self.future,self.callback=None,None
        self.deadline=0.
        self.quiet_since=None
        self.status={name:False for name in (self.jtc,self.pos)}
        self.matched={name:None for name in (self.jtc,self.pos)}
        self.telemetry={name:(0.,False) for name in (self.jtc,self.pos)}
        self.settling={name:(SettlingWindow(),SettlingWindow()) for name in (self.jtc,self.pos)}
        self.owned_position_publisher=False
        self.subs=[]
        for name in (self.jtc,self.pos):
            self.subs.append(node.create_subscription(GoalStatusArray,'/'+name+'/follow_joint_trajectory/_action/status',
                lambda msg,n=name:self.on_status(n,msg),qos_profile_action_status_default))
            self.subs.append(node.create_subscription(JointTrajectoryControllerState,'/'+name+'/controller_state',
                lambda msg,n=name:self.on_state(n,msg),qos_profile_sensor_data))
        manager=node.get_parameter('controller_manager').value
        self.switch=node.create_client(SwitchController,manager+'/switch_controller')
        self.list=node.create_client(ListControllers,manager+'/list_controllers')

    def on_status(self,name,msg):
        self.status[name]=any(s.status in (0,1,2,3) for s in msg.status_list)

    def on_state(self,name,msg):
        import math
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        age=self.node.get_clock().now().nanoseconds*1e-9-stamp
        ref_window, feedback_window=self.settling[name]
        now=time.monotonic()
        valid=False
        try:
            if stamp and not -.2<=age<=.5:raise ValueError('stale controller state')
            indices=[msg.joint_names.index('joint'+str(i)) for i in range(1,7)]
            ref=[msg.reference.positions[i] for i in indices]
            feedback=[msg.feedback.positions[i] for i in indices]
            rv=[msg.reference.velocities[i] for i in indices]
            # The commissioned position JTC exports position-only feedback.
            # Its motion evidence comes from positions plus the independent
            # fresh joint-state window required by guard(). Impedance still
            # requires its advertised velocity feedback.
            fv=None if name==self.pos and not msg.feedback.velocities else [msg.feedback.velocities[i] for i in indices]
            rq=ref_window.update(now,ref,rv,stamp)
            fq=feedback_window.update(now,feedback,fv,stamp)
            valid=rq and fq and all(math.isfinite(v) and abs(v)<.03 for v in rv)
        except (ValueError,IndexError):
            ref_window.reset();feedback_window.reset()
        self.telemetry[name]=(now,valid)

    def guard(self,controllers):
        return self.check({c.name for c in controllers if c.state=='active'})

    def would_pass(self):
        """Local-only dry run of guard() for the chain this node last observed; sends nothing."""
        if self.mock:return ''
        # Only the JTC chains export the telemetry guard() demands; damping and servo have none active.
        active={'impedance':{self.jtc},'position':{self.pos},'damping':set(),'servo':set()}.get(self.mode)
        if active is None:return 'controller mode not observed'
        return self.check(active)

    def check(self,active):
        now=time.monotonic()
        if now-self.motion.pose_at>.5 or not self.motion.stationary or self.quiet_since is None or now-self.quiet_since<.5:
            return JOINTS_SETTLING
        if self.motion.busy or any(self.status.values()):
            return 'trajectory still active'
        if self.node.count_publishers('/'+self.imp+'/commands')>1:
            return 'another direct impedance publisher exists'
        for name in (self.jtc,self.pos):
            if self.node.count_publishers('/'+name+'/joint_trajectory')>int(name==self.pos and self.owned_position_publisher):
                return 'another direct trajectory publisher exists'
            if name in active:
                at,stationary=self.telemetry[name]
                matched=self.matched[name]
                if now-at>.5 or not stationary or matched is None or now-matched<1.:
                    return JTC_SETTLING
        return ''

    def request(self,target,callback):
        if self.busy:
            return False,'mode transition unresolved'
        if not self.mock and not self.motion.enabled:
            return False,'motion_enabled is false'
        if not self.mock and (not self.list.service_is_ready() or not self.switch.service_is_ready()):
            return False,'controller manager unavailable'
        self.busy,self.cancelled,self.target,self.callback=True,False,target,callback
        self.refused=False
        self.phase='mock' if self.mock else 'preflight'
        self.deadline=time.monotonic()+(.1 if self.mock else 2.)
        if not self.mock:
            self.future=self.list.call_async(ListControllers.Request())
        return True,'accepted'

    def complete(self,ok,message):
        callback,self.callback=self.callback,None
        self.busy,self.phase,self.future=False,'idle',None
        if callback:
            callback(ok and not self.cancelled,message if not self.cancelled else 'mode transition interrupted')

    def cancel(self):
        self.cancelled=True
        # A switching service cannot be cancelled. Retain the gate until verified.

    def tick(self):
        now=time.monotonic()
        if now-self.motion.pose_at>.5 or not self.motion.stationary:
            self.quiet_since=None
        elif self.quiet_since is None:
            self.quiet_since=now
        for name in (self.jtc,self.pos):
            matched=self.node.count_publishers('/'+name+'/follow_joint_trajectory/_action/status')>0
            if not matched:self.matched[name]=None
            elif self.matched[name] is None:self.matched[name]=now
        if not self.busy:return
        if self.phase=='mock':
            if now>=self.deadline:
                if not self.cancelled:self.mode=self.target
                self.complete(True,'mode observed')
            return
        if self.phase=='quarantine':return
        if not self.future.done():
            if now>=self.deadline:
                self.phase='quarantine'
                if self.callback:
                    callback,self.callback=self.callback,None
                    callback(False,'controller service outcome unknown; verify controllers before restart')
            return
        try:
            result=self.future.result()
            if self.phase=='preflight':
                self.mode=observed(result.controller,self.prefix)
                if self.cancelled:
                    self.complete(False,'cancelled before switch');return
                reason=self.guard(result.controller)
                if reason:raise ValueError(reason)
                activate,deactivate=switch_plan(result.controller,self.target,self.prefix)
                if not activate and not deactivate:
                    self.complete(True,'mode already active');return
                request=SwitchController.Request()
                request.activate_controllers,request.deactivate_controllers=activate,deactivate
                request.strictness=2
                request.activate_asap=False
                request.timeout.sec=2
                self.phase,self.deadline='switch',now+3.
                self.future=self.switch.call_async(request)
            elif self.phase=='switch':
                if not result.ok:raise ValueError('controller switch rejected; verify observed state')
                self.phase,self.deadline='verify',now+2.
                self.future=self.list.call_async(ListControllers.Request())
            else:
                self.mode=observed(result.controller,self.prefix)
                self.complete(self.mode==self.target,'verified mode: '+self.mode)
        except Exception as exc:
            if self.phase in ('switch','verify'):
                self.phase='quarantine'
                callback,self.callback=self.callback,None
                if callback:callback(False,'mode outcome unresolved: '+str(exc))
            else:
                self.refused=True
                self.complete(False,str(exc))
