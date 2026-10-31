"""Independent sound-effect node. Errors here never stop robot motion."""
import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

from .audio_events import AUDIO_QOS
from .audio_player import AudioPlayer,load_sounds
from .common import STATE_QOS,finite,param,publish


class AudioNode(Node):
    def __init__(self):
        super().__init__('cyberdog_audio')
        self.enabled=param(self,'audio_enabled',True)
        directory=param(self,'audio_directory',str(Path(get_package_share_directory('cyberdog'))/'audio'))
        backend=param(self,'audio_backend','auto');server=param(self,'audio_server','')
        device=param(self,'audio_device','');volume=finite(param(self,'audio_volume',.7),0.,1.,'audio volume')
        self.status=self.create_publisher(String,'/cyberdog/audio_status',STATE_QOS)
        self.player=None;self.error='';self.sequences={};self.last_status=None
        if self.enabled:
            try:self.player=AudioPlayer(load_sounds(directory),backend=backend,server=server,
                device=device,volume=volume,report=self.problem)
            except (OSError,ValueError,KeyError) as exc:self.problem(str(exc))
        self.create_subscription(String,'/cyberdog/audio_event',self.receive,AUDIO_QOS)
        self.create_timer(.025,self.tick)
        self.tick()

    def problem(self,message):
        self.error=message;self.get_logger().warning('音频不可用：'+message)

    def receive(self,msg):
        if self.player is None:return
        try:
            d=json.loads(msg.data)
            if not isinstance(d,dict) or d.get('schema_version')!=1:return
            session=d['session'];sequence=d['sequence'];events=d['events'];replace=d['replace']
            stamp=finite(d['stamp'],0.,1e12,'audio stamp')
            if (not isinstance(session,str) or not session or len(session)>64
                    or type(sequence) is not int or sequence<=self.sequences.get(session,-1)
                    or type(replace) is not bool or not isinstance(events,list) or len(events)>4
                    or any(not isinstance(e,str) or e not in self.player.sounds for e in events)):return
            age=self.get_clock().now().nanoseconds*1e-9-stamp
            if age<-.2 or (age>3. and events):return
            if len(self.sequences)>=32 and session not in self.sequences:self.sequences.pop(next(iter(self.sequences)))
            self.sequences[session]=sequence
            self.player.submit(events,replace)
        except (ValueError,TypeError,KeyError):return

    def tick(self):
        if self.player is not None:self.player.tick()
        state=dict(enabled=self.enabled,available=self.player is not None,
            playing=self.player.event if self.player else None,
            error=self.player.last_error if self.player else self.error)
        if state!=self.last_status:
            self.last_status=state;publish(self.status,state)

    def destroy_node(self):
        if self.player is not None:self.player.close()
        return super().destroy_node()


def main():
    import rclpy
    from rclpy.executors import ExternalShutdownException
    rclpy.init();node=None
    try:node=AudioNode();rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
