"""Independent sound and music node. Errors here never stop robot motion."""
import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

from .audio_events import AUDIO_QOS
from .audio_player import AudioPlayer,load_sounds
from .dance_music import DanceMusic,music_spec,EVENT as MUSIC_EVENT
from .common import STATE_QOS,finite,param,publish


class AudioNode(Node):
    def __init__(self):
        super().__init__('cyberdog_audio')
        self.enabled=param(self,'audio_enabled',True)
        directory=param(self,'audio_directory',str(Path(get_package_share_directory('cyberdog'))/'audio'))
        backend=param(self,'audio_backend','auto');server=param(self,'audio_server','')
        device=param(self,'audio_device','');volume=finite(param(self,'audio_volume',.7),0.,1.,'audio volume')
        self.music_enabled=param(self,'dance_music_enabled',True)
        if type(self.music_enabled) is not bool:raise ValueError('dance_music_enabled must be boolean')
        music_volume=finite(param(self,'dance_music_volume',.7),0.,1.,'dance music volume')
        self.status=self.create_publisher(String,'/cyberdog/audio_status',STATE_QOS)
        self.player=None;self.music=None;self.error='';self.sequences={};self.last_status=None
        if self.enabled:
            try:self.player=AudioPlayer(load_sounds(directory),backend=backend,server=server,
                device=device,volume=volume,report=self.problem)
            except (OSError,ValueError,KeyError) as exc:self.problem(str(exc))
        if self.player is not None and self.music_enabled:
            try:self.music=DanceMusic(self.player,volume=music_volume,report=self.problem)
            except (OSError,ValueError) as exc:self.problem(str(exc))
        self.get_logger().info(f'音频节点 PID={os.getpid()}，舞蹈音乐'
            f'{"已就绪" if self.music is not None else "未启用/不可用"}，音量={music_volume:.2f}')
        self.create_subscription(String,'/cyberdog/audio_event',self.receive,AUDIO_QOS)
        self.create_timer(.025,self.tick)
        self.tick()

    def problem(self,message):
        self.error=message;self.get_logger().warning('音频不可用：'+message)

    def receive(self,msg):
        if self.player is None:return
        music=None
        try:
            d=json.loads(msg.data)
            if not isinstance(d,dict) or d.get('schema_version')!=1:return
            session=d['session'];sequence=d['sequence'];events=d['events'];replace=d['replace']
            music=d.get('music')
            stamp=finite(d['stamp'],0.,1e12,'audio stamp')
            if (not isinstance(session,str) or not session or len(session)>64
                    or type(sequence) is not int or sequence<=self.sequences.get(session,-1)
                    or type(replace) is not bool or not isinstance(events,list) or len(events)>4
                    or any(not isinstance(e,str) or e not in self.player.sounds or e==MUSIC_EVENT for e in events)):return
            if music is not None:
                if not replace or events:return
                music=music_spec(music)
            age=self.get_clock().now().nanoseconds*1e-9-stamp
            if age<-.2 or (age>3. and (events or music is not None)):
                if music is not None:self.problem(f'舞蹈音乐事件已过期/时钟不一致：age={age:.3f}s')
                return
            if len(self.sequences)>=32 and session not in self.sequences:self.sequences.pop(next(iter(self.sequences)))
            self.sequences[session]=sequence
            if replace and self.music is not None:self.music.cancel()
            if music is not None:
                self.player.stop()
                if self.music is not None:
                    self.get_logger().info(f'收到舞蹈音乐：{music["bpm"]:.1f} BPM，{music["duration"]:.1f} 秒')
                    self.music.start(music,max(0.,age))
                else:self.problem('收到舞蹈音乐，但音乐功能未启用或初始化失败')
                return
            self.player.submit(events,replace)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            if music is not None:self.problem('舞蹈音乐请求无效：'+str(exc))

    def tick(self):
        if self.player is not None:self.player.tick()
        if self.music is not None:self.music.tick()
        if (self.player is not None and self.player.event is not None and not self.player.last_error
                and self.player.event!=(self.last_status or {}).get('playing')):self.error=''
        state=dict(enabled=self.enabled,available=self.player is not None,
            music_enabled=self.music_enabled,music_available=self.music is not None,pid=os.getpid(),
            playing=self.player.event if self.player else None,
            error=self.error or (self.player.last_error if self.player else ''))
        if state!=self.last_status:
            old=self.last_status or {}
            if state['playing']==MUSIC_EVENT and old.get('playing')!=MUSIC_EVENT:
                self.get_logger().info('舞蹈音乐开始播放：M1_dance_music')
            elif old.get('playing')==MUSIC_EVENT and state['playing']!=MUSIC_EVENT:
                self.get_logger().info('舞蹈音乐播放结束/已停止')
            self.last_status=state;publish(self.status,state)

    def destroy_node(self):
        if self.player is not None:self.player.close()
        if self.music is not None:self.music.close()
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
