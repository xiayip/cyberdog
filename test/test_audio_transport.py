"""Use actual DDS publication, with a fake speaker process and mock robot."""
from pathlib import Path
import time

from cyberdog.audio_node import AudioNode
from cyberdog.audio_player import AudioPlayer
from cyberdog.behavior import Cyberdog
from test_behavior import graph,until


def test_latched_startup_reset_and_stop_reach_independent_audio_node(monkeypatch):
    processes=[]
    class Process:
        def __init__(self):self.code=None;self.started=time.monotonic()
        def poll(self):
            if self.code is None and time.monotonic()-self.started>.1:self.code=0
            return self.code
        def terminate(self):self.code=-15
        def kill(self):self.code=-9
        def wait(self,timeout=None):self.code=0;return self.code
    def popen(cmd,**kwargs):
        p=Process();processes.append((cmd,p));return p
    monkeypatch.setattr('cyberdog.audio_node.AudioPlayer',lambda sounds,**kw:AudioPlayer(sounds,popen=popen,**kw))
    directory=Path(__file__).parents[1]/'audio'
    with graph((Cyberdog,AudioNode),[f'audio_directory:={directory}']) as ([g,a],e):
        # The subscriber is created after startup publication. DDS durability
        # must deliver the greeting without any extra behavior reset/motion.
        until(e,lambda:len(processes)==1)
        assert '--stream-name=A1' in processes[0][0]
        assert g.phase=='startup' and not g.robot.busy
        g.command('reset');until(e,lambda:g.state=='idle')
        until(e,lambda:a.sequences.get(g.audio.session,0)==g.audio.sequence)
        assert any('--stream-name=E4' in cmd for cmd,_ in processes)
        g.command('stop')
        until(e,lambda:a.sequences.get(g.audio.session,0)==g.audio.sequence and a.player.process is None)
        assert not a.player.queue and g.state=='paused'
