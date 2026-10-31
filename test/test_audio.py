"""Stage mapping, queue cancellation, real ROS transport; never moves hardware."""
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from std_msgs.msg import String

from cyberdog.audio_events import AudioEvents
from cyberdog.audio_node import AudioNode
from cyberdog.audio_player import AudioPlayer,load_sounds
from cyberdog.behavior import Cyberdog
from cyberdog.senses import Senses
from test_behavior import graph,until,spin
from test_grasp_retry import lock_target

DIRECTORY=Path(__file__).parents[1]/'audio'


class Game:
    def __init__(self):
        self.frames=[];self.warnings=[];self.state='paused'
        self.robot=SimpleNamespace(modes=SimpleNamespace(mode='damping'),busy=False,mock=True,
            fresh=lambda:True,current=[0.]*6)
    def declare_parameter(self,name,value,*args):return SimpleNamespace(value=value)
    def create_publisher(self,*args):return SimpleNamespace(publish=lambda msg:self.frames.append(json.loads(msg.data)))
    def get_clock(self):return SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=1000000000))
    def get_logger(self):return SimpleNamespace(warning=self.warnings.append)


def cue_events(game):return [event for frame in game.frames for event in frame['events']]


def state(g,a,state,phase,now=0.,attempt=0):
    g.state=state;a.transition(state,phase,attempt,now=now)


def test_manifest_all_23_files_match_hash_format_duration_and_silent_events_have_no_wav():
    sounds=load_sounds(DIRECTORY)
    assert len(sounds)==23 and not {'A4','C4','E3'}&sounds.keys()


def test_manifest_rejects_changed_audio(tmp_path):
    shutil.copytree(DIRECTORY,tmp_path/'audio')
    path=tmp_path/'audio'/'A1_开机招呼.wav';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash'):load_sounds(tmp_path/'audio')


def test_colcon_symlink_install_can_load_all_verified_audio(tmp_path):
    installed=tmp_path/'share/cyberdog/audio';installed.mkdir(parents=True)
    for p in DIRECTORY.iterdir():
        if p.is_file():(installed/p.name).symlink_to(p.resolve())
    assert len(load_sounds(installed))==23


def test_full_stage_sequence_reset_handshake_grasp_and_completion():
    g=Game();a=AudioEvents(g);a.reset()
    for s,p in [('resetting','waiting_stop'),('transition','settling'),('transition','resetting'),
                ('idle','holding'),('transition','handshake_start'),('transition','settling'),
                ('transition','wrist_lift'),('transition','head_tilt'),('transition','head_looking'),
                ('transition','lifting'),('active','handshake'),('transition','ball_entering'),
                ('active','tracking'),('active','grasp_settling'),('active','grasp_approach'),
                ('active','grasp_closing'),('active','grasp_place'),('active','grasp_releasing'),('paused','complete')]:
        state(g,a,s,p)
    assert cue_events(g)==['A1','E4','E1','E2','A2','B1','B3','B2','B3','C1','C2','C5','C6']
    assert any(f['replace'] and f['events']==[] for f in g.frames)  # C4 clears ongoing tracking audio.
    state(g,a,'paused','complete');assert cue_events(g).count('C6')==1


@pytest.mark.parametrize('away,expected',[(1.,'B1'),(19.9,'B1'),(20.,'W1'),(59.9,'W1'),(60.,'W2'),(300.,'W2'),(301.,'W3')])
def test_idle_wake_routes_without_stacking_touch_cue(away,expected):
    g=Game();a=AudioEvents(g);state(g,a,'idle','holding')
    state(g,a,'transition','handshake_start',away)
    assert g.frames[-1]['events']==[expected] and g.frames[-1]['replace']


def test_idle_sound_is_30_seconds_apart_and_does_not_reset_wake_age():
    g=Game();a=AudioEvents(g);state(g,a,'idle','holding')
    for now in (29.,30.,30.1,59.,60.,60.1):a.tick(now)
    assert cue_events(g).count('A3')==2
    state(g,a,'transition','handshake_start',65.)
    assert g.frames[-1]['events']==['W2']
    state(g,a,'paused','stopped',66.);a.tick(500.)
    assert g.frames[-1]['events']==[]


def test_damping_movement_sound_works_with_auto_gripper_disabled_and_is_throttled():
    g=Game();g.handshake_gripper_enabled=False;a=AudioEvents(g)
    state(g,a,'active','handshake')
    for i in range(31):
        if i>=12:g.robot.current[2]=.1
        a.tick(i*.05)
    assert cue_events(g).count('B4')==1 and 'B5' not in cue_events(g)
    state(g,a,'active','tracking');a.tick(9.)
    assert cue_events(g).count('B4')==1


def test_reacquired_ball_is_debounced_and_never_sounds_during_locked_blind_grasp():
    g=Game();a=AudioEvents(g);state(g,a,'active','tracking')
    for now in (0.,.1,.9,1.5,1.6,6.):a.seen_ball(now)
    assert cue_events(g).count('C3')==2
    state(g,a,'active','grasp_settling');a.seen_ball(10.)
    assert g.frames[-1]['events']==[]


def test_retry_cues_and_taken_bow_use_correct_emotions():
    g=Game();a=AudioEvents(g)
    state(g,a,'active','grasp_retry_wait',attempt=1)
    state(g,a,'active','tracking');state(g,a,'active','grasp_settling',attempt=2)
    state(g,a,'active','grasp_retry_wait',attempt=2)
    state(g,a,'paused','stopped');state(g,a,'transition','ball_taken_bowing')
    assert cue_events(g)==['A1','D1','C2','D4','D2','D3']


def test_audio_publish_failure_disables_sound_without_raising_into_behavior():
    g=Game();a=AudioEvents(g)
    def broken(msg):raise RuntimeError('transport unavailable')
    a.pub.publish=broken;a.emit('C1');a.emit('C2')
    assert not a.enabled and len(g.warnings)==1


class Process:
    def __init__(self):self.code=None;self.terminated=False;self.killed=False
    def poll(self):return self.code
    def terminate(self):self.terminated=True
    def kill(self):self.killed=True;self.code=-9
    def wait(self,timeout=None):
        if self.code is None:raise subprocess.TimeoutExpired('fake',timeout)
        return self.code


def player(monkeypatch):
    monkeypatch.setattr('cyberdog.audio_player.shutil.which',lambda _: '/usr/bin/paplay')
    now=[0.];processes=[];commands=[];errors=[]
    def popen(cmd,**kwargs):
        p=Process();processes.append(p);commands.append((cmd,kwargs));return p
    p=AudioPlayer(load_sounds(DIRECTORY),popen=popen,clock=lambda:now[0],report=errors.append,
        server='tcp:127.0.0.1:4713',volume=.7)
    return p,now,processes,commands,errors


def test_player_never_overlaps_and_replace_cancels_queued_stale_audio(monkeypatch):
    p,now,processes,commands,errors=player(monkeypatch)
    p.submit(['B1','B3']);p.tick();assert p.event=='B1'
    p.submit(['C1'],replace=True);p.tick()
    assert len(processes)==1 and processes[0].terminated
    now[0]=.21;p.tick();assert processes[0].killed
    p.tick();assert len(processes)==2 and p.event=='C1'
    assert not errors and all('B3' not in str(c) for c in commands)
    assert commands[1][1]['env']['PULSE_SERVER']=='tcp:127.0.0.1:4713'
    assert '--volume=45875' in commands[1][0]
    p.close();assert processes[1].killed


def test_repeat_after_stop_can_replay_same_reset_event(monkeypatch):
    p,now,processes,_,_=player(monkeypatch)
    p.submit(['E4']);p.tick();p.submit(['E4'],replace=True)
    processes[0].code=-15;p.tick()
    assert len(processes)==2 and p.event=='E4'
    p.close()


def test_queue_expires_and_device_failure_backoff_drops_old_cues(monkeypatch):
    p,now,processes,_,errors=player(monkeypatch)
    p.submit(['A1']);now[0]=3.1;p.tick();assert not processes
    p.submit(['C1','C2']);p.tick();processes[0].code=1;p.tick()
    assert errors and not p.queue
    p.submit(['B1']);now[0]=7.;p.tick();assert len(processes)==1
    now[0]=8.2;p.tick();assert len(processes)==1
    p.submit(['E4']);p.tick();assert len(processes)==2
    p.close()


def test_hung_player_times_out_and_does_not_hold_the_next_phase_forever(monkeypatch):
    p,now,processes,_,errors=player(monkeypatch)
    p.submit(['A1']);p.tick();now[0]=4.;p.tick()
    assert processes[0].terminated and errors
    now[0]=4.21;p.tick();assert processes[0].killed
    p.tick();assert p.process is None


def test_unavailable_player_reports_installation_error(monkeypatch):
    monkeypatch.setattr('cyberdog.audio_player.shutil.which',lambda _:None)
    with pytest.raises(ValueError,match='install'):AudioPlayer(load_sounds(DIRECTORY))


def test_ros_audio_node_rejects_duplicates_stale_cues_and_honors_late_stop(monkeypatch):
    processes=[]
    def fake(cmd,**kwargs):p=Process();processes.append(p);return p
    monkeypatch.setattr('cyberdog.audio_player.subprocess.Popen',fake)
    # Default arguments bind at import, so replace the construction dependency too.
    monkeypatch.setattr('cyberdog.audio_node.AudioPlayer',lambda sounds,**kw:AudioPlayer(sounds,popen=fake,**kw))
    with graph((AudioNode,),[f'audio_directory:={DIRECTORY}']) as ([a],e):
        stamp=a.get_clock().now().nanoseconds*1e-9
        def send(sequence,events,replace=False,age=0.):
            a.receive(String(data=json.dumps(dict(schema_version=1,session='test',sequence=sequence,
                stamp=stamp-age,events=events,replace=replace))))
        send(1,['A1']);a.tick();assert len(processes)==1
        send(1,['C1']);send(2,['C1'],age=5.);a.tick();assert not a.player.queue
        send(3,[],True,age=5.);assert processes[0].terminated
        send(2,['B1']);assert not a.player.queue
        send(4,['not-an-event']);assert not a.player.queue


def test_behavior_emits_retry_exhaustion_but_fault_and_manual_stop_are_silent():
    with graph() as ([g],e):
        frames=[];g.audio.pub=SimpleNamespace(publish=lambda msg:frames.append(json.loads(msg.data)))
        seq=lock_target(g);seq.attempts=seq.max_attempts;seq.fail('path incomplete',retryable=True)
        assert frames[-1]['events']==['D3'] and g.state=='paused'
        g.command('stop');assert frames[-1]['events']==[]
        g.command('reset');until(e,lambda:g.state=='idle')
        assert any(f['events']==['E4'] for f in frames)
        g.pause('joint feedback stale');assert frames[-1]['events']==[]


def test_real_behavior_flow_publishes_silence_at_lock_and_sound_at_close_completion():
    with graph((Cyberdog,Senses),['mock_ball_visible:=true','ball_still_sec:=0.5']) as ([g,s],e):
        frames=[];g.audio.pub=SimpleNamespace(publish=lambda msg:frames.append((g.phase,json.loads(msg.data))))
        g.command('reset');until(e,lambda:g.phase=='complete',timeout=6.)
        assert any(phase=='grasp_settling' and f['events']==[] and f['replace'] for phase,f in frames)
        assert [event for _,f in frames for event in f['events'] if event in ('C5','C6')]==['C5','C6']
        g.command('stop');assert frames[-1][1]['events']==[]
