"""Bounded, interruptible WAV playback; no ROS or robot control dependencies."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import wave


def load_sounds(directory):
    root=Path(directory).resolve()
    entries=json.loads((root/'验证记录.json').read_text(encoding='utf-8'))
    mapping=json.loads((root/'映射清单.json').read_text(encoding='utf-8'))['byId']
    sounds={}
    for item in entries:
        event=item['event'];filename=item['file'];record=mapping[event]
        # colcon --symlink-install links each installed WAV to the source tree.
        # Restrict manifest names, while allowing those verified file symlinks.
        if (not isinstance(filename,str) or Path(filename).name!=filename
                or Path(filename).suffix!='.wav' or event in sounds or record['mode']!='sound'):
            raise ValueError('invalid audio manifest entry: '+event)
        path=(root/filename).resolve()
        if path.name!=record['filename'] or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('audio hash/name mismatch: '+event)
        with wave.open(str(path),'rb') as wav:
            duration=wav.getnframes()/wav.getframerate()
            if ((wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),wav.getcomptype())!=(48000,1,2,'NONE')
                    or not 0<duration<=5. or abs(duration-item['seconds'])>.001):
                raise ValueError('invalid audio format/duration: '+event)
        sounds[event]=(path,duration)
    if set(sounds)!={k for k,v in mapping.items() if v['mode']=='sound'}:
        raise ValueError('audio verification list is incomplete')
    return sounds


class AudioPlayer:
    def __init__(self,sounds,*,backend='auto',server='',device='',volume=.7,
                 clock=time.monotonic,popen=subprocess.Popen,report=lambda message:None):
        self.sounds=sounds;self.clock=clock;self.popen=popen;self.report=report
        self.volume=volume;self.volumes={};self.device=device;self.env=os.environ.copy()
        if server:self.env['PULSE_SERVER']=server
        if backend not in ('auto','paplay'):raise ValueError('unknown audio player')
        # Prefer the desktop's default output (including Bluetooth) over a raw
        # ALSA card. No shell invocation and no global mixer-volume changes.
        self.backend='paplay' if shutil.which('paplay') else None
        if self.backend is None:raise ValueError('install pulseaudio-utils (paplay)')
        self.queue=deque(maxlen=4);self.process=None;self.event=None
        self.deadline=0.;self.kill_at=None;self.retry_at=0.;self.last_error=''

    def submit(self,events,replace=False):
        if replace:self.stop()
        now=self.clock()
        for event in events:
            if event not in self.sounds:continue
            if (event==self.event and self.process is not None and self.process.poll() is None
                    and self.kill_at is None) or any(e==event for e,_ in self.queue):continue
            self.queue.append((event,now+3.))

    def stop(self):
        self.queue.clear()
        if self.process is not None and self.process.poll() is None and self.kill_at is None:
            self.process.terminate();self.kill_at=self.clock()+.2

    def command(self,event):
        path,_=self.sounds[event]
        args=['paplay','--client-name=cyberdog','--stream-name='+event,'--latency-msec=50',
              '--volume='+str(round(65536*self.volumes.get(event,self.volume)))]
        if self.device:args+=['--device='+self.device]
        return args+[str(path)]

    def fail(self,message):
        self.last_error=message;self.report(message)
        self.queue.clear();self.retry_at=self.clock()+5.

    def tick(self):
        now=self.clock()
        if self.process is not None:
            code=self.process.poll()
            if code is None:
                if self.kill_at is not None and now>=self.kill_at:self.process.kill()
                elif self.kill_at is None and now>self.deadline:
                    self.stop();self.fail('audio player timed out')
                return
            interrupted=self.kill_at is not None
            self.process=None;self.event=None;self.kill_at=None
            if code and not interrupted:self.fail(f'audio player exited {code}; check output/server')
        if now<self.retry_at:return
        while self.queue:
            event,expires=self.queue.popleft()
            if now>expires:continue
            try:
                self.process=self.popen(self.command(event),env=self.env,stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                self.event=event;self.deadline=now+self.sounds[event][1]+3.
                self.last_error=''
            except OSError as exc:self.fail(str(exc))
            return

    def close(self):
        self.stop()
        if self.process is not None:
            try:self.process.wait(timeout=.5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=.5)
            self.process=None;self.event=None
