"""Original synthesized dance groove, rendered away from robot control callbacks."""
import math
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

RATE=48000
EVENT='M1_dance_music'


def music_spec(value):
    if not isinstance(value,dict):raise ValueError('invalid dance music timing')
    result={key:float(value[key]) for key in ('bpm','duration','core_start','core_end')}
    if not all(math.isfinite(x) for x in result.values()):raise ValueError('non-finite music timing')
    if not (30.<=result['bpm']<=150. and .1<=result['duration']<=60.
            and 0.<=result['core_start']<result['core_end']<=result['duration']):
        raise ValueError('dance music timing outside bounds')
    return result


def synthesize(spec):
    """16-beat C/Am/F/G phrase: kick, backbeat, hats, bass and a bell melody.

    The dance's core_start anchors beat zero; entry and return receive quieter
    accompaniment. No recordings, network access or external synthesizer needed.
    """
    spec=music_spec(spec);duration=spec['duration'];beat=60./spec['bpm']
    mix=np.zeros(math.ceil(duration*RATE),dtype=np.float32)
    rng=np.random.default_rng(20261031)
    def put(at,sound,gain=1.):
        start=round(at*RATE);offset=max(0,-start);start=max(0,start)
        count=min(len(sound)-offset,len(mix)-start)
        if count>0:mix[start:start+count]+=sound[offset:offset+count]*gain
    def clock(seconds):return np.arange(max(1,round(seconds*RATE)))/RATE
    t=clock(.30)
    kick=np.sin(2*np.pi*(48*t+100*.022*(1-np.exp(-t/.022))))*np.exp(-t/ .07)
    t=clock(.17);noise=rng.uniform(-1,1,len(t))
    snare=(.65*(noise-np.roll(noise,1))+.35*np.sin(2*np.pi*185*t))*np.exp(-t/.035)
    t=clock(.055);noise=rng.uniform(-1,1,len(t))
    hat=(noise-np.roll(noise,1))*np.exp(-t/.012)
    def tone(note,seconds,bell=False):
        t=clock(seconds);freq=440.*2**((note-69)/12.)
        attack=np.minimum(1.,t/.008)
        tail=np.minimum(1.,np.maximum(0.,seconds-t)/.03)
        if bell:
            signal=(np.sin(2*np.pi*freq*t)+.23*np.sin(2*np.pi*freq*2*t))*np.exp(-t/.13)
        else:
            signal=(np.sin(2*np.pi*freq*t)+.22*np.sin(2*np.pi*freq*2*t))*np.exp(-t/.28)
        return signal*attack*tail
    roots=(36,33,29,31)
    melody=(72,None,76,79,76,None,74,76,
            69,None,72,76,79,76,72,None,
            69,None,72,77,76,None,72,69,
            67,None,71,74,79,74,71,74)
    first=math.floor(-spec['core_start']/beat)
    last=math.ceil((duration-spec['core_start'])/beat)
    for n in range(first,last+1):
        at=spec['core_start']+n*beat
        core=spec['core_start']<=at<spec['core_end']
        gain=1. if core else .38
        put(at,kick,.60*gain)
        if n%4 in (1,3):put(at,snare,.24*gain)
        root=roots[(n//4)%4]
        put(at,tone(root,beat*.6),.24*gain)
        if n%2:put(at+beat*.75,tone(root+7,beat*.22),.16*gain)
        for half in (0,1):
            hit=at+half*beat/2
            put(hit,hat,(.07 if half==0 else .045)*gain)
            note=melody[(2*n+half)%len(melody)]
            if note is not None:put(hit,tone(note,min(.35,beat*.45),True),.13*gain)
    # A short arrival accent makes the trajectory entry audible immediately.
    put(0.,kick,.16)
    t=np.arange(len(mix))/RATE
    fade=np.minimum(1.,t/.04)*np.minimum(1.,np.maximum(0.,duration-t)/.35)
    return np.asarray(np.tanh(mix)*fade*.82*32767,dtype='<i2')


def write_wav(path,pcm):
    with wave.open(str(path),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())


class DanceMusic:
    """One pending score. Stops invalidate rendering before it can start playback."""
    def __init__(self,player,volume=.7,report=lambda message:None,clock=time.monotonic):
        self.player=player;self.volume=volume;self.report=report;self.clock=clock
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='dance_music')
        self.directory=tempfile.TemporaryDirectory(prefix='cyberdog-music-')
        self.future=None;self.started=0.

    def start(self,spec,elapsed=0.):
        spec=music_spec(spec)
        self.cancel();self.player.stop()
        self.started=self.clock()-elapsed
        self.future=self.pool.submit(synthesize,spec)

    def cancel(self):
        future,self.future=self.future,None
        if future is not None:future.cancel()

    def tick(self):
        if self.future is None or not self.future.done() or self.player.process is not None:return
        future,self.future=self.future,None
        try:
            pcm=future.result()
            # Skip synthesis/transport time so the score follows the dance's
            # timeline instead of starting late from its first beat.
            offset=max(0,round((self.clock()-self.started)*RATE))
            if offset>=len(pcm):
                self.report('舞蹈音乐生成/等待播放器超时，本轮已结束');return
            pcm=pcm[offset:];path=Path(self.directory.name)/'dance.wav'
            write_wav(path,pcm)
            self.player.sounds[EVENT]=(path,len(pcm)/RATE)
            self.player.volumes[EVENT]=self.volume
            self.player.submit([EVENT],replace=True)
            self.player.tick()
        except Exception as exc:self.report('舞蹈音乐不可用：'+str(exc))

    def close(self):
        self.cancel();self.pool.shutdown(wait=False,cancel_futures=True)
        self.directory.cleanup()
