# Armie 音频包（奶音 · 哼唧）

给机器小狗的 12 个事件音效 + 6 个动作 × 3 种心情的回应短句，全部是**同一只小狗、同一次录音**里的真实发声（Freesound / johnnypanic「Puppy」pack，728022–728030，CC0）。

## 格式

- WAV，48kHz，单声道，16bit
- 事件音效峰值 −6 dBFS，回应短句峰值 −3 dBFS
- 每条都是完整的一声，只在静音处切，**不变调**

## 命名

```
event_<事件>_v<1|2>.wav      12 个事件各两条变体，播的时候随机取一条
phrase_<动作>_<心情>[_alt].wav   拼好的整句，_alt 是同一种心情的另一种说法
```

## 事件对照

| 文件 | 含义 |
|---|---|
| event_happy | 开心 |
| event_excited | 兴奋 |
| event_curious | 好奇 |
| event_question | 疑问（上扬的问号音） |
| event_sad | 委屈 |
| event_blue | 难过（长、低） |
| event_needy | 撒娇 |
| event_angry | 生气（尖、短） |
| event_pet | 被摸（轻声） |
| event_startle | 受惊（尖、急） |
| event_bored | 无聊（低声嘟囔） |
| event_alert | 警觉（短、平） |

## 短句对照

| 动作 | 心情 | 文件 |
|---|---|---|
| pet 摸头 | happy / neutral / sad | phrase_pet_happy 等 |
| poke 戳一下 | 同上 | phrase_poke_* |
| play 扔球 | 同上 | phrase_play_* |
| drag 被拖动 | 同上 | phrase_drag_* |
| idle 没人理（自言自语，最长 5 秒出头） | 同上 | phrase_idle_* |
| alert 有人靠近 | 同上 | phrase_alert_* |

每个「动作 × 心情」都有主版本和 `_alt` 两种说法，随机挑一种播即可；心情怎么定（精力/亲密度阈值）不在音频包的范围里。
