"""Local desktop operator panel; no HMI display or face graphics."""
import json
import os
from .common import STATE_QOS


def main():
    import tkinter as tk
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    rclpy.init()
    node=Node('cyberdog_panel')
    try:root=tk.Tk()
    except tk.TclError as exc:
        node.get_logger().error(f'cannot open the panel window ({exc}); set DISPLAY or launch with panel:=false')
        node.destroy_node();rclpy.try_shutdown()
        raise SystemExit(1)
    node.get_logger().info(f'panel open on DISPLAY={os.environ.get("DISPLAY","")}')
    root.title('Cyberdog · 桌面宠物')
    root.geometry('640x300')
    label=tk.StringVar(value='等待 Cyberdog 状态…')
    info=tk.StringVar(value='复位 → J3 触碰握手 / 自动追球')
    status_at=[0.]
    phase=['']
    progress=tk.StringVar(value='')
    import time
    def state(msg):
        value=json.loads(msg.data)
        if value.get('schema_version')!=2:return
        status_at[0]=time.monotonic()
        phase[0]=value['phase']
        if phase[0]!='tracking' and not phase[0].startswith('grasp_'):progress.set('')
        label.set(f'{"模拟演示" if value["mock"] else "实机"} | {value["mode"] or "待机"} | {value["phase"]}')
        info.set(value['message']+'\n'+(value['outcome'] or ''))
    sub=node.create_subscription(String,'/cyberdog/state',state,STATE_QOS)
    def tracking(msg):
        data=json.loads(msg.data);ball=data.get('ball',{})
        if phase[0].startswith('grasp_'):
            text=f"抓取尝试 {ball.get('attempt',0)} / {ball.get('max_attempts',3)}"
            if ball.get('retry_pending'):text+=' · 等待停止后重新定位'
            elif ball.get('vision_locked'):text+=' · 已锁定位置，本轮不再依赖视觉'
            elif ball.get('using_frozen_target'):text+=' · 毛球暂不可见，继续抓取已确认位置'
            position=data.get('gripper_state')
            if position is not None:text+=f' · 夹爪实测 {position:.3f}'
            progress.set(text);return
        if phase[0]!='tracking':return
        reason=ball.get('last_reason','')
        messages={'target unavailable':'等待看到毛球','waiting for depth':'等待深度定位',
            'waiting for timestamped TF':'等待空间定位同步','RGB-D/TF sync timeout':'定位同步超时，等待新画面',
            'sphere fit does not reliably describe selected ball':'球面深度不可靠，请调整球的位置'}
        text=f"空间稳定 {ball.get('still_sec',0):.1f} / {ball.get('dwell_sec',1):.1f} 秒"
        if reason:text+=' · '+messages.get(reason,'等待可靠定位')
        else:
            gate_reason=ball.get('still_reason','')
            if gate_reason=='movement_pending':text+=' · 位置短暂跳动，保留计时'
            elif gate_reason=='position_outlier':text+=' · 忽略定位跳点，等待确认'
        reset_reason=ball.get('still_reset_reason','')
        reset_messages={'sample_gap':'定位间断过长','target_moving':'持续移位',
                        'new_sensor_session':'相机检测重新启动'}
        if ball.get('still_sec',0)<.2 and reset_reason in reset_messages:
            text+=' · 重新计时：'+reset_messages[reset_reason]
        progress.set(text)
    node.create_subscription(String,'/cyberdog/tracking_status',tracking,1)
    pub=node.create_publisher(String,'/cyberdog/command',10)
    clients={name:node.create_client(Trigger,'/cyberdog/'+name) for name in ['reset','stop']}
    def send(command):
        if command in clients:
            client=clients[command]
            if not client.service_is_ready():info.set('服务未就绪');return
            future=client.call_async(Trigger.Request())
            def done(f):
                try:info.set(f.result().message)
                except Exception:info.set('请求未完成，请查看状态')
            future.add_done_callback(done)
        else:pub.publish(String(data=command))
    tk.Label(root,text='Cyberdog',font=('',24)).pack(pady=12)
    tk.Label(root,textvariable=label,font=('',13)).pack()
    tk.Label(root,textvariable=info,wraplength=600,height=3).pack()
    tk.Label(root,textvariable=progress,wraplength=600).pack()
    actions=tk.Frame(root);actions.pack(pady=18)
    for i,(name,text) in enumerate([('reset','复位 / 重新开始'),('stop','暂停并保持')]):
        tk.Button(actions,text=text,width=22,height=2,command=lambda n=name:send(n)).grid(row=0,column=i,padx=8)
    tk.Label(root,text='暂停按钮仅取消本程序的动作，不代替硬件急停。').pack(pady=15)
    def poll():
        if rclpy.ok():rclpy.spin_once(node,timeout_sec=0.)
        if status_at[0] and time.monotonic()-status_at[0]>2.:label.set('连接中断：请检查游戏节点')
        root.after(25,poll)
    def close():
        send('stop')
        until=time.monotonic()+.3
        while rclpy.ok() and time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.02)
        root.destroy()
    root.protocol('WM_DELETE_WINDOW',close)
    root.after(25,poll)
    try:root.mainloop()
    except KeyboardInterrupt:pass  # launch Ctrl+C
    finally:node.destroy_node();rclpy.try_shutdown()
