from controller_manager_msgs.msg import ControllerState
from cyberdog.controller_modes import BASE, KNOWN, CHAINS, controller_names
from cyberdog.startup_check import ACTIVE, GUARD, COLOR, DEPTH, missing
from rebot_arm_hmi_bridge.protocol import Mode
import pytest


@pytest.mark.parametrize('mode',[Mode.POSITION,Mode.IMPEDANCE,Mode.DAMPING])
def test_startup_accepts_existing_verified_chain(mode):
    names=controller_names('zephyr_arm')
    rows=[ControllerState(name=names[n],state='active' if n in {BASE,*CHAINS[mode]} else 'inactive') for n in {BASE,*KNOWN}]
    rows += [ControllerState(name=n,state='active') for n in ACTIVE if n!=names[BASE]]
    assert missing(rows,{GUARD},{COLOR,DEPTH})==[]
    rows.append(ControllerState(name='other',state='active',claimed_interfaces=['joint2/position']))
    assert any('controller chain' in x for x in missing(rows,{GUARD},{COLOR,DEPTH}))
