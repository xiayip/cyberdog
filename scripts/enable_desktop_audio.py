#!/usr/bin/env python3
"""Run on the desktop host to share its default output with a host-network container."""
import json
import subprocess


def main():
    modules=json.loads(subprocess.check_output(['pactl','--format=json','list','modules'],text=True))
    for module in modules:
        args=module.get('argument','').split()
        if (module['name']=='module-native-protocol-tcp' and 'listen=127.0.0.1' in args
                and 'port=4713' in args):
            print('Desktop audio already available at tcp:127.0.0.1:4713');return
    module=subprocess.check_output(['pactl','load-module','module-native-protocol-tcp',
        'listen=127.0.0.1','port=4713','auth-ip-acl=127.0.0.1'],text=True).strip()
    print('Desktop audio available at tcp:127.0.0.1:4713; module '+module)
    print('Undo with: pactl unload-module '+module)


if __name__=='__main__':main()
