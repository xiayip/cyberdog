from glob import glob
from setuptools import setup

setup(name='cyberdog', version='0.1.0', packages=['cyberdog'],
    data_files=[('share/ament_index/resource_index/packages',['resource/cyberdog']),
                ('share/cyberdog',['package.xml','README.md']),
                ('share/cyberdog/launch',glob('launch/*.launch.py')),
                ('share/cyberdog/config',glob('config/*')),
                ('share/cyberdog/models',glob('models/*'))],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='Cyberdog developers', maintainer_email='dev@example.com',
    description='Core Zephyr arm pet behaviors', license='Apache-2.0',
    entry_points={'console_scripts':[
        'behavior=cyberdog.behavior:main', 'senses=cyberdog.senses:main',
        'panel=cyberdog.panel:main', 'table_scene=cyberdog.table_scene:main',
        'startup_check=cyberdog.startup_check:main']})
