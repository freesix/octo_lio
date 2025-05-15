from setuptools import find_packages, setup
import os
import glob
package_name = 'web_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), ['launch/demo_run.launch.py']),
        (os.path.join('share', package_name, 'launch'), ['launch/demo_runSC.launch.py']),
        (os.path.join('share', package_name, 'launch'), ['launch/demo_runLocalize.launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='freesix',
    maintainer_email='freesix@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
