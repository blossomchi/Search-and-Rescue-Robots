from setuptools import setup
from glob import glob
pkg = "sar_gazebo"
setup(name=pkg, version="0.1.0", packages=[pkg],
      data_files=[("share/ament_index/resource_index/packages", ["resource/" + pkg]), ("share/" + pkg, ["package.xml"]),
                  ("share/" + pkg + "/launch", glob("launch/*.py")), ("share/" + pkg + "/worlds", glob("worlds/*.sdf"))],
      install_requires=["setuptools"], zip_safe=True, entry_points={"console_scripts": ["replay_node = sar_gazebo.replay_node:main"]})
