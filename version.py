import os
from setuptools_scm import get_version

version = get_version(root=os.path.dirname(os.path.abspath(__file__)))
version = ".".join(version.split(".")[:3])
print(version)
