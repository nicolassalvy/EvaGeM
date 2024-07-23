from setuptools import setup, find_packages
import os


def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname)) as fh:
            return fh.read()
    except IOError:
        return ""


requirements = read("requirements.txt").splitlines()


setup(
    name="EvaGeM",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
)
