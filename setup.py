from setuptools import setup, find_packages
import os

# get the current path and read the content in requirements.txt
here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'requirements.txt'), 'r', encoding="utf-8") as f:
    requirements = f.read().splitlines()

# define setup 
setup(
    name='WildfireThomas',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    author='IMPERIAL COLLEGE LONDON MSC EDSML - GROPU THOMAS',
    description='A package for wildfire prediction and data assimilation',
    long_description=open(os.path.join(here, 'README.md'), encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ese-msc-2023/acds3-wildfire-thomas',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
