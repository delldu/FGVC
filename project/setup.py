"""Setup."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="video_patch",
    version="1.0.0",
    author="Dell Du",
    author_email="18588220928@163.com",
    description="video patch package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/delldu/FGVC.git",
    packages=["video_patch"],
    package_data={"video_patch": ["models/video_patch.pth"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 1.9.0",
        "torchvision >= 0.10.0",
        "Pillow >= 7.2.0",
        "numpy >= 1.19.5",
        "einops >= 0.3.0",
        "redos >= 1.0.0",
        "todos >= 1.0.0",
    ],
)
