#!/bin/bash
wget https://files.pythonhosted.org/packages/33/bc/fa0b5347139cd9564f0d44ebd2b147ac97c36b2403943dbee8a25fd74012/virtualenv-16.0.0.tar.gz
tar xf virtualenv-16.0.0.tar.gz

# Make sure to install using Python 3, as TensorFlow only provides Python 3 artifacts
python3 virtualenv-16.0.0/virtualenv.py opencvdl
. opencvdl/bin/activate
pip3 install tensorflow-gpu
pip3 install pyqt5
pip3 install matplotlib
pip3 install opencv-contrib-python
pip3 install easygui
deactivate
