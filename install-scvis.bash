#!/bin/bash

# Upgrade scvis to support tensorflow 2
cd scvis/lib/scvis
tf_upgrade_v2 --intree . --outtree . --print_all
cd ../..

# install
python setup.py install
cd ..
