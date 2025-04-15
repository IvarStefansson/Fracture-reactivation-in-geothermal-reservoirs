#!/bin/bash
cd $HOME
NCP_DIR=$HOME/ncp
if [ ! -d "$NCP_DIR" ]; then
    git clone https://github.com/jwboth/ncp-contact-mechanics.git $NCP_DIR
    cd $NCP_DIR
else
    cd $NCP_DIR
    git pull
fi
pip install -e .