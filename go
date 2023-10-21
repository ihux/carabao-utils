#!/bin/bash
# .go: setup some aliases to be ready to go

export BARC=$HOME/.bash_profile
export REPO=`pwd`
source local/bin/alias.sh

if [ -d venv ]; then
  source venv/bin/activate
  ec -g '  type ? for local help'
else
  ec -c '  creating virtual environment ...'
  ec -y 'python3 -m venv venv'
  python3 -m venv venv
  ec -g 'next step => make   # to install required python packages'
  source go
fi
