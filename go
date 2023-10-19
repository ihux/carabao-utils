#!/bin/bash
# .go: setup some aliases to be ready to go

export BARC=$HOME/.bash_profile
export REPO=`pwd`
source local/bin/alias.sh
ec -g '  type ? for local help'

if [ -d venv ]; then
  source venv/bin/activate
fi
