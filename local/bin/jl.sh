#!/bin/bash
# jl.sh: launch jupyter lab

   if [ "$*" == "-?" ]; then
      echo '   usage: jl      # launch jupyter lab'
      echo '          jl -?   # help on jl'
      echo ''
      echo '   see also: de, ec, ve'
      exit 1
   fi

   if [ "$*" == "" ]; then
      jupyter lab || exit 1
   fi

   exit 0
