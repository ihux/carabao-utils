#!/bin/bash
# ve.sh: deal with virtual python environment
#        executes script .venv in current directory for activation

   if [ "$*" == "-?" ]; then
      echo '   usage: ve           # activate virtual python environment'
      echo '          ve -a        # activate virtual python environment'
      echo '          ve -c        # create virtual environment'
      echo '          ve -d        # deactivate virtual python environment'
      echo '          ve -i        # install python library wheel'
      echo '          ve -p        # install python packages for lib generation'
      echo '          ve -s <lib>  # creating library skeleton files'
      echo '          ve -u <lib>  # uninstall python library package'
      echo '          ve -w        # generate python library wheel'
      echo '          ve -?        # help on virtual environment'
      echo ''
      echo '   Combined options'
      echo '          ve -cap      # create, activate virtual environment, install packages'
      echo '          ve -wi       # generate and install python library wheel'
      echo ''
      echo '   To create a python library:'
      echo '      ve -c            # create virtual python library'
      echo '      ve -a            # activate virtual environment'
      echo '      ve -p            # install python packages for lib generation'
      echo '      ve -s <lib>      # create skeleton files for python lib'
      echo '      ...              # provide the library files'
      echo '      ve -w            # python setup.py bdist_wheel'
      echo '      ve -i            # pip install dist/*.whl'
      echo '      ...'
      echo '      ve -u <lib>      # pip uninstall <lib>'
      echo ''
      echo '   see also: de, python, pip'
   fi

#===============================================================================
# activate virtual python environment
#===============================================================================

   if [ "$*" == "-a" ] || [ "$*" == "" ]; then
      if [ -d venv ]; then
        source venv/bin/activate
      else
        echo '*** error: no venv directory => use: $ ve -c'
      fi
   fi

#===============================================================================
# create virtual python environment
#===============================================================================

   if [ "$*" == "-c" ]; then
     ec -y creating venv directory for virtual python environment
     python3 -m venv venv
   fi

#===============================================================================
# deactivate virtual python environment
#===============================================================================

   if [ "$*" == "-d" ]; then
     deactivate 2>/dev/null
   fi

#===============================================================================
# install python library wheel
#===============================================================================

   if [ "$1" == "-i" ]; then
     ec -y 'pip install  --force-reinstall dist/*.whl   # install dist/*.whl'
     pip install --force-reinstall dist/*.whl
   fi

#===============================================================================
# install python packages for library generation
#===============================================================================

   if [ "$*" == "-p" ]; then
     ec -y "pip install --upgrade wheel"
     pip install --upgrade wheel

     ec -y "pip install --upgrade setuptools"
     pip install  --upgrade setuptools

     ec -y "pip install --upgrade twine"
     pip install  --upgrade twine

     ec -y "pip install pytest==4.4.1"
     pip install pytest==4.4.1

     ec -y "pip install pytest-runner==4.4"
     pip install pytest-runner==4.4

     ec -y "python3 -m pip install --upgrade build"
     python3 -m pip install --upgrade build

     ec -y "pip install --upgrade numpy"
     pip install --upgrade numpy

     ec -y "pip install --upgrade torch"
     pip install --upgrade torch

     ec -y "pip install jupyterlab"
     pip install jupyterlab
   fi

#===============================================================================
# creating library skeleton files
#===============================================================================

   if [ "$1" == "-s" ]; then
     if [ "$2" == "" ]; then
       echo '*** error: name of library expected (arg2)'
     else
       ec -y "creating skeleton file structure for python lib $2 ..."
       echo >setup.py
       echo >README.md

       mkdir $2             # library folder
       echo >$2/__init__.py
       echo >$2/$2.py

       mkdir tests          # library folder
       echo >tests/__init__.py
       echo >tests/test_$2.py
     fi
   fi

#===============================================================================
# generate python library wheel
#===============================================================================

   if [ "$1" == "-u" ]; then
     if [ "$2" == "" ]; then
       echo '*** error: name of library expected (arg2)'
     else
       ec -y 'pip uninstall $2   # uninstall python lib package'
       pip uninstall $2
     fi
   fi

#===============================================================================
# generate python library wheel
#===============================================================================

   if [ "$1" == "-w" ]; then
     #ec -y 'python setup.py bdist_wheel # generate dist/*.whl'
     #python setup.py bdist_wheel
     ec -y 'python3 -m build # generate dist/*.whl'
     python3 -m build
   fi

#===============================================================================
# combined options
#===============================================================================

   if [ "$1" == "-cap" ]; then
     source ve.sh -d
     source ve.sh -c
     source ve.sh -a
     source ve.sh -l
   fi

   if [ "$1" == "-wi" ]; then
     source ve.sh -w
     source ve.sh -i
   fi
