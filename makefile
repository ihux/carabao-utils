# makefile to manage carabao toolbox

all:
	make install
	make carabao

help:
	@echo '  make venv      # make virtual environment'
	@echo '  make install   # install packages for playground'
	@echo '  make carabao   # build carabao wheel and install'
	@echo '  make clean     # cleanup folder'
	@echo '  make scrap     # cleanup folder and scrap virtual environment'
	@echo ''
	@echo '  helpful commands:'
	@echo '    source venv/bin/activate    # activate virtual environment'
	@echo '    python                      # run python interpreter'
	@echo '    jupyter lab                 # start jupyter lab'

venv:
	python3 -m venv venv
	@bash local/bin/ec -g '=> . go'

install:
	python3 -m pip install --upgrade pip
	pip install --upgrade wheel
	pip install --upgrade setuptools
	pip install --upgrade twine
	pip install --upgrade ypstruct==0.0.2
	pip install pytest==4.4.1
	pip install pytest-runner==4.4
	python3 -m pip install --upgrade build
	pip install --upgrade numpy
	pip install matplotlib
	pip install --upgrade torch
	pip install jupyterlab
	pip install xeus-python
	pip install gitpython

carabao: dist/carabao-0.0.1-py3-none-any.whl

dist/carabao-0.0.1-py3-none-any.whl: carabao/src/carabao/*.py
	python carabao/src/carabao/neurotron.py
	cd carabao &&	make carabao

clean:
	cd carabao &&	make clean

scrap:
	make clean
	rm -rf venv/
