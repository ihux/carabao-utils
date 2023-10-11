# makefile to manage carabao toolbox

all:
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
	@echo 'invoke: $ source venv/bin/activate (or: . go; ve)'

install:
	pip install --upgrade wheel
	pip install  --upgrade setuptools
	pip install  --upgrade twine
	pip install pytest==4.4.1
	pip install pytest-runner==4.4
	python3 -m pip install --upgrade build
	pip install --upgrade numpy
	pip install matplotlib
	pip install --upgrade torch
	pip install jupyterlab

carabao: venv
	#ls venv/bin/activate
	#source venv/bin/activate
	cd carabao &&	make carabao

clean:
	cd carabao &&	make clean

scrap:
	make clean
	rm -rf venv/