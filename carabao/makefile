# makefile to manage carabao toolbox

all:
	make carabao

help:
	@echo '  make carabao   # build carabao wheel and install'
	@echo '  make clean     # cleanup folder'
	@echo ''
	@echo '  helpful commands:'
	@echo '    python                      # run python interpreter'
	@echo '    jupyter lab                 # start jupyter lab'

carabao:
	python3 -m build
	pip install --force-reinstall dist/*.whl

clean:
	rm -rf dist
	rm -rf src/carabao.egg-info
