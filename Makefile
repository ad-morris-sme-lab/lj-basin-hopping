.PHONY: install test run lint
install:
	python -m pip install -e . -r requirements.txt
test:
	python -m pytest -q
run:
	python examples/basin_hopping_lj13.py
lint:
	python -m compileall -q src examples
