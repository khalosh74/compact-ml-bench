PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || which python3)
PIP := $(shell [ -x .venv/bin/pip ] && echo .venv/bin/pip || which pip)

setup:
	$(PY) -m pip install --upgrade pip setuptools wheel
	- $(PIP) install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt || $(PIP) install -r requirements.txt

status:
	$(PY) scripts/status.py

sanity:
	$(PY) scripts/sanity.py

diagnose:
	$(PY) scripts/diagnose.py

freeze:
	$(PIP) freeze > requirements.lock.txt

clean:
	rm -rf __pycache__ outputs/*

deepclean: clean
	rm -rf .venv runs data outputs __pycache__ *.lock *.log
