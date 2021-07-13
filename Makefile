# define the name of the virtual environment directory
VENV := venv

# default target, when make executed without arguments
all: venv

# venv is a shortcut target
venv: $(VENV)/bin/activate
$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip3 install -r requirements.txt
# source venv/bin/activate

clean:
	@rm -rf $(VENV)
	@rm -rf __pycache__
	@rm weights.npy
	@rm houses.csv

.PHONY: all venv clean