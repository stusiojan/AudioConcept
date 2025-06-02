#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = AudioConcept
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

# ## Set up Python interpreter environment
# .PHONY: create_environment
# create_environment:
# 	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
# 	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	
## Set up Python interpreter environment with conda
.PHONY: create_conda_env
create_conda_env:
	conda -V
	conda env create -f environment.yml
	conda activate wimu


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m AudioConcept.dataset

## Prepare features
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m AudioConcept.features

## Train CNN model
.PHONY: train_cnn
train_cnn:
	$(PYTHON_INTERPRETER) -m AudioConcept.train main CNN

## Train VGGish model
.PHONY: train_vgg
train_vgg:
	$(PYTHON_INTERPRETER) -m AudioConcept.train main VGGish

## Train VGGish model on 3,96s audio
.PHONY: train_short_chunk_vgg
train_short_chunk_vgg:
	$(PYTHON_INTERPRETER) -m AudioConcept.train main VGGish --audio-length VGG

## Train SVM model
.PHONY: train_svm
train_svm:
	$(PYTHON_INTERPRETER) -m AudioConcept.train main SVM


## Evaluate CNN model
.PHONY: evaluate_cnn
evaluate_cnn:
	$(PYTHON_INTERPRETER) -m AudioConcept.evaluate CNN

## Evaluate VGGish model
.PHONY: evaluate_vgg
evaluate_vgg:
	$(PYTHON_INTERPRETER) -m AudioConcept.evaluate VGGish

## Evaluate VGGish model on 3,96s audio
.PHONY: evaluate_short_chunk_vgg
evaluate_short_chunk_vgg:
	$(PYTHON_INTERPRETER) -m AudioConcept.evaluate VGGish --audio-length VGG

## Evaluate SVM model
.PHONY: evaluate_svm
evaluate_svm:
	$(PYTHON_INTERPRETER) -m AudioConcept.evaluate SVM

## Predict using CNN model
.PHONY: predict_cnn
predict_cnn:
	$(PYTHON_INTERPRETER) -m AudioConcept.predict CNN

## Predict using VGGish model
.PHONY: predict_vgg
predict_vgg:
	$(PYTHON_INTERPRETER) -m AudioConcept.predict VGGish

## Predict using SVM model
.PHONY: predict_svm
predict_svm:
	$(PYTHON_INTERPRETER) -m AudioConcept.predict SVM


## Run frontend app
.PHONY: run_app
run_app:
	streamlit run demo/app.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
