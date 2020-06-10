.PHONY: process features concepts keywords categories structure requirements \
		sync_data_to_s3 sync_data_from_s3 sync_raw_data_from_s3 pipeline plots \
		tests docs check_clean clean_experiment clean

#.SHELLFLAGS := -o nounset -c
SHELL := /bin/bash

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = hq-ocio-ci-bigdata/home/DataSquad/classifier_scripts/
PROFILE = moderate
PROJECT_NAME = classifier_scripts
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# These three variables should be tailored for you use case.
EXPERIMENT_NAME=test
IN_CORPUS=data/raw/STI_public_metadata_records_sample100.jsonl
IN_CONFIG=config/test_config.yml

INTERIM_DATA=data/interim/$(EXPERIMENT_NAME)
INTERIM_CORPUS=data/interim/$(EXPERIMENT_NAME)/abs_kwds.jsonl

FIELD=text
SUBJ_MAPPING=data/interim/subj_mapping.json
FEATURES=data/interim/$(EXPERIMENT_NAME)/features.jsonl

CONCEPT_FIELD='keywords'
CAT_FIELD='categories'
OUT_KWD_INDICES=data/interim/$(EXPERIMENT_NAME)/kwd_indices.json
OUT_CAT_INDICES=data/interim/$(EXPERIMENT_NAME)/cat_indices.json
OUT_KWD_RAW_TO_LEMMA=models/$(EXPERIMENT_NAME)/kwd_raw2lemma.json
OUT_CAT_RAW_TO_LEMMA=models/$(EXPERIMENT_NAME)/cat_raw2lemma.json

OUT_OUTER_MODEL_DIR=models/$(EXPERIMENT_NAME)
OUT_KWD_MODEL_DIR=$(OUT_OUTER_MODEL_DIR)/keywords
OUT_CAT_MODEL_DIR=$(OUT_OUTER_MODEL_DIR)/categories

METRICS_LOC=reports/$(EXPERIMENT_NAME)
BERT_MODELS_DIR=models/bert_models

GIT_REMOTE='origin'
IMAGE_NAME=concept_trainer


## Test underlying dsconcept library
tests:
	nosetests --with-coverage --cover-package dsconcept --cover-html; \
	open cover/index.html

## Run through all steps to create all classifiers
pipeline: structure process features concepts vectorizer_and_matrix \
		  categories keywords metrics plots

## create directory structure if necessary
structure:
	mkdir -p data
	mkdir -p data/raw
	mkdir -p data/interim
	mkdir -p data/interim/$(EXPERIMENT_NAME)
	mkdir -p models/$(EXPERIMENT_NAME)
	mkdir -p config
	mkdir -p reports
	mkdir -p reports/$(EXPERIMENT_NAME)

## install newest version of dependencies. Untested.
approximate-install:
	pip install scikit-learn spacy tqdm textacy pyyaml pandas h5py \
		testfixtures hypothesis dask pytest matplotlib
	$(PYTHON_INTERPRETER) -m spacy download en_core_web_sm

## install precise python dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m spacy download en_core_web_sm

## processing by merging text and keyword fields
process: $(INTERIM_CORPUS)
$(INTERIM_CORPUS): $(IN_CORPUS) src/process.py
	mkdir -p data/interim/$(EXPERIMENT_NAME)
	mkdir -p models/$(EXPERIMENT_NAME)
	$(PYTHON_INTERPRETER) src/process.py $(IN_CORPUS) $(SUBJ_MAPPING) $(INTERIM_CORPUS)

## creature feature sets for processed data
features: $(FEATURES)
$(FEATURES): $(INTERIM_CORPUS) src/features.py
	$(PYTHON_INTERPRETER) src/features.py $(INTERIM_CORPUS) $(FIELD) $(FEATURES)

## create concepts indices json and mappings from raw to lemmas
concepts: $(OUT_KWD_INDICES) $(OUT_CAT_INDICES)
$(OUT_KWD_INDICES) $(OUT_CAT_INDICES): $(INTERIM_CORPUS) src/concepts.py
	$(PYTHON_INTERPRETER) src/concepts.py \
		$(INTERIM_CORPUS) \
		$(CONCEPT_FIELD) $(CAT_FIELD) \
		$(OUT_KWD_INDICES) $(OUT_CAT_INDICES) \
		$(OUT_KWD_RAW_TO_LEMMA) $(OUT_CAT_RAW_TO_LEMMA)

## create vectorizer and feature matrix from feature records
vectorizer_and_matrix: $(INTERIM_DATA)/feature_matrix.jbl
$(INTERIM_DATA)/feature_matrix.jbl: src/make_vec_and_matrix.py $(FEATURES) $(IN_CONFIG)
	mkdir -p $(OUT_OUTER_MODEL_DIR) && \
	cp $(IN_CONFIG) $(OUT_OUTER_MODEL_DIR)/config.yml && \
	$(PYTHON_INTERPRETER) src/make_vec_and_matrix.py \
		$(FEATURES) $(IN_CONFIG) $(INTERIM_DATA) $(OUT_OUTER_MODEL_DIR)/vectorizer.jbl
# TODO: separate outputs for vec and matrix, send matrix to INTERIM_DATA

## train category models
categories: src/make_cat_models.py $(OUT_CAT_INDICES) $(INTERIM_DATA)/feature_matrix.jbl $(IN_CONFIG)
	mkdir -p $(OUT_CAT_MODEL_DIR) && \
	$(PYTHON_INTERPRETER) src/make_cat_models.py \
		$(INTERIM_DATA)/feature_matrix.jbl \
		$(INTERIM_DATA)/train_inds.npy \
		$(INTERIM_DATA)/test_inds.npy \
		$(OUT_CAT_INDICES) \
		$(OUT_CAT_RAW_TO_LEMMA) \
		$(IN_CONFIG) $(OUT_CAT_MODEL_DIR)

## train keyword models
keywords: src/make_kwd_models.py $(OUT_KWD_INDICES) $(INTERIM_DATA)/feature_matrix.jbl $(IN_CONFIG) $(INTERIM_DATA)/test_inds.npy
	mkdir -p $(OUT_KWD_MODEL_DIR) && \
	$(PYTHON_INTERPRETER) src/make_kwd_models.py \
		$(INTERIM_DATA)/feature_matrix.jbl \
		$(INTERIM_DATA)/train_inds.npy \
		$(INTERIM_DATA)/test_inds.npy \
		$(OUT_KWD_INDICES) $(OUT_CAT_INDICES) \
		$(OUT_KWD_RAW_TO_LEMMA) $(OUT_CAT_RAW_TO_LEMMA) \
		$(IN_CONFIG) $(OUT_KWD_MODEL_DIR)

## Only train keywords on full training set. No topic splitting.
keywords-no-topics:
	mkdir -p $(OUT_KWD_MODEL_DIR) && \
	$(PYTHON_INTERPRETER) src/make_kwd_models.py \
		$(INTERIM_DATA)/feature_matrix.jbl \
		$(INTERIM_DATA)/train_inds.npy \
		$(INTERIM_DATA)/test_inds.npy \
		$(OUT_KWD_INDICES) $(OUT_CAT_INDICES) \
		$(OUT_KWD_RAW_TO_LEMMA) $(OUT_CAT_RAW_TO_LEMMA) \
		$(IN_CONFIG) $(OUT_KWD_MODEL_DIR) --no-topics ${VERBOSE}

## Get predictions from category models made with BERT classification
bert_cat_model_scores:
	mkdir -p $(METRICS_LOC) && \
	$(PYTHON_INTERPRETER) src/get_bert_cat_models_preds.py \
		--data_dir $(INTERIM_DATA) \
		--models_dir $(OUT_OUTER_MODEL_DIR) \
		--reports_dir $(METRICS_LOC) \
		--base_model_dir ../nlp-working-with-bert/models/base/cased_L-12_H-768_A-12 \
		--finetuned_model_dir ../nlp-working-with-bert/models/01_02_2020/ \
		--sample 1000
#		--base_model_dir models/bert_models/cased_L-12_H-768_A-12 \
#		--finetuned_model_dir models/bert_models/cased_L-12_H-768_A-12/cache

## Create cleaned dataset for training transformer category models
bert_cat_clean_dataset:
	$(PYTHON_INTERPRETER) src/make_records_for_cat_bert.py \
		$(INTERIM_CORPUS) \
		$(INTERIM_DATA) \
		$(OUT_OUTER_MODEL_DIR)/bert

## Get metrics for test data
metrics:
	mkdir -p $(METRICS_LOC) && \
	$(PYTHON_INTERPRETER) src/dsconcept/get_metrics.py \
		--experiment_name $(EXPERIMENT_NAME) \
		--out_store $(METRICS_LOC)/store.h5 \
		--out_cat_preds $(METRICS_LOC)/cat_preds.npy \
		--batch_size 500

## Synthesize predictions for keywords and classifiers to create full classification
synthesize:
	mkdir -p $(METRICS_LOC) && \
	$(PYTHON_INTERPRETER) src/synthesize_predictions.py \
		--experiment_name $(EXPERIMENT_NAME) \
		--synth_strat mean \
		--in_cat_preds $(METRICS_LOC)/cat_preds.npy \
		--store $(METRICS_LOC)/store.h5 \
		--synth_batch_size 3000 \
		--threshold 0.5 \
		--out_synth_scores $(METRICS_LOC)/synth_mean_results.csv

## Synthesize predictions for keywords and classifiers to create full classification
synthesize-bert:
	mkdir -p $(METRICS_LOC) && \
	$(PYTHON_INTERPRETER) src/synthesize_predictions.py \
		--experiment_name $(EXPERIMENT_NAME) \
		--synth_strat mean \
		--in_cat_preds $(METRICS_LOC)/bert_cat_preds.npy \
		--store $(METRICS_LOC)/store.h5 \
		--synth_batch_size 3000 \
		--threshold 0.5 \
		--out_synth_scores $(METRICS_LOC)/synth_bert_mean_results.csv

## create plots from performance metrics
plots:
	mkdir -p $(METRICS_LOC)/figures && \
	$(PYTHON_INTERPRETER) src/make_plots.py \
		--mean $(METRICS_LOC)/synth_mean_results.csv \
		--in_cats_dir $(OUT_CAT_MODEL_DIR)/models \
		--in_kwds_dir $(OUT_KWD_MODEL_DIR)/models \
		--in_cats_dir $(OUT_CAT_MODEL_DIR)/models \
		--in_vectorizer $(OUT_OUTER_MODEL_DIR)/vectorizer.jbl \
		--in_clean_data $(INTERIM_CORPUS) \
		--in_config $(OUT_OUTER_MODEL_DIR)/config.yml \
		--out_plots_dir $(METRICS_LOC)/figures

## create plots from performance metrics
plots-bert:
	mkdir -p $(METRICS_LOC)/figures_bert && \
	$(PYTHON_INTERPRETER) src/make_plots.py \
		--mean $(METRICS_LOC)/synth_bert_mean_results.csv \
		--in_cats_dir $(OUT_CAT_MODEL_DIR)/models \
		--in_kwds_dir $(OUT_KWD_MODEL_DIR)/models \
		--in_cats_dir $(OUT_CAT_MODEL_DIR)/models \
		--in_vectorizer $(OUT_OUTER_MODEL_DIR)/vectorizer.jbl \
		--in_clean_data $(INTERIM_CORPUS) \
		--in_config $(OUT_OUTER_MODEL_DIR)/config.yml \
		--out_plots_dir $(METRICS_LOC)/figures_bert

## Build docker image for training
build:
	export COMMIT=$$(git log -1 --format=%H); \
	export REPO_URL=$$(git remote get-url $(GIT_REMOTE)); \
	export REPO_DIR=$$(dirname $$REPO_URL); \
	export BASE_NAME=$$(basename $$REPO_URL .git); \
	export GIT_LOC=$$REPO_DIR/$$BASE_NAME/tree/$$COMMIT; \
	export VERSION=$$(python version.py); \
	echo $$GIT_LOC; \
	echo $$VERSION; \
	docker build -t $(IMAGE_NAME):$$VERSION \
		--build-arg GIT_URL=$$GIT_LOC \
		--build-arg VERSION=$$VERSION .

## Start docker container for running full pipeline
container:
	export VERSION=$$(python version.py); \
	docker run -it \
		 -v $$(pwd)/data:/home/data \
		 -v $$(pwd)/models:/home/models \
		 -v $$(pwd)/config:/home/config \
		 -v $$(pwd)/reports:/home/reports \
		$(IMAGE_NAME):$$VERSION pipeline \
			EXPERIMENT_NAME=$(EXPERIMENT_NAME) \
			IN_CORPUS=$(IN_CORPUS) \
			IN_CONFIG=$(IN_CONFIG)

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

check_clean:
	@echo $(OUT_OUTER_MODEL_DIR)
	@echo data/interim/$(EXPERIMENT_NAME)
	@echo $(METRICS_LOC)
	@echo -n "Are you sure you want to remove the above folders? [y/N] " && read ans && [ $${ans:-N} = y ]

## delete all interim data, models, and reports for the given experiment
clean_experiment: check_clean
	rm -r $(OUT_OUTER_MODEL_DIR)
	rm -r data/interim/$(EXPERIMENT_NAME)
	rm -r $(METRICS_LOC)

## sync this experiment to s3
sync_experiment_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync models/$(EXPERIMENT_NAME) s3://$(BUCKET)models/$(EXPERIMENT_NAME)
	aws s3 sync data/interim/$(EXPERIMENT_NAME) s3://$(BUCKET)data/interim/$(EXPERIMENT_NAME)
	aws s3 sync reports/$(EXPERIMENT_NAME) s3://$(BUCKET)reports/$(EXPERIMENT_NAME)
else
	aws s3 sync models/$(EXPERIMENT_NAME) s3://$(BUCKET)models/$(EXPERIMENT_NAME) --profile $(PROFILE)
	aws s3 sync data/interim/$(EXPERIMENT_NAME) s3://$(BUCKET)data/interim/$(EXPERIMENT_NAME) --profile $(PROFILE)
	aws s3 sync reports/$(EXPERIMENT_NAME) s3://$(BUCKET)reports/$(EXPERIMENT_NAME) --profile $(PROFILE)
endif

## sync this experiment from s3
sync_experiment_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)models/$(EXPERIMENT_NAME) models/$(EXPERIMENT_NAME)
	aws s3 sync s3://$(BUCKET)reports/$(EXPERIMENT_NAME) reports/$(EXPERIMENT_NAME)
	aws s3 sync s3://$(BUCKET)data/processed/$(EXPERIMENT_NAME) data/processed/$(EXPERIMENT_NAME)
else
	aws s3 sync s3://$(BUCKET)models/$(EXPERIMENT_NAME) models/$(EXPERIMENT_NAME) --profile $(PROFILE)
	aws s3 sync s3://$(BUCKET)reports/$(EXPERIMENT_NAME) reports/$(EXPERIMENT_NAME) --profile $(PROFILE)
	aws s3 sync s3://$(BUCKET)data/processed/$(EXPERIMENT_NAME) data/processed/$(EXPERIMENT_NAME) --profile $(PROFILE)
endif

## sync raw starting data from s3
sync_raw_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 cp s3://hq-ocio-ci-bigdata/data/STI/STI_records_metadata.jsonl data/raw/STI_records_metadata.jsonl
else
	aws s3 cp s3://hq-ocio-ci-bigdata/data/STI/STI_records_metadata.jsonl data/raw/STI_records_metadata.jsonl --profile $(PROFILE)
endif
	echo "These records should be handled as moderate data assets. Handle these records with care."

## zip models necessary for running the app
zip-experiment-for-app:
	cd models/; \
	zip -r $(EXPERIMENT_NAME).zip \
		$(EXPERIMENT_NAME)/categories/models \
		$(EXPERIMENT_NAME)/keywords/models \
		$(EXPERIMENT_NAME)/kwd_raw2lemma.json \
		$(EXPERIMENT_NAME)/cat_raw2lemma.json \
		$(EXPERIMENT_NAME)/vectorizer.jbl \
		$(EXPERIMENT_NAME)/config.yml \

## Upload zipped experiment app files to s3
upload-experiment-zip-to-s3:
	aws s3 cp models/$(EXPERIMENT_NAME).zip s3://$(BUCKET)models/$(EXPERIMENT_NAME).zip --profile $(PROFILE)
#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')