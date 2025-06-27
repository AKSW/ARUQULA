#############################
## Front Matter            ##
#############################

SHELL := /bin/bash

.PHONY: help

.DEFAULT_GOAL := help

# Load environment variables from the .env file
ifneq (,$(wildcard ./.env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

#############################
## Targets                 ##
#############################

## activate the conda env
conda-activate:
	conda activate text2sparql

## deactivate the conda env
conda-deactivate:
	conda-deactivate

## install all conda and pip dependencies
conda-install:
	conda env update --file conda_env.yaml --prune

## remove all conda and pip deps
conda-deinstall:
	conda remove -n text2sparql --all

## run all dbpedia queries
run-dbpedia:
	inv evaluate-parser --parser-type part_to_whole --subsample=-1 --offset=2 --engine=deepseek-v3-202503 \
		--text2sparql-dataset-id="https://text2sparql.aksw.org/2025/dbpedia/" \
		--dataset=datasets/dbpedia_t2s/questions-dbpedia-en.json \
		--output-file=datasets/dbpedia_t2s/questions-dbpedia-results.json \
		--regex-use-select-distinct-and-id-not-label \
		--llm-extract-prediction-if-null

## run all dbpedia queries & select engine
run-dbpedia-select-engine:
	@echo "Select an LLM engine to run:"
	@echo "1) deepseek-v3-202503"
	@echo "2) gpt-4o-mini"
	@echo "3) gpt-4o"
	@echo "4) gpt-4.1-nano"
	@echo "5) gpt-4.1-mini"
	@echo "6) gpt-4.1"
	@read -p "Enter choice [1-6]: " engine_choice; \
	case $$engine_choice in \
		1) engine=deepseek-v3-202503 ;; \
		2) engine=gpt-4o-mini ;; \
		3) engine=gpt-4o ;; \
		4) engine=gpt-4.1-nano ;; \
		5) engine=gpt-4.1-mini ;; \
		6) engine=gpt-4.1 ;; \
		*) echo "Invalid option"; exit 1 ;; \
	esac; \
	echo "Running dbpedia task with engine $$engine..."; \
	inv evaluate-parser --parser-type part_to_whole --batch-size=16 --subsample=150 --engine="$$engine" \
		--text2sparql-dataset-id="https://text2sparql.aksw.org/2025/dbpedia/" \
		--dataset=datasets/dbpedia_t2s/questions-dbpedia-en.json \
		--output-file=datasets/dbpedia_t2s/questions-dbpedia-results.json \
		--regex-use-select-distinct-and-id-not-label \
		--llm-extract-prediction-if-null

## run all org queries
run-org:
	inv evaluate-parser --parser-type part_to_whole --subsample=-1 --offset=0 --engine=deepseek-v3-202503 \
		--text2sparql-dataset-id="https://text2sparql.aksw.org/2025/corporate/" \
		--dataset=datasets/text2sparql/org/en/test.json \
		--output-file=datasets/text2sparql/org/en/test-results.json \
		--regex-use-select-distinct-and-id-not-label \
		--llm-extract-prediction-if-null

## run all org queries & select engine
run-org-select-engine:
	@echo "Select an LLM engine to run:"
	@echo "1) deepseek-v3-202503"
	@echo "2) gpt-4o-mini"
	@echo "3) gpt-4o"
	@echo "4) gpt-4.1-nano"
	@echo "5) gpt-4.1-mini"
	@echo "6) gpt-4.1"
	@read -p "Enter choice [1-6]: " engine_choice; \
	case $$engine_choice in \
		1) engine=deepseek-v3-202503 ;; \
		2) engine=gpt-4o-mini ;; \
		3) engine=gpt-4o ;; \
		4) engine=gpt-4.1-nano ;; \
		5) engine=gpt-4.1-mini ;; \
		6) engine=gpt-4.1 ;; \
		*) echo "Invalid option"; exit 1 ;; \
	esac; \
	echo "Running org task with engine $$engine..."; \
	inv evaluate-parser --parser-type part_to_whole --batch-size=16 --subsample=-1 --offset=0 --engine="$$engine" \
		--text2sparql-dataset-id=https://text2sparql.aksw.org/2025/corporate/ \
		--dataset=datasets/text2sparql/org/en/test.json \
		--output-file=datasets/text2sparql/org/en/test-results.json \
		--regex-use-select-distinct-and-id-not-label \
		--llm-extract-prediction-if-null

## run with selectable dataset and engine (generic)
run-generic:
	@echo "Select a dataset to run:"
	@echo "1) dbpedia (en)"
	@echo "2) dbpedia (es)"
	@echo "3) org"
	@read -p "Enter choice [1-3]: " dataset_choice; \
	case $$dataset_choice in \
		1) dataset_id=https://text2sparql.aksw.org/2025/dbpedia/; \
		   dataset_path=datasets/dbpedia_t2s/questions-dbpedia-en.json; \
		   output_path=datasets/dbpedia_t2s/questions-dbpedia-en-results.json ;; \
		2) dataset_id=https://text2sparql.aksw.org/2025/dbpedia/; \
		   dataset_path=datasets/dbpedia_t2s/questions-dbpedia-es.json; \
		   output_path=datasets/dbpedia_t2s/questions-dbpedia-es-results.json ;; \
		3) dataset_id=https://text2sparql.aksw.org/2025/corporate/; \
		   dataset_path=datasets/text2sparql/org/en/test.json; \
		   output_path=datasets/text2sparql/org/en/test-results.json ;; \
		*) echo "Invalid dataset option"; exit 1 ;; \
	esac; \
	\
	echo "Select an LLM engine to run:"; \
	echo "1) deepseek-v3-202503"; \
	echo "2) gpt-4o-mini"; \
	echo "3) gpt-4o"; \
	echo "4) gpt-4.1-nano"; \
	echo "5) gpt-4.1-mini"; \
	echo "6) gpt-4.1"; \
	echo "7) llama-4-maverick"; \
	read -p "Enter choice [1-7]: " engine_choice; \
	case $$engine_choice in \
		1) engine=deepseek-v3-202503 ;; \
		2) engine=gpt-4o-mini ;; \
		3) engine=gpt-4o ;; \
		4) engine=gpt-4.1-nano ;; \
		5) engine=gpt-4.1-mini ;; \
		6) engine=gpt-4.1 ;; \
		7) engine=meta-llama-4-maverick ;; \
		*) echo "Invalid engine option"; exit 1 ;; \
	esac; \
	\
	read -p "Select batch size (default: 16): " batchSize; \
	batchSize=$${batchSize:-16}; \
	read -p "Select subsample (default: -1/everything): " subsample; \
	subsample=$${subsample:--1}; \
	read -p "Select offset (default: 0): " offset; \
	offset=$${offset:-0}; \
	echo "Running task with dataset $$dataset_id with $$subsample questions from $$dataset_path, engine: $$engine, batchSize: $$batchSize, offset: $$offset ..."; \
	inv evaluate-parser --parser-type part_to_whole --batch-size="$$batchSize" --subsample="$$subsample" --offset="$$offset" --engine="$$engine" \
		--text2sparql-dataset-id="$$dataset_id" \
		--dataset="$$dataset_path" \
		--output-file="$$output_path" \
		--regex-use-select-distinct-and-id-not-label \
		--llm-extract-prediction-if-null

## run with selectable dataset, iterate over multiple engines, and number of iterations
run-iterations-engines:
	@echo "Select a dataset to run:"
	@echo "1) dbpedia"
	@echo "2) org"
	@read -p "Enter choice [1-2]: " dataset_choice; \
	case $$dataset_choice in \
		1) dataset_id=https://text2sparql.aksw.org/2025/dbpedia/; \
		   dataset=dbpedia; \
		   dataset_path=datasets/dbpedia_t2s/questions-dbpedia-en.json ;; \
		2) dataset_id=https://text2sparql.aksw.org/2025/corporate/; \
		   dataset=org; \
		   dataset_path=datasets/text2sparql/org/en/test.json ;; \
		*) echo "Invalid dataset option"; exit 1 ;; \
	esac; \
	\
	read -p "Enter number of iterations: " iterations; \
	engines=("gpt-4.1-mini" "gpt-4.1-nano"); \
	for engine in $${engines[@]}; do \
		for i in $$(seq 1 $$iterations); do \
		    mkdir -p "datasets/text2sparql/$$dataset/$$engine/"; \
			output_path="datasets/text2sparql/$$dataset/$$engine/results_iteration_$$i.json"; \
			echo "Running task for iteration $$i with engine $$engine..."; \
			inv evaluate-parser --parser-type part_to_whole --batch-size=16 --subsample=1 --offset=0 --engine="$$engine" \
				--text2sparql-dataset-id="$$dataset_id" \
				--dataset="$$dataset_path" \
				--output-file="$$output_path" \
				--regex-use-select-distinct-and-id-not-label \
				--llm-extract-prediction-if-null; \
		done; \
	done; \
	make aggregate-f1-scores dataset_id=$$dataset engines=$${engines[@]} iterations=$$iterations

## run evaluation for both datasets with multiple engines and number of iterations
run-eval:
	@echo "Select the number of iterations for all datasets:"
	@read -p "Enter number of iterations: " iterations; \
	datasets=("dbpedia" "org"); \
	for dataset in $${datasets[@]}; do \
		if [ "$$dataset" = "dbpedia" ]; then \
			dataset_id=https://text2sparql.aksw.org/2025/dbpedia/; \
			dataset_path=datasets/dbpedia_t2s/questions-dbpedia-en.json; \
		else \
			dataset_id=https://text2sparql.aksw.org/2025/corporate/; \
			dataset_path=datasets/text2sparql/org/en/test.json; \
		fi; \
		\
		engines=("gpt-4.1-nano" "gpt-4.1-mini"); \
		for engine in $${engines[@]}; do \
			for i in $$(seq 1 $$iterations); do \
				mkdir -p "datasets/text2sparql/$$dataset/$$engine/"; \
				output_path="datasets/text2sparql/$$dataset/$$engine/results_iteration_$$i.json"; \
				echo "Running task for iteration $$i with engine $$engine on dataset $$dataset..."; \
				inv evaluate-parser --parser-type part_to_whole --batch-size=20 --subsample=50 --offset=0 --engine="$$engine" \
					--text2sparql-dataset-id="$$dataset_id" \
					--dataset="$$dataset_path" \
					--output-file="$$output_path" \
					--regex-use-select-distinct-and-id-not-label \
					--llm-extract-prediction-if-null; \
			done; \
		done; \
		make aggregate-f1-scores dataset_id=$$dataset engines="$${engines[*]}" iterations=$$iterations; \
	done

## aggregate F1 scores from JSON outputs into a final CSV
aggregate-f1-scores:
	@echo "Aggregating F1 scores..."
	@dataset_id=$(dataset_id); \
	engines=($(engines)); \
	iterations=$(iterations); \
	output_csv="datasets/text2sparql/$$dataset_id/f1_scores_summary.csv"; \
	echo "dataset,num_iterations,num_questions,engine,f1" > $$output_csv; \
	for engine in $${engines[@]}; do \
	    echo "Engine $$engine"; \
		total_f1=0; \
		count=0; \
		for i in $$(seq 1 $$iterations); do \
			json_file="datasets/text2sparql/$$dataset_id/$$engine/results_iteration_$$i.json"; \
			if [ -f $$json_file ]; then \
				f1_scores=$$(jq '.[] | .f1' $$json_file); \
				for f1_score in $$f1_scores; do \
					total_f1=$$(echo "$$total_f1 + $$f1_score" | bc); \
					count=$$((count + 1)); \
				done; \
			else \
				echo "Warning: $$json_file does not exist." ; \
			fi; \
		done; \
		echo "Total F1: $$total_f1, Count: $$count"; \
		if [ $$count -gt 0 ]; then \
		    #echo "Calculating average: scale=4; $$total_f1 / $$count"; \
			average_f1=$$(echo "scale=4; $$total_f1 / $$count" | bc); \
			num_questions=$$((count / iterations)); \
			#echo "Average F1: $$average_f1"; \
			echo "$$dataset_id,$$iterations,$$num_questions,$$engine,$$average_f1" >> $$output_csv; \
		else \
			echo "$$dataset_id,$$iterations,$$num_questions,$$engine,N/A" >> $$output_csv; \
		fi; \
	done; \
	echo "Aggregation complete. Results saved to $$output_csv."

## write custom query
query-endpoint:
	@read -p "Select dataset [dbpedia/corporate] (default: dbpedia): " d; \
	d=$${d:-dbpedia}; \
	read -p "Enter your question: " q; \
	curl -G "http://localhost:8085/text2sparql" --data-urlencode "dataset=https://text2sparql.aksw.org/2025/$$d/" --data-urlencode "question=$$q"

## build the docker image for the challenge interface
docker-build-t2s-endpoint:
	docker build --pull -f deploy/Dockerfile -t "eti/projects/text2sparql/api" .

#############################
## Help Target             ##
#############################

## Show this help
help:
	@printf "Available targets:\n\n"
	@awk '/^[a-zA-Z\-_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  %-30s %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)
