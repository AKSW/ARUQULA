from flask import Flask, request, jsonify
import asyncio
import time
import os
from datetime import datetime
import csv
import yaml

LLM_ENGINE = os.environ["LLM_ENGINE"]
QUERY_LOG_FILE_PATH = os.environ.get("QUERY_LOG_FILE_PATH", "./logs/query-log.csv")

from spinach_agent.part_to_whole_parser import PartToWholeParser
from spinach_agent.evaluate_parser import post_processing

KNOWN_DATASETS = [
    "https://text2sparql.aksw.org/2025/dbpedia/",
    "https://text2sparql.aksw.org/2025/corporate/"
]

def t2s(question, dataset_id, parser):

    questions = [{"question": question, "conversation_history": []}]
    results = asyncio.run(post_processing(
        dataset_id,
        parser.run_batch(questions, batch_size = 1),
        regex_use_select_distinct_and_id_not_label=True,
        llm_extract_prediction_if_null=True
    ))
    return results[0].get("predicted_sparql")

def write_query_log(ip, dataset, question, sparql):
    file_exists = os.path.isfile(QUERY_LOG_FILE_PATH)

    with open(QUERY_LOG_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists or os.stat(QUERY_LOG_FILE_PATH).st_size == 0:
            writer.writerow(['date', 'ip', 'dataset', 'model', 'question', 'sparql'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ip, dataset, LLM_ENGINE, question, sparql])
        file.close()

    csv_to_yaml(QUERY_LOG_FILE_PATH, QUERY_LOG_FILE_PATH.replace(".csv", ".yml"))

def csv_to_yaml(csv_file_path, yaml_file_path):
    with open(csv_file_path, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        data = list(reader)  # Convert to list of dictionaries

    with open(yaml_file_path, mode='w') as yaml_file:
        yaml.dump(data, yaml_file, sort_keys=False)

def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify(status="ok"), 200

    @app.route('/text2sparql', methods=["GET"])
    def text2sparql():

        question = request.args.get("question", type=str)
        dataset = request.args.get("dataset", type=str)
        ip = ""
        sparql = ""

        if question is None or dataset is None:
            return "You must supply a question and the name of a dataset.", 422

        if dataset not in KNOWN_DATASETS:
            return "Unknown dataset: " + dataset, 404

        print(f"Incoming request for <{dataset}>: '{question}'")

        try:
            semantic_parser_class = PartToWholeParser
            semantic_parser_class.initialize(engine=LLM_ENGINE, dataset_id=dataset)

            start = time.time()
            sparql = t2s(question, dataset, semantic_parser_class)
            duration = time.time() - start
            print(f"It took {int(duration)}s to answer the query: '{question}' on dataset <{dataset}>", flush=True)

            if request.headers.get('X-Forwarded-For'):
                ip = request.headers.get('X-Forwarded-For').split(',')[0]
            else:
                ip = request.remote_addr

            write_query_log(ip, dataset, question, sparql)

            return {
                "query": sparql,
                "dataset": dataset,
                "question": question }, 200
        except Exception as e:
            print(f"Exception occurred: {e}")
            return "Backend Error", 500

    return app

if __name__ == "__main__":

    app = create_app()
    app.run()