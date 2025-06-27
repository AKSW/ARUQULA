# Dockerized Version of Spinach

First, copy the `env.dist` file to `.env`, then open it in your favorite text editor and fill in the following values:

- `OPENAI_API_KEY` is necessary at the moment because Spinach uses GPT4-o to extract SPARQL from LLM answers
- `LLM_API_KEY` is the API Key for the LLM that you actually want to use. Do not change the name of this variable, just enter the API Key.
- `API_BASE` is the base path of the API.
- `LLM_ENGINE` is the concrete name of the model that you would like to use, e.g. `gpt-4-turbo-2024-04-09` or `mistral/mistral-large-latest`

When you are done making changes, run `sudo docker compose up -d`. By default, the app will listen on port 5555 (can be changed in `./docker-compose.yaml`) and uses the path `/text2sparql`. You must supply a question via the `question` parameter and the name of a dataset via the `dataset` parameter, which must be one of:

- `https://text2sparql.aksw.org/2025/dbpedia/`
- `https://text2sparql.aksw.org/2025/corporate/`

Send these parameters via GET and you will receive a json object containing the query as answer.