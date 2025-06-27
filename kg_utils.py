from enum import Enum
import json
import logging
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Dict, Any

import requests
import tenacity
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rdflib.plugins.sparql.parser import parseQuery
from redis import StrictRedis
from redis_cache import RedisCache

# Cache Wikidata queries in Redis
client = StrictRedis(host="localhost", decode_responses=True)
cache = RedisCache(redis_client=client, prefix="wikidata")
# example of how to clean cache for a function: cached_requests.invalidate_all()

from chainlite import get_logger
import time

logger = get_logger(__name__)

inverse_property_path_regex = re.compile(
    "^(\?\w+)\s+wdt:P31\/wdt:P279\*\s+([^\s]+)\s*.$", re.IGNORECASE
)
count_regex = re.compile("COUNT\(\?\w+\)", re.IGNORECASE)

DATASETS = {
    "https://text2sparql.aksw.org/2025/dbpedia/": {
        "endpoint_url": os.environ.get("DBPEDIA_SPARQL_SERVICE_URL", "https://copper.coypu.org/text2sparql-2025-dbpedia"),
        "label": "DBpedia",
        "lookup_url": {
            "en": os.environ.get("DBPEDIA_LOOKUP_SERVICE_URL", "http://localhost:8082/api/search"),
            "es": os.environ.get("DBPEDIA_ES_LOOKUP_SERVICE_URL", "http://localhost:8083/api/search")
        }
    },
    "https://text2sparql.aksw.org/2025/corporate/": {
        "endpoint_url": os.environ.get("ORG_SPARQL_SERVICE_URL", "https://copper.coypu.org/text2sparql-2025-corporate-kg"),
        "label": "Corporate",
        "lookup_url": {
            "en": os.environ.get("ORG_LOOKUP_SERVICE_URL", "http://localhost:8084/api/search")
        }
    }
}

def try_to_optimize_query(query: str) -> str:
    # inverse property path
    matches = re.findall(inverse_property_path_regex, query)
    if len(matches) > 0:
        for m in matches:
            subst = f"{m[1]} ^wdt:P279*/^wdt:P31 {m[0]} ."
            query = re.sub(inverse_property_path_regex, subst, query, count=1)

    # count
    matches = re.findall(count_regex, query)
    if len(matches) > 0:
        for m in matches:
            subst = "COUNT(*)"
            query = re.sub(count_regex, subst, query, count=1)

    return query


current_script_directory = os.path.dirname(__file__)
jinja_environment = Environment(
    loader=FileSystemLoader(current_script_directory),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
    line_comment_prefix="#",
)


def fill_template(template_file, prompt_parameter_values={}):
    template = jinja_environment.get_template(template_file)

    filled_prompt = template.render(**prompt_parameter_values)
    filled_prompt = "\n".join(
        [line.strip() for line in filled_prompt.split("\n")]
    )  # remove whitespace at the beginning and end of each line
    return filled_prompt


class SPARQLResultsTooLargeError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = "Response too large, saving it in Redis will run into errors."
    
    def __str__(self):
        return f"SPARQLResultsTooLargeError: {self.message}"

class SPARQLSyntaxError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return f"SPARQLSyntaxError: {self.message}"
    
class SPARQLTimeoutError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return f"SPARQLTimeoutError: Query timed out"

def _serialize_error(exception):
    return json.dumps({
        "type": type(exception).__name__,
        "message": exception.message
    })
    
def _if_known_SPARQL_errors(exception_str):
    """
    Whether the (cached) input json object denotes one of:
    - SPARQLResultsTooLarge
    - SPARQL syntax error
        
    in which case directly raise these errors.
    """
    try:
        exc_info = json.loads(exception_str)
    except Exception:
        return
    
    if 'type' in exc_info and 'message' in exc_info:
        exc_type = exc_info['type']
        message = exc_info['message']

        if exc_type == "SPARQLResultsTooLargeError":
            raise SPARQLResultsTooLargeError("Response too large, saving it in Redis will run into errors.")
        
        if exc_type == "SPARQLSyntaxError":
            raise SPARQLSyntaxError(message)
        
        if exc_type == "SPARQLTimeoutError":
            raise SPARQLTimeoutError("timeout")

def _extract_wikidata_syntax_error(response: str):
    """
    Wikidata server returns a error str for a syntactic invalid SPARQL query directly,
    meaning that parsing it using response.json() could raise a JSON decoder error.
    
    In case of a JSON decoder error, go through the response text to determine if it is
    a syntactic compliant.
    """
    if response.startswith("Parse error:"): # typical fuseki error message start for syntax error
        return SPARQLSyntaxError(response)
    elif "java.util.concurrent.ExecutionException" in response:
        index = response.find("java.util.concurrent.ExecutionException")
        index_end = response.find("at java.util.concurrent", index)
        return SPARQLSyntaxError(response[index: index_end])
    elif "java.util.concurrent.TimeoutException" in response:
        return SPARQLTimeoutError("timeout")
    return None
        

@cache.cache()
def _cached_requests(url, params=None) -> tuple[dict, int]:
    """
    Make a GET request to the specified URL with the given parameters and return the response decoded as JSON.
    The result of this function is cached in Redis.

    If the response is too large(>10MB), return an error message to keep this from Redis cache.
    If the response is a known SPARQL error, return a JSON object with the error message.
    If the response is not a valid JSON and not a known SPARQL error, raise a JSONDecodeError.

    returns:
        - json object
        - status code
    """
    
    if params is not None:
        params = dict(params)
    
    start_time = time.time() # memorize start time for debugging output of the duration
    r = requests.get(
        url,
        params=params,
        timeout=70,
        headers={"User-Agent": "Stanford OVAL, WikiSP project"},
    )
    duration = time.time() - start_time
    responseStrReduced = str(r.content[:200]) + '...' if len(r.content) > 200 else r.content
    logging.debug(f"Requested {url} with params {params}, duration: {duration}s. Response code: {r.status_code}. Response (length: {len(r.content)}): {responseStrReduced}")

    if len(r.content) > 10 * 1024 * 1024:  # 10MB
        return _serialize_error(SPARQLResultsTooLargeError("Response too large, saving it in Redis will run into errors.")), 500

    try:
        res_json = r.json()
        return res_json, r.status_code
    except requests.exceptions.JSONDecodeError as e:
        syntax_error_object = _extract_wikidata_syntax_error(r.text)
        if syntax_error_object is not None:
            return _serialize_error(syntax_error_object), 500
        raise e
    except Exception as e:
        raise e


def cached_requests(url, params=None) -> tuple[dict, int]:
    """
    params: instead of a dict, it is a tuple of (k, v) so that it is hashable and we can cache it
    """
    res = _cached_requests(url, params)
    json_object, status_code = res
    
    # raise error if it is a known SPARQL error
    _if_known_SPARQL_errors(json_object)
    
    return json_object, status_code


def extract_id_from_uri(uri: str) -> str:
    if "wikidata.org" in uri:
        return uri[uri.rfind("/") + 1 :]
    else:
        # e.g. can be a statement like "statement/Q66072077-D5E883AA-1A7A-4F7C-A4B7-2723229A4385"
        return uri


def spans(mention):
    spans = []
    tokens = mention.split()
    for length in range(1, len(tokens) + 1):
        for index in range(len(tokens) - length + 1):
            span = " ".join(tokens[index : index + length])
            spans.append(span)
    return spans


def search_span(lookup_url: str, span: str, limit: int = 5, return_full_results=False, type="item"):
    """
    Searches for entities in DBpedia using the Lookup service.
    type should be one or more DBpedia classes from the ontology.
    """
    candidates = []
    params = {
        "query": span,
        "maxResults": limit,
        "format": "JSON"
    }

    if type:
        params["type"] = type

    response = requests.get(lookup_url, params=params)

    if response.status_code != 200:
        with open("parser_error.log", "a") as fd:
            fd.write(f"{span} threw error for search_span with status {response.status_code}\n")
        return []

    data = response.json()

    if "docs" not in data:
        return []

    results = data["docs"]
    if return_full_results:
        return results

    # Collecting the URIs of the results
    for result in results:
        candidates.append(result["id"][0])

    return candidates

def remove_whitespace_string(string):
    return "".join(string.split())


class SparqlExecutionStatus(str, Enum):
    OK = "ok"
    SYNTAX_ERROR = "syntax_error"
    TIMED_OUT = "timed_out"
    OTHER_ERROR = "other_error"
    TOO_LARGE = "too_large"

    @staticmethod
    def from_http_status_code(http_status_code: int):
        if http_status_code == 400:
            return SparqlExecutionStatus.SYNTAX_ERROR
        elif http_status_code > 400:
            return SparqlExecutionStatus.OTHER_ERROR
        else:
            return SparqlExecutionStatus.OK

    def __init__(self, default_message):
        self.default_message = default_message
        self.custom_message = None

    def set_message(self, msg):
        self.custom_message = msg

    def get_message(self):
        return self.custom_message if self.custom_message else self.default_message


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(requests.exceptions.RequestException),
    wait=tenacity.wait_random_exponential(min=20, exp_base=3),
    stop=tenacity.stop_after_attempt(3),
    after=tenacity.after_log(logger, logging.INFO),
)
def execute_sparql(
    dataset_id: str,
    sparql: str, return_status: bool = False
) -> bool | list | tuple[bool | list, SparqlExecutionStatus]:
    """
    For syntactically incorrect SPARQLs returns None
    """

    PREFIXES = {
        "dbr": "http://dbpedia.org/resource/",
        "dbo": "http://dbpedia.org/ontology/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "pv": "http://ld.company.org/prod-vocab/",
        "ecc": "https://ns.eccenca.com/",
        "owl": "http://www.w3.org/2002/07/owl#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "dct": "http://purl.org/dc/terms/",
        "void": "http://rdfs.org/ns/void#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "foaf": "http://xmlns.com/foaf/0.1/",
        "vann": "http://purl.org/vocab/vann/"
    }

    prefixes_to_add = ""
    for prefix, uri in PREFIXES.items():
        if f"PREFIX {prefix}:" not in sparql and f"{prefix}:" in sparql:
            prefixes_to_add += f"PREFIX {prefix}: <{uri}>\n"
    sparql = f"{prefixes_to_add}\n{sparql}"

    try:
        dataset_url = DATASETS[dataset_id]["endpoint_url"]
        r, status_code = cached_requests(
            dataset_url, params=tuple({"format": "json", "query": sparql}.items())
        )
        if status_code in {500, 400, 431, 413, 414}:
            # These are unrecoverable errors
            # 500: internal server error, 400: bad request (can happen when the SPARQL query is syntactically incorrect)
            # 431 for Request Header Fields Too Large
            # 413 for Content Too Large
            # 414 for URI Too Long for url
            logging.debug(f"Error: {status_code} for query: {sparql}")
            if return_status:
                return [], SparqlExecutionStatus.from_http_status_code(status_code)
            else:
                return []
        if status_code >= 400:
            logging.debug(f"Error: {status_code} for query: {sparql}")
            # 429, too many tries, would be included in this case.
            raise  # Reraise the exception so that we can retry using tenacity

        if "boolean" in r:
            res = r["boolean"]
        else:
            res = r["results"]["bindings"]
            if res == [] or (len(res) == 1 and res[0] == {}):
                res = []

    except requests.exceptions.ReadTimeout:
        logging.debug(f"Read Timeout catched for query: {sparql}")
        if return_status:
            return [], SparqlExecutionStatus.TIMED_OUT
        else:
            return []
    # except requests.exceptions.JSONDecodeError or json.decoder.JSONDecodeError:
    #     if return_status:
    #         return [], SparqlExecutionStatus.TIMED_OUT  # TODO it this always the case?
    #     else:
    #         return []
    except requests.exceptions.ConnectionError:
        logging.debug(f"Connection Error catched for query: {sparql}")
        if return_status:
            return [], SparqlExecutionStatus.OTHER_ERROR
        else:
            return []
    except SPARQLResultsTooLargeError:
        logging.debug(f"SPARQLResultsTooLargeError catched for query: {sparql}")
        if return_status:
            return [], SparqlExecutionStatus.TOO_LARGE
        else:
            return []
    except SPARQLTimeoutError:
        logging.debug(f"SPARQLTimeoutError catched for query: {sparql}")
        if return_status:
            return [], SparqlExecutionStatus.TIMED_OUT
        else:
            return []
    except SPARQLSyntaxError as e:
        logging.debug(f"Syntax Error catched: {e} for query: {sparql}")
        if return_status:
            status = SparqlExecutionStatus.SYNTAX_ERROR
            status.set_message(e.message)
            return [], status
        else:
            return []
    # except Exception as e:
    #     return res
    #     logger.exception(e)
    if return_status:
        return res, SparqlExecutionStatus.OK
    else:
        return res

def format_date(match):
    date_string = match.group(0)
    if date_string == "0000-00-00T00:00:00Z":
        return "0"
    if date_string.startswith("0000-"):
        # year zero is not allowed in Wikidata dates, but we still encounter it rarely
        return datetime.strptime(date_string[5:], "%m-%dT%H:%M:%SZ").strftime("%-d %B")
    return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ").strftime("%-d %B %Y")

def convert_if_date(x: str) -> str:
    if x is None:
        return x
    if type(x) is bool:
        return x
    return re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", format_date, x)

def check_sparql_syntax(query):
    try:
        # Attempt to parse the SPARQL query
        parse_result = parseQuery(query)
        # If parsing succeeds, the query is syntactically correct
        return True
    except Exception as e:
        # If parsing fails, the query is syntactically incorrect
        return False


def get_property_examples(datasetId: str, lang: str, pid: str):
    """Get a list of examples for a given property from DBpedia.

    Args:
        pid (str): e.g. "P155" or "dbpedia:P155"

    Returns:
        a list of tuples, [(subject, pid_label, object)], where in DBpedia
        there exists a tuple of the form:
        (subject, pid, object)
        and subject and object are already in their labels

        e.g., for P155, we will have:
        [..., ('April', 'follows', 'March'), ...]
        meaning that April follows March
    """

    if pid.startswith("http:"):
        pid = f"<{pid}>"
    elif not pid.startswith("dbo:"):
        pid = "dbo:" + pid


    sparql_query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT DISTINCT ?subLabel ?objLabel WHERE {{
      ?sub {pid} ?obj .
      
      OPTIONAL {{ ?sub rdfs:label ?subLabel_en. FILTER (lang(?subLabel_en) = "en")}}
      OPTIONAL {{ ?sub rdfs:label ?subLabel_. FILTER (lang(?subLabel_) = "")}}
      BIND(COALESCE(?subLabel_en, ?subLabel_, ?sub) AS ?subLabel)
      
      OPTIONAL {{ ?obj rdfs:label ?objLabel_en. FILTER (lang(?objLabel_en) = "en")}}
      OPTIONAL {{ ?obj rdfs:label ?objLabel_. FILTER (lang(?objLabel_) = "")}}
      BIND(COALESCE(?objLabel_en, ?objLabel_, ?obj) AS ?objLabel)
    }} LIMIT 5
    """

    res = execute_sparql(datasetId, sparql_query)
    try:
        res = [
            (
                i["subLabel"]["value"],
                pid,  # Assuming pid is the label for the property
                i["objLabel"]["value"],
            )
            for i in res
        ]
    except Exception as e:
        logger.warning(
            "Property %s threw exception %s in get_property_examples()", pid, str(e)
        )
    if not res:
        logger.warning("Property %s has no examples", pid)
    return res


def get_property_or_entity_description(p_or_qid: str):
    """Get label and description for a given property or entity

    Args:
        p_or_qid (str): e.g. "P155" or "wd:P155" or "wdt:P155"

    Returns:
        {
            "label": property label in English,
            "description": property description in English
        }
    """

    if p_or_qid.startswith("dbo:"):
        p_or_qid = p_or_qid.replace("dbo:", "")

    sparql_query = f"""
SELECT ?propertyLabel ?propertyDesc
WHERE {{
    dbo:{p_or_qid} rdfs:label ?propertyLabel .
    FILTER(LANG(?propertyLabel) = "en")
    OPTIONAL {{
        dbo:{p_or_qid} rdfs:comment ?propertyDesc .
        FILTER(LANG(?propertyDesc) = "en")
    }}
}}
"""
    res = execute_sparql(sparql_query)
    
    # not "propertyDesc" in res[0] would denote no description available
    if not res or not "propertyDesc" in res[0]:
        return None
    return {
        "label": res[0]["propertyLabel"]["value"],
        "description": res[0]["propertyDesc"]["value"],
    }


def get_type_information_for_uris(dataset_id: str, candidate_uris: list) -> Dict[str, dict]:
    candidate_uris_str = " ".join(f"<{uri}>" for uri in candidate_uris)

    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT * WHERE {{
      VALUES ?s {{
        {candidate_uris_str}
      }}
      ?s a ?cls .
      OPTIONAL {{
        ?cls rdfs:label ?cls_label . FILTER(LANG(?cls_label) = "en")
      }}
      OPTIONAL {{
        ?cls rdfs:comment ?cls_description . FILTER(LANG(?cls_description) = "en")
      }}
      OPTIONAL {{
        ?s rdfs:label ?s_label . FILTER(LANG(?s_label) = "en")
      }}
      OPTIONAL {{
        ?s rdfs:comment ?s_description . FILTER(LANG(?s_description) = "en")
      }}
      OPTIONAL {{
        ?s rdfs:domain ?s_domain .
        OPTIONAL {{
          ?s_domain rdfs:label ?s_domain_label . FILTER(LANG(?s_domain_label) = "en")
        }}
      }}
      OPTIONAL {{
        ?s rdfs:range ?s_range .
        OPTIONAL {{
          ?s_range rdfs:label ?s_range_label . FILTER(LANG(?s_range_label) = "en")
        }}
      }}
    }}
    """

    logger.debug(f"SPARQL query for get_candidate_types:\n{query}")

    results = execute_sparql(dataset_id, query)

    property_types = {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",
        "http://www.w3.org/2002/07/owl#ObjectProperty",
        "http://www.w3.org/2002/07/owl#DatatypeProperty"
    }

    result_dict = {}
    for res in results:
        uri = res["s"]["value"]
        entry = {}

        cls_uri = res.get("cls", {}).get("value") if isinstance(res.get("cls"), dict) else res.get("cls", None)
        if cls_uri:
            if cls_uri.startswith("http://www.w3.org/2002/07/owl#"):
                s_label = res.get("s_label", {}).get("value")
                s_description = res.get("s_description", {}).get("value")
                if s_label:
                    entry["type_label"] = s_label
                if s_description:
                    entry["type_description"] = s_description

            if cls_uri == "http://www.w3.org/2002/07/owl#Class":
                entry["type"] = "class"
            elif cls_uri in property_types:
                entry["type"] = "property"
                s_domain_label = res.get("s_domain_label", {}).get("value")
                s_range_label = res.get("s_range_label", {}).get("value")
                if s_domain_label:
                    entry["domain_label"] = s_domain_label
                if s_range_label:
                    entry["range_label"] = s_range_label
            else:
                entry["type"] = "entity"
                cls_label = res.get("cls_label", {}).get("value")
                cls_description = res.get("cls_description", {}).get("value")
                if cls_label:
                    entry["type_label"] = cls_label
                if cls_description:
                    entry["type_description"] = cls_description

        result_dict[uri] = entry

    return result_dict

def get_label_from_uri(uri: str) -> str:
    if '#' in uri:
        return uri.rsplit('#', 1)[-1]
    else:
        return uri.rsplit('/', 1)[-1]


@lru_cache()
def get_outgoing_edges(dataset_id: str, lang: str, entity: str, compact: bool):
    """
    QID example: Q679545
    compact: if True, will exclude the "Description" field from the output
    """
    logger.debug(f"\t- getting outgoing edges for <{entity}>")

    if entity.startswith("<") and entity.endswith(">"):
        entity = entity
    elif entity.startswith("http:"):
        entity = f"<{entity}>"
    elif not entity.startswith("dbr:"):
        entity = "dbr:" + entity.replace(" ", "_")

    ## TODO hier müsste man mit dem State vergleichen und da nach einer URL mit am aktuellen Label suchen
    ## da es immer wieder zu dem fehler kommt, dass AI nur Label und nicht URI einsetzt

    property_sparql = f"""
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    
    SELECT ?p ?v ?pLabel ?vLabel
    WHERE {{
        {entity} ?p ?v .
      
        FILTER(?p not in (rdfs:comment, rdfs:label, rdfs:isDefinedBy, foaf:depiction, rdfs:subClassOf))
        BIND(if(?p = rdf:type, "is a", COALESCE()) AS ?pLabel)
        OPTIONAL {{?p rdfs:label ?pLabel FILTER(LANG(?pLabel) = "en") }}
      
        OPTIONAL {{?v rdfs:label ?vLabel_en FILTER(LANG(?vLabel_en) = "en") }}
        OPTIONAL {{?v rdfs:label ?vLabel_ FILTER(LANG(?vLabel_) = "") }}
        BIND(COALESCE(?vLabel_en, ?vLabel_) AS ?vLabel)
    }}
    """

    logger.debug(f"{property_sparql}\n{dataset_id}")

    property_values: list = execute_sparql(dataset_id, property_sparql)

    all_properties = {}

    if property_values is not None:
        for pv in property_values:
            pid = extract_id_from_uri(pv["p"]["value"])

            if "pLabel" in pv:
                p_name = pv["pLabel"]["value"]
            else:
                p_name = get_label_from_uri(pid)

            if "vLabel" in pv:
                p_value_label = convert_if_date(pv["vLabel"]["value"])
            else:
                p_value_label = convert_if_date(pv["v"]["value"])
            p_value_id = extract_id_from_uri(pv["v"]["value"])
            key = f"{p_name} ({pid})"
            if key not in all_properties:
                all_properties[key] = {}
            if p_value_id.startswith("Q"):
                key2 = f"{p_value_label} ({p_value_id})"
            else:
                key2 = f"{p_value_label}"
            if key2 in all_properties[key]:
                # this indicates that it was already added by the qualifiers above
                continue
            all_properties[key][key2] = {}
            if "vDescription" in pv:
                all_properties[key][key2]["Description"] = pv["vDescription"]["value"]


    # print(all_properties)

    # TODO if compact == True, and if series ordinal is the only qualifier in all values, we can convert the whole thing into a list
    if compact:
        for p in all_properties:
            should_compact = True
            for v in all_properties[p]:
                if "Description" in all_properties[p][v]:
                    del all_properties[p][v]["Description"]
                if len(all_properties[p][v]) > 0:
                    should_compact = False
            if should_compact:
                all_properties[p] = [v for v in all_properties[p]]
            if len(all_properties[p]) == 1:
                if isinstance(all_properties[p], list):
                    all_properties[p] = all_properties[p][0]

    return all_properties

def normalize_result_string(result_string: str) -> str:
    """
    Helps when calculating EM or Superset metrics
    Sorts multiple answers alphabetically so that it is always consistent. Lowercases everything.
    """
    # Sometimes there are duplicates in the gold string. Remove them:
    results = list(set(result_string.split(";")))
    results = [r.strip() for r in results]
    return "; ".join(s.strip() for s in sorted(results)).lower()

import re

# Regex to match triples with predicate rdf:type or a, capturing subject and object
TRIPLE_PATTERN = re.compile(
    r"(\?\w+|<[^>]+>|[^\s]+)\s+(rdf:type|a)\s+(\?\w+|<[^>]+>|\"[^\"]*\")"
)

def rewrite_sparql_for_subclasses_regex(query: str) -> str:
    def repl(match):
        subj, pred, obj = match.groups()
        return f"{subj} rdf:type/rdfs:subClassOf* {obj}"
    return TRIPLE_PATTERN.sub(repl, query)


from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.term import URIRef, Variable
from rdflib.namespace import RDF, RDFS

def term_to_str(term):
    if isinstance(term, URIRef):
        return f"<{term}>"
    elif isinstance(term, Variable):
        return f"?{term}"
    else:
        return str(term)

def rewrite_sparql_for_subclasses_parser_based(query: str) -> str:
    """
    Parses the SPARQL query and replaces predicates rdf:type or 'a' with rdf:type/rdfs:subClassOf* in triple patterns.
    Returns the modified SPARQL query string.
    """
    parsed = parseQuery(query)
    algebra = translateQuery(parsed)

    # Helper function to recursively process algebra and replace predicates
    def process_algebra(expr):
        # If it's a BGP (basic graph pattern), process triples
        if expr.name == 'BGP':
            new_triples = []
            for triple in expr.triples:
                s, p, o = triple
                # Replace predicate if rdf:type or 'a'
                if (p == RDF.type) or (isinstance(p, URIRef) and str(p) == 'a'):
                    # Replace predicate with property path rdf:type/rdfs:subClassOf*
                    # rdflib does not support property paths in triples directly,
                    # so we replace predicate with a string representing the path
                    # This is a limitation; we will reconstruct the query string manually below.
                    new_triples.append((s, "rdf:type/rdfs:subClassOf*", o))
                else:
                    new_triples.append(triple)
            expr.triples = new_triples
        # Recursively process children if any
        for k, v in expr.items():
            if isinstance(v, list):
                for item in v:
                    if hasattr(item, 'name'):
                        process_algebra(item)
            elif hasattr(v, 'name'):
                process_algebra(v)

    process_algebra(algebra.algebra)

    query_type = parsed[1].name

    triples = []

    def extract_triples(expr):
        if expr.name == 'BGP':
            for s, p, o in expr.triples:
                pred_str = p if isinstance(p, str) else f"<{p}>"
                s_str = term_to_str(s)
                o_str = term_to_str(o)
                triples.append(f"  {s_str} {pred_str} {o_str} .")
        for k, v in expr.items():
            if isinstance(v, list):
                for item in v:
                    if hasattr(item, 'name'):
                        extract_triples(item)
            elif hasattr(v, 'name'):
                extract_triples(v)

    extract_triples(algebra.algebra)
    where_clause = "\n".join(triples) + "\n}"

    if query_type == 'SelectQuery':
        select_vars = parsed[1].vars
        if select_vars is None:
            select_clause = "SELECT * WHERE {"
        else:
            select_clause = "SELECT " + " ".join(f"?{v}" for v in select_vars) + " WHERE {"
        return f"{select_clause}\n{where_clause}"
    elif query_type == 'AskQuery':
        return f"ASK WHERE {{\n{where_clause}"
    else:
        # fallback: return original query if unsupported type
        return query



if __name__ == "__main__":
#     dataset_id = "https://text2sparql.aksw.org/2025/corporate/"
#
#     entities = "Santa Claus"
#     print(search_span(DATASETS[dataset_id]["lookup_url"], entities))
#
#     types = get_candidate_types(dataset_id,
#                         ["http://ld.company.org/prod-instances/srv-O662-4012383",
#                         "http://ld.company.org/prod-vocab/Service",
#                          "http://ld.company.org/prod-vocab/amount",
#                         "http://ld.company.org/prod-instances/dept-41622"]
#                         )
#     print(json.dumps(types, indent=4))
#
#     import redis
#     r = redis.Redis(host='localhost', port=6379, decode_responses=True)
#     property = "dbo:birthPlace"
#     print(get_property_examples(dataset_id, property))
#
#     sparql="""
# PREFIX dbo: <http://dbpedia.org/ontology/>
# PREFIX dbr: <http://dbpedia.org/resource/>
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
# PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
#     SELECT ?p ?v ?pLabel ?vLabel
#     WHERE {
#         dbr:Salt_Lake_City ?p ?v .
#
#         FILTER(?p not in (rdfs:comment, rdfs:label))
#         BIND(if(?p = rdf:type, "is a", COALESCE()) AS ?pLabel)
#         OPTIONAL {?p rdfs:label ?pLabel FILTER(LANG(?pLabel) = "en") }
#
#         OPTIONAL {?v rdfs:label ?vLabel FILTER(LANG(?vLabel) = "en") }
#
#   FILTER(STRSTARTS(STR(?p), STR(dbo:)))
#
#     }
#
#     """
#     print(execute_sparql(sparql, return_status=True))

    # Example usage
    original_sparql = """
        SELECT ?s WHERE {
            ?s rdf:type <http://example.org/ClassName> .
            ?s a <http://example.org/ClassName> .
        }
        """

    modified_sparql = rewrite_sparql_for_subclasses_regex(original_sparql)
    print(modified_sparql)

    modified_sparql = rewrite_sparql_for_subclasses_parser_based(original_sparql)
    print(modified_sparql)

    original_sparql = """
        ASK WHERE {
            ?s rdf:type <http://example.org/ClassName> .
            ?s a <http://example.org/ClassName> .
        }
        """
    modified_sparql = rewrite_sparql_for_subclasses_parser_based(original_sparql)
    print(modified_sparql)

    original_sparql = """
    SELECT ?service ?serviceName (xsd:decimal(?priceValue) AS ?numericPrice)
WHERE {
  ?service a <http://ld.company.org/prod-vocab/Service> ;
           <http://ld.company.org/prod-vocab/name> ?serviceName ;
           <http://ld.company.org/prod-vocab/price> ?priceStr .
  #BIND(STRREPLACE(?priceStr, ',', '.') AS ?priceDot)
  BIND(STRAFTER(?priceDot, ' ') AS ?priceNumberStr)
  BIND(xsd:decimal(?priceNumberStr) AS ?priceValue)
}
ORDER BY DESC(?priceValue)
LIMIT 1
    """
    modified_sparql = rewrite_sparql_for_subclasses_regex(original_sparql)
    print(modified_sparql)