import json
import sys

from typing import List, Dict, Any
from chainlite import chain, get_logger, llm_generation_chain, pprint_chain
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnablePassthrough

from langdetect import detect

from spinach_agent.parser_state import (
    Action,
    PartToWholeParserState,
    SparqlQuery,
    state_to_dict,
    state_to_string,
)
from spinach_agent.parser_utils import (
    BaseParser,
    execute_sparql_object,
    get_prune_edges_chain,
    parse_string_to_json,
    sparql_string_to_sparql_object,
)

from kg_utils import (
    SparqlExecutionStatus,
    get_outgoing_edges,
    search_span,
    get_property_examples,
    DATASETS,
    get_type_information_for_uris
)

logger = get_logger(__name__)

def remove_html_tags(text: str):
    text = text.strip()
    text = text.replace("<B>", "").replace("</B>", "")
    return text

def format_search_result(arr):
    ret = []
    for item in arr:
        label = item.get(
            "label", ""
        )
        # the lookup service does return a list here, we just pick the first element
        if isinstance(label, list):
            label = label[0]
        label = remove_html_tags(label)

        # Get the id or an empty string if not available
        id = item.get("id", "")
        id = id[0]

        # Get the description or an empty string if not available
        description = item.get("comment", "")
        if isinstance(description, list):
            description = description[0]
        description = remove_html_tags(description)

        display_string = f" - {label} ({id}): {description}"

        ret.append(display_string)
    return "\n".join(ret)

@chain
async def json_to_string(j: dict) -> str:
    return json.dumps(j, indent=2, ensure_ascii=False)


@chain
async def json_to_action(action_dict: dict) -> Action:
    thought = action_dict["thought"]
    action_name = action_dict["action_name"]
    action_argument = action_dict["action_argument"]

    if action_name == "execute_sparql":
        assert action_argument, action_dict

    return Action(
        thought=thought,
        action_name=action_name,
        action_argument=action_argument,
    )


# async def sparql_debugger_router(state):
#     # last action was an execute_sparql()
#     current_action = PartToWholeParser.get_current_action(state)
#     assert current_action.action_name == "execute_sparql"
#     sparql_to_debug = state["generated_sparqls"][-1]
#     sparqls_to_help_debug = []
#     # TODO feed in previous observations of PIDs and QIDs and entity pages as well
#     if not sparql_to_debug.has_results():
#         print("sparql_to_debug = ", sparql_to_debug)
#         for _ in range(15):
#             o = await PartToWholeParser.debug_sparql_chain.ainvoke(
#                 {
#                     "sparql_to_debug": sparql_to_debug,
#                     "sparqls_to_help_debug": sparqls_to_help_debug,
#                 }
#             )
#             print("sparql to help debug =  ", o)
#             print("execution result:", o.execution_result)
#             sys.stdout.flush()
#             sparqls_to_help_debug.append(o)
#         exit()
#     return "controller"

def display_actions(actions):
    # truncate actions
    actions = actions[-10:]
    
    action_history = []
    for i, a in enumerate(actions):
        include_observation = True
        if i < len(actions) - 2 and a.action_name in [
            "get_knowledgegraph_entry",
            "search_knowledgegraph",
        ]:
            include_observation = False
        if a.action_name == "stop":
            include_observation = False
        action_history.append(a.to_jinja_string(include_observation))

    # print(action_history)
    return action_history


class PartToWholeParser(BaseParser):
    @classmethod
    def initialize(
        cls,
        engine: str,
        dataset_id: str
    ):
        @chain
        async def initialize_state(input):
            return PartToWholeParserState(
                question=input["question"],
                conversation_history=input["conversation_history"],
                questionId=input.get("questionId"),
                engine=engine,
                action_counter=0,
                actions=[],
                response="",
                dataset=dataset_id,
                language=detect(input["question"])
                # generated_sparqls=[],
            )

        # build the graph
        graph = StateGraph(PartToWholeParserState)
        graph.add_node("start", lambda x: {})
        graph.add_node("controller", PartToWholeParser.controller)
        graph.add_node("execute_sparql", PartToWholeParser.execute_sparql)
        graph.add_node("get_knowledgegraph_entry", PartToWholeParser.get_knowledgegraph_entry)

        graph.add_node("search_entity_by_label", PartToWholeParser.search_entity_by_label)
        graph.add_node("search_property_by_label", PartToWholeParser.search_property_by_label)
        graph.add_node("search_class_by_label", PartToWholeParser.search_class_by_label)

        graph.add_edge("search_entity_by_label", "controller")
        graph.add_edge("search_property_by_label", "controller")
        graph.add_edge("search_class_by_label", "controller")

        graph.add_node("get_property_examples", PartToWholeParser.get_property_examples)
        graph.add_node("reporter", PartToWholeParser.reporter)
        graph.add_node("stop", PartToWholeParser.stop)

        graph.set_entry_point("start")

        graph.add_edge("start", "controller")
        graph.add_conditional_edges(
            "controller",
            PartToWholeParser.router,  # the function that will determine which node is called next.
        )
        for n in [
            "execute_sparql",
            "get_knowledgegraph_entry",
            "get_property_examples",
        ]:
            graph.add_edge(n, "controller")

        graph.add_edge("stop", "reporter")
        graph.add_edge("reporter", END)

        # graph.add_edge("execute_sparql", sparql_debugger_router)

        dataset_name = DATASETS[dataset_id]

        cls.controller_chain = (
            {
                "input": llm_generation_chain(
                    template_file="controller.prompt",
                    engine=engine,
                    max_tokens=700,
                    temperature=1.0,
                    top_p=0.9,
                    # stop_tokens=["Observation:"],
                    keep_indentation=True,
                    # bind_prompt_values={"dataset": dataset_name}
                )
            }
            | llm_generation_chain(
                template_file="format_actions.prompt",
                engine=engine,
                max_tokens=700,
                keep_indentation=True,
                output_json=True,
                # bind_prompt_values={"dataset": dataset_name}
            )
            | parse_string_to_json
            | json_to_action
        )

        cls.sparql_chain = sparql_string_to_sparql_object | execute_sparql_object
        cls.prune_edges_chain = get_prune_edges_chain(engine=engine, dataset_id=dataset_id) | json_to_string
        cls.reporter_chain = llm_generation_chain(
            template_file="conversation_reporter.prompt",
            engine=engine,
            max_tokens=2000,
            temperature=0.0,
            keep_indentation=True,
            # bind_prompt_values={"dataset": dataset_name}
        )
        cls.language_detection_chain = llm_generation_chain(
                    template_file="language_detection.prompt",
                    engine=engine,
                    max_tokens=50,
                    temperature=0.0,
                    top_p=0.9,
                    keep_indentation=True,
                )

        # cls.debug_sparql_chain = (
        #     llm_generation_chain(
        #         template_file="sparql_debugger.prompt", engine=engine, max_tokens=1000
        #     )
        #     | extract_code_block_from_output.bind(code_block="sparql")
        #     | cls.sparql_chain
        # )

        compiled_graph = graph.compile()
        cls.runnable = initialize_state | compiled_graph
        logger.info("Finished initializing the graph.")
        # compiled_graph.get_graph().print_ascii()
        # sys.stdout.flush()

    @staticmethod
    def get_current_action(state):
        return state["actions"][-1]

    @staticmethod
    async def router(state):
        move_back_on_duplicate_action = 2
        current_action = PartToWholeParser.get_current_action(state)
        if current_action in state["actions"][-5:-1]:
            logger.warning(
                "Took duplicate action %s, going back %d steps.",
                current_action.action_name,
                move_back_on_duplicate_action,
            )
            # current_action.observation = "I have already taken this action. I should not repeat the same action twice."
            # Remove generated_sparqls as well
            if len(state["actions"]) - 2 >= 0:
                for i in range(len(state["actions"]) - 2, -1, -1):
                    if state["actions"][i].action_name == "execute_sparql":
                        state["generated_sparqls"] = state["generated_sparqls"][:-1]
            state["actions"] = state["actions"][:-move_back_on_duplicate_action]
            state["action_counter"] -= move_back_on_duplicate_action
            return "controller"

        if state["action_counter"] >= 15:
            return "reporter"
        
        return state["actions"][-1].action_name

    @staticmethod
    @chain
    async def controller(state):
        print("\n##############################################################################################")
        print(f"###########  Q: {state['question']} (QID: { state['questionId']}) #############")
        print("##############################################################################################\n")
        if state["actions"]:
            print("Last-Action: \n", state["actions"][-1])
            print("\n....................................\n")
            sys.stdout.flush()

        # make the history shorter
        actions = state["actions"]
        action_history = display_actions(actions)

        # try two times if there is an assertion error,
        # happens if an action not specified is chosen
        # if fails after 2 times, end the reasoning.
        attempts = 3
        for attempt in range(attempts):
            try:
                action = await PartToWholeParser.controller_chain.ainvoke(
                    {
                        "conversation_history": state["conversation_history"],
                        "question": state["question"],
                        "action_history": action_history,
                        "dataset": DATASETS[state["dataset"]]["label"]
                    }
                )
                print("Current-Action: \n", action)
                return {"actions": action, "action_counter": 1}
            except AssertionError:
                if attempt == attempts - 1:
                    return {"actions": Action(thought="", action_name="stop", action_argument=""), "action_counter": 1}
        # action = await PartToWholeParser.controller_chain.ainvoke(
        #     {
        #         "conversation_history": state["conversation_history"],
        #         "question": state["question"],
        #         "action_history": action_history,
        #         "dataset": DATASETS[state["dataset"]]["label"]
        #     }
        # )
        # TODO should never be reached
        print("Current-Action never be reached: \n", action)
        return {"actions": action, "action_counter": 1}

    @staticmethod
    @chain
    async def execute_sparql(state):
        print("Executing SPARQL query ...")
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "execute_sparql"
        from kg_utils import rewrite_sparql_for_subclasses_regex
        sparql_query = current_action.action_argument

        # rewrite with property path rdf:type/rdfs:subClassOf* to gather inferred instances
        if state['dataset'] == "https://text2sparql.aksw.org/2025/corporate/":
            sparql_query = rewrite_sparql_for_subclasses_regex(sparql_query)
            print("rewritten query: ", sparql_query, "\n")
        sparql = PartToWholeParser.sparql_chain.invoke({
            "sparql": sparql_query,
            "datasetId": state["dataset"]
        })
        current_action.action_argument = (
            sparql.sparql
        )  # update it with the cleaned and optimized SPARQL
        if sparql.has_results():
            current_action.observation = sparql.results_in_table_format()
            current_action.observation_markdown = sparql.results_in_markdown_format()
        else:
            if sparql.execution_status == SparqlExecutionStatus.SYNTAX_ERROR:
                msg = sparql.execution_status.get_message()
                current_action.observation = "Query had syntax error." if not msg else msg
            elif sparql.execution_status == SparqlExecutionStatus.TIMED_OUT:
                current_action.observation = "Query execution timed out."
            elif sparql.execution_status == SparqlExecutionStatus.OTHER_ERROR:
                current_action.observation = "Query execution ran into an error."
            else:
                current_action.observation = "Query returned empty result."

        return {"generated_sparqls": sparql}

    @staticmethod
    @chain
    async def get_knowledgegraph_entry(state):
        print("\nGetting knowledge graph entity data ...")
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "get_knowledgegraph_entry"
        wikidata_entry = get_outgoing_edges(
            state["dataset"],
            state["language"],
            current_action.action_argument, compact=True
        )
        # NOTE: The only seen exception from this so far
        # is due to contex too long
        try:
            action_result = await PartToWholeParser.prune_edges_chain.ainvoke(
                {
                    "question": state["question"],
                    "conversation_history": state["conversation_history"],
                    "outgoing_edges": json.dumps(
                        wikidata_entry, indent=2, ensure_ascii=False
                    ),
                    "entity_and_description": current_action.action_argument,
                }
            )
        except Exception:
            action_result = ""
        current_action.observation = action_result

    @staticmethod
    @chain
    async def search_knowledgegraph_entity(state):
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "search_knowledgegraph"
        dataset_id = state["dataset"]
        lang = state["language"]
        lookup = DATASETS[dataset_id]["lookup_url"]
        if lang in lookup:
            lookup_url = DATASETS[dataset_id]["lookup_url"][lang]
        else:
            lookup_url = DATASETS[dataset_id]["lookup_url"]["en"]

        action_results = search_span(
            lookup_url,
            current_action.action_argument,
            limit=8,
            return_full_results=True,
            type="item",
        )
        # action_results += search_span(
        #     current_action.action_argument,
        #     limit=4,
        #     return_full_results=True,
        #     type="property",
        # )


        # workaround for missing comments in Org dataset
        if dataset_id == "https://text2sparql.aksw.org/2025/corporate/":
            from kg_utils import get_type_information_for_uris
            candidate_types = get_type_information_for_uris(dataset_id, [entry["id"][0] for entry in action_results])

            for e in action_results:
                uri = e["id"][0]

                if uri in candidate_types:
                    type_data = candidate_types[uri]

                    comment = ""
                    if "type" in type_data:
                        if type_data["type"] == "entity":
                            if "type_label" in type_data:
                                comment += " is a " + type_data["type_label"].lower()
                        else:
                            comment += " is a " + type_data["type"]
                            if "type_description" in type_data:
                                comment += f" described as '{type_data['type_description'].lower()}'"
                            if type_data["type"] == "property":
                                if "domain_label" in type_data:
                                    comment += f"| Domain: '{type_data['domain_label'].lower()}'"
                                if "range_label" in type_data:
                                    comment += f"| Range: '{type_data['range_label'].lower()}'"

                    e["comment"] = comment


        current_action.observation = format_search_result(action_results)
        print(f"\nExecuting Knowledge Graph lookup... showing results for search '{current_action.action_argument}':\n{current_action.observation}")


    @staticmethod
    @chain
    async def get_property_examples(state):
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "get_property_examples"
        examples = get_property_examples(state["dataset"], state["language"], current_action.action_argument)
        action_result = []
        try:
            for e in examples:
                action_result.append(f"{e[0]} -- {e[1]} --> {e[2]}")
        except Exception as e:
            logger.exception(e)
        current_action.observation = "\n".join(action_result)

    @staticmethod
    @chain
    async def stop(state):
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "stop"
        for s in state["generated_sparqls"]:
            assert isinstance(s, SparqlQuery)
        if (
            len(state["generated_sparqls"]) == 0
            or not state["generated_sparqls"][-1].has_results()
        ):
            logger.warning("Stop() was called without a good SPARQL. Starting over.")
            state["generated_sparqls"] = []
            state["actions"] = []
            state["action_counter"] = 0
            return

        logger.info("Finished run for question %s", state["question"])
        final_sparql = state["generated_sparqls"][-1]
        if final_sparql.sparql.strip().endswith("LIMIT 10"):
            logger.info("Removing LIMIT 10 from the final SPARQL")
            s = final_sparql.sparql.strip()[:-len("LIMIT 10")]
            final_sparql = SparqlQuery(sparql=s)
            final_sparql.execute()

        return {"final_sparql": final_sparql}

    @staticmethod
    @chain
    async def reporter(state):
        actions = state["actions"]
        action_history = []
        for i, a in enumerate(actions):
            include_observation = True
            if i < len(actions) - 2 and a.action_name in [
                "get_knowledgegraph_entry",
                "search_entity_by_label",
                "search_property_by_label",
                "search_class_by_label",
            ]:
                include_observation = False
            elif a.action_name == "stop":
                include_observation = False
            action_history.append(a.to_jinja_string(include_observation))
        
        language = await PartToWholeParser.language_detection_chain.ainvoke(
            {
                "question": state["question"],
            }
        )
        
        response = await PartToWholeParser.reporter_chain.ainvoke(
            {
                "language": language,
                "dataset": DATASETS[state["dataset"]]["label"],
                "question": state["question"],
                "conversation_history": state["conversation_history"],
                "action_history": action_history
            }
        )
        state["response"] = response
        return {"response": response}

    @staticmethod
    @chain
    async def search_entity_by_label(state: Dict[str, Any]) -> None:
        logger.debug("search entity")
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "search_entity_by_label"

        dataset_id = state["dataset"]
        lang = state["language"]
        lookup_url = DATASETS[dataset_id]["lookup_url"].get(lang) or DATASETS[dataset_id]["lookup_url"].get("en")

        action_results = search_span(
            lookup_url,
            current_action.action_argument,
            limit=10,
            return_full_results=True,
            type="item",
        )

        uris = [entry["id"][0] for entry in action_results]
        candidate_types = get_type_information_for_uris(dataset_id, uris)

        # Filter results to only those with type 'entity'
        res = [
            e for e in action_results
            if (uri := e["id"][0]) in candidate_types
               and candidate_types[uri].get("type") == "entity"
        ]

        current_action.observation = format_search_result(res)
        logger.debug(f"Entity search for '{current_action.action_argument}' returned:\n{current_action.observation}")


    @staticmethod
    @chain
    async def search_property_by_label(state: Dict[str, Any]) -> None:
        logger.debug("search property")
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "search_property_by_label"

        dataset_id = state["dataset"]
        lang = state["language"]
        lookup_url = DATASETS[dataset_id]["lookup_url"].get(lang) or DATASETS[dataset_id]["lookup_url"].get("en")

        action_results = search_span(
            lookup_url,
            current_action.action_argument,
            limit=10,
            return_full_results=True,
            type="property",
        )

        uris = [entry["id"][0] for entry in action_results]
        candidate_types = get_type_information_for_uris(dataset_id, uris)

        # Filter and enrich results with domain/range info and comments
        res = []
        for e in action_results:
            uri = e["id"][0]
            type_data = candidate_types.get(uri)
            if not type_data or type_data.get("type") != "property":
                continue

            comment_parts = []
            domain_label = type_data.get("domain_label")
            if domain_label:
                comment_parts.append(f"Domain: '{domain_label.lower()}'")
            range_label = type_data.get("range_label")
            if range_label:
                comment_parts.append(f"Range: '{range_label.lower()}'")

            comment_suffix = ""
            if comment_parts:
                comment_suffix = "| " + " | ".join(comment_parts)

            old_comment = e.get("comment", "")
            type_desc = type_data.get("type_description", "").lower()

            if isinstance(old_comment, list):
                new_comment = [f"{c} {type_desc} {comment_suffix}".strip() for c in old_comment]
            else:
                new_comment = f"{old_comment} {type_desc} {comment_suffix}".strip()

            e["comment"] = new_comment
            res.append(e)

        current_action.observation = format_search_result(res)
        logger.debug(f"\nProperty search for '{current_action.action_argument}' returned:\n{current_action.observation}")


    @staticmethod
    @chain
    async def search_class_by_label(state: dict) -> None:
        logger.debug("search class")
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "search_class_by_label"

        dataset_id = state["dataset"]
        lang = state["language"]
        lookup_url = DATASETS[dataset_id]["lookup_url"].get(lang, DATASETS[dataset_id]["lookup_url"].get("en"))

        action_results = search_span(
            lookup_url,
            current_action.action_argument,
            limit=10,
            return_full_results=True,
            type="class",
        )

        uris = [entry["id"][0] for entry in action_results]
        candidate_types = get_type_information_for_uris(dataset_id, uris)

        res = []
        for e in action_results:
            uri = e["id"][0]
            type_data = candidate_types.get(uri)
            if not type_data or type_data.get("type") != "class":
                continue

            if not e.get("comment") and type_data.get("type_description"):
                e["comment"] = type_data["type_description"].lower()

            res.append(e)

        current_action.observation = format_search_result(res)
        logger.debug(f"\nClass search for '{current_action.action_argument}' returned:\n{current_action.observation}")
