from .logger import LOGGER
from ..models.agent_state import AgentState
from ..models.pydantic_models import FinancialQueries, SQLScript

from ..service.gemini_service import GeminiSimpleChatEngine, GeminiJsonEngine
from ..service.generic_utils import load_json_data, save_json_data, load_cached_results
from ..service.sql_engine_service import get_title_data_inmemory_db, TITLE_DATA_INMEM_DB

import pandas as pd
import hashlib
import json
from typing import Dict, Any, List, Union



############################################################################################
# Agent Function 1
# Name: start_agent
# LLM-Call: No
# Description: It is used to start the agent-assist
############################################################################################

def start_agent(state: AgentState):
    LOGGER.info("Starting the agent-assist")
    state['state'] = 'start'
    return state




############################################################################################
# Agent Function 2
# Name: end_agent
# LLM-Call: No
# Description: It is used to end the agent-assist and print the END message
############################################################################################

def end_agent(state: AgentState):
    LOGGER.info("Ending the agent-assist")
    state['state'] = 'end'
    return state







############################################################################################
# Agent Function 3
# Name: refine_old_summary_agent
# LLM-Call: Yes
# Description: It takes the old summaries and refines them
############################################################################################

def refine_old_summary_agent(state: AgentState):
    state['state'] = 'refine_old_summaries'
    LOGGER.info(f"State: {state['state']} | Initializing the agent to refine old summaries")

    # Load the data from cache if the cache flag is set to True
    cached_result = load_cached_results(state)
    if cached_result:
        return state
    
    # Initialize the model
    gemini_simple_chat_engine = GeminiSimpleChatEngine(model_name=state['model']['model_name'], 
                                                   temperature=state['model']['temperature'],
                                                   max_output_tokens=state['model']['max_output_tokens'],
                                                   systemInstructions="You are an expert financial bot. You will be given a financial report and you need to refine the report. Keep everything in a single large paragraph. Dont use any markdown or bullet points. ",
                                                   max_retries=state['model']['max_retries'],
                                                   wait_time=state['model']['wait_time'])

    # Old summaries from sample_summarized_pnl_commentaries (Note: This is a sample data not cached, admin will provide the data)
    print(state['cache_location']['sample_summarized_pnl_commentaries'])
    sample_summarized_pnl_commentaries = load_json_data(state['cache_location']['sample_summarized_pnl_commentaries'])
    print(sample_summarized_pnl_commentaries)
    LOGGER.info(f"State: {state['state']} | Loaded the sample data, {len(sample_summarized_pnl_commentaries)} old summaries found")
    
    # Refine the old summaries
    result = []

    for summary in sample_summarized_pnl_commentaries:
        _refinement_prompt = [
            f"Given financial report: {summary}",
            f"Please refine the financial report in a more readable and meangingful way without losing any important information and entitites and technical/financial terms. Dont unnecessarily change the meaning of the report and dont increase the length of the report. "
        ]
        refined_summary = gemini_simple_chat_engine(_refinement_prompt)
        result.append(refined_summary)
        LOGGER.info(f"State: {state['state']} | Summary refined , {summary[:30]}... to {refined_summary[:30]}...")

    # Save the result to state var and set the cache flag to True
    state['results'][state['state']] = result
    state['cache_flag'][state['state']] = True

    # Save the result to cache with the state name and {result} key
    save_json_data(state['cache_location'][state['state']], {"result":state['results'][state['state']]})
    
    LOGGER.info(f"State: {state['state']} | Refinement of old summaries completed, saved the result to cache and set the cache flag to True")
    return state







############################################################################################
# Agent Function 4
# Name: generate_subj_query_agent
# LLM-Call: Yes
# Description: It generates the subjective queries. It uses the refined summaries, rule based templates and generates the queries
############################################################################################

def generate_subj_query_agent(state: AgentState):
    state['state'] = 'subj_query_generation'
    LOGGER.info(f"State: {state['state']} | Initializing the agent to refine old summaries")

    # Load the data from cache if the cache flag is set to True
    cached_result = load_cached_results(state)
    if cached_result:
        return state
    
    # Initialize the model
    fin_qry_engine =  GeminiJsonEngine(
                                    model_name=state['model']['model_name'],
                                    basemodel=FinancialQueries,
                                    temperature=state['model']['temperature'],
                                    max_output_tokens=state['model']['max_output_tokens'],
                                    systemInstructions=None,
                                    max_retries=state['model']['max_retries'],
                                    wait_time=state['model']['wait_time'])

    # Outputs from the previous state
    refined_sample_summarized_pnl_commentaries = state['results']['refine_old_summaries']

    # Prompt
    title_comment_template = "For Buisness <BUIS>, on  <DATE>, driven by <NET>$  <FACTOR> <PROF_LOSS> to PL on <CUR> Currency on Portfolio <PF> and Desk <DSK>"
    user_prompt_list = [
    "You are a financial assistant. Your task is to generate structured queries from given templates to create financial summaries.",
    
    f"Here is an example pattern for financial summaries: {refined_sample_summarized_pnl_commentaries[0]}.",
    
    f"You are provided with a list of rule-based templates in the format List[{title_comment_template}]. Extract meaningful queries from these templates.",
    
    """Generate at least 15 diverse queries that can be used to generate sample financial summaries.
    
    - The queries should focus on aggregations such as min, max, mean, and sum, or retrieve the top 5 / bottom 5 entities.  
    - Avoid queries that fetch all rows or list all entities without aggregation.  
    - Do not create separate queries for different aggregations on the same entity; instead, combine them into a single query.  
    - Dont ask for a particular value; instead, ask for a top k or bottom k value. Say, top 5 Business Units or bottom 5 Desks.
    - The queries should be sufficient to address the financial summary patterns mentioned above.  
    - Replace all field values with placeholders using the format <FIELD>. Do not include actual values.  
    - Do not summarize the data; just generate structured queries.""",
    
    "You must use the tool `FinancialQueries`. Your response must strictly follow the argument structure of `FinancialQueries`."
    ]

    # Generate queries
    queries = fin_qry_engine(user_prompt_list)[0]['queries']
    LOGGER.info(f"State: {state['state']} | Generated {len(queries)} queries")

    # Save the result to state var and set the cache flag to True
    state['results'][state['state']] = queries
    state['cache_flag'][state['state']] = True

    # Assign id to each query with sha hash
    for query in state['results'][state['state']]:
        query['id'] = hashlib.sha256(json.dumps(query).encode()).hexdigest()
        query['flag'] = 'hot'

    # Save the result to cache with the state name and {result} key
    save_json_data(state['cache_location'][state['state']], {"result":state['results'][state['state']]})

    LOGGER.info(f"State: {state['state']} | Query generation completed, saved the result to cache and set the cache flag to True")
    return state






############################################################################################
# Agent Function 5
# Name: generate_stat_query_agent
# LLM-Call: No
# Description: It generates the statistical queries like average, max, min, variance, sum etc. It contains some generic statistical queries
############################################################################################

def generate_stat_query_agent(state: AgentState):
    state['state'] = 'stat_query_generation'
    LOGGER.info(f"State: {state['state']} | Initializing the agent to refine old summaries")

    # Load the data from cache if the cache flag is set to True
    cached_result = load_cached_results(state)
    if cached_result:
        return state
    
    ## It is not an AI task, it is a rule-based task. So, we can directly write the code here.

    statistical_queries = [
        "What is average, max, min, varaince, sum of <NET> profit/loss?",
        "What is average, max, min, varaince, sum of <FACTOR> profit/loss grouped by <BUIS>?",
        "What is average, max, min, varaince, sum of <NET> profit/loss grouped by <CUR> currency?",
        "What is average, max, min, varaince, sum of <NET> profit/loss grouped by top 5 <PF> portfolios?",
        "What is average, max, min, varaince, sum of <NET> profit/loss grouped by bottom 5 <PF> portfolios?",
        "What is average, max, min, varaince, sum of <NET> profit/loss grouped by top 5 <DSK> desks?",
        "What is average, max, min, varaince, sum of <NET> profit/loss grouped by bottom 5 <DSK> desks?",
        "What are the top currencies by average <NET> profit/loss?",
        "What are the bottom currencies by average <NET> profit/loss?",
        "What is the total count of transactions for each <FACTOR>?",
        "What is the percentage contribution of each <FACTOR> to total profit/loss?",
        "What is the trend of total <NET> profit/loss over time (daily, monthly, yearly)?",
        "What is the moving average of <NET> profit/loss over the past 7 days?",
        "What is the standard deviation of <NET> profit/loss grouped by <BUIS>?",
        "What is the correlation between <FACTOR> and <NET> profit/loss?",
        "What is the skewness and kurtosis of <NET> profit/loss distribution?",
        "Which <PF> portfolios have the highest standard deviation in <NET> profit/loss?",
        "Which <DSK> desks have the highest variance in <NET> profit/loss?",
        "What is the probability distribution of <NET> profit/loss?",
        "What is the cumulative sum of <NET> profit/loss over time?",
        "Which <CUR> currency has the most volatile <NET> profit/loss?",
        "What is the ratio of profitable to loss-making transactions per <FACTOR>?",
        "Which <PF> portfolios have the most consistently positive (low variance) profits?",
        "Which <FACTOR> contributes most to total profit/loss variance?",
        "What is the ratio of profit to loss per <DSK> desk?",
        "Which <CUR> currency contributes the most to total profit?",
        "Which <FACTOR> has the highest frequency of losses?"
    ]
    statistical_queries = [{"query": query} for query in statistical_queries]

    # Save the result to state var and set the cache flag to True
    state['results'][state['state']] = statistical_queries
    state['cache_flag'][state['state']] = True

    # Assign id to each query with sha hash
    for query in state['results'][state['state']]:
        query['id'] = hashlib.sha256(json.dumps(query).encode()).hexdigest()
        query['flag'] = 'hot'
        
    # Save the result to cache with the state name and {result} key
    save_json_data(state['cache_location'][state['state']], {"result":state['results'][state['state']]})

    LOGGER.info(f"State: {state['state']} | Query generation completed, saved the result to cache and set the cache flag to True")

    return state








############################################################################################
# Agent Function 6
# Name: register_data
# LLM-Call: No
# Description: It is used to register the data to in-memory DB. It reads the rule-based title comment json data and saves it to in-memory DB
############################################################################################

def register_data(state: AgentState):
    state['state'] = 'register_data'
    LOGGER.info(f"State: {state['state']} | Initializing the agent to register data")

    # It is a very naive implementation, we can directly write the code here.
    rule_based_title_comment_data  = load_json_data(state['cache_location']['rule_based_title_comment_data'])
    
    global TITLE_DATA_INMEM_DB
    TITLE_DATA_INMEM_DB = get_title_data_inmemory_db(rule_based_title_comment_data)

    LOGGER.info(f"State: {state['state']} | Data registration completed, saved the data to in-memory DB. Global variable TITLE_DATA_INMEM_DB is set")

    return state





############################################################################################
# Agent Function 7
# Name: generate_sql_script_agent
# LLM-Call: Yes
# Description: It generates the SQL script to query the data from the table. It takes the generated subjective and statistical queries and generates the SQL script to query the data from the table.
# The Table is `title_data` and the schema is provided in the prompt.
# The SQL script is generated using the `SQLScript` tool.
############################################################################################

def generate_sql_script_agent(state: AgentState):
    state['state'] = 'sql_script_generation'
    LOGGER.info(f"State: {state['state']} | Initializing the agent to generate SQL script")

    # Load the data from cache if the cache flag is set to True
    cached_result = load_cached_results(state)

    # We avoid total cache here as the data. We selectively load the cache for the title_data table
    # if cached_result:
    #     return state

    # Initialize the model
    sql_script_engine =  GeminiJsonEngine(
                                    model_name=state['model']['model_name'],
                                    basemodel=SQLScript,
                                    temperature=state['model']['temperature'],
                                    max_output_tokens=state['model']['max_output_tokens'],
                                    systemInstructions="You are an expert financial bot. You will be given a table and you need to generate a SQL script to query the data from the table. ",
                                    max_retries=state['model']['max_retries'],
                                    wait_time=state['model']['wait_time'])

    # Previous state outputs
    subj_queries = state['results']['subj_query_generation']
    stat_queries = state['results']['stat_query_generation']
    all_queries = subj_queries + stat_queries

    global TITLE_DATA_INMEM_DB
    # Head of the table
    _rule_based_title_comment_data_cols,_rule_based_title_comment_data_head = TITLE_DATA_INMEM_DB.query_data("SELECT * FROM title_data LIMIT 5")
    head = pd.DataFrame(_rule_based_title_comment_data_head, columns=_rule_based_title_comment_data_cols).drop(columns=['id','COMMENT']).head()

    sql_scripts = []
    for i, query in enumerate(all_queries):
        sql_script = None 
        if query['flag']=='cold' and cached_result['results'][state['state']]:
            for k in cached_result['results'][state['state']] :
                if k['id']==query['id']:
                    sql_script = k
                    sql_scripts.append(sql_script)
                    LOGGER.info(f"State: {state['state']} | {i}/{len(all_queries)} SQL script loaded from cache for the query: {query['query'][:20]} ... to {sql_script['sql_script'][:20]} ...")
                    break
    
        if sql_script is None:
            user_sql_prompt = [
                f"You are a SQL expert. Your task is to write a SQL script to query data from the given table. Note: you are generating a SQL script for SQLLite's python library. You must be careful while writing complex queries as it is very sensitive.",
                f"Library specific notes: STDDEV is not supported in SQLLite. You can use AVG and SUM to calculate the standard deviation.",
                f"Here is the schema of the table `title_data`: {TITLE_DATA_INMEM_DB.metadata.tables}",
                f"Here is the are the first few rows of the table `title_data`: {head}",
                f"User is trying to answer the following query: {query['query']}",
                f"Write a SQL script to answer the query using the tool `SQLScript`. Your answer must follow the argument strucure of the tool `SQLScript`. You are encouraged to use compound and complex SQL queries to answer the query."
            ]
            sql_script = sql_script_engine(user_sql_prompt)[0]
            sql_scripts.append(sql_script)
            LOGGER.info(f"State: {state['state']} | {i}/{len(all_queries)} SQL script generated for the query: {query['query'][:20]} ... to {sql_script['sql_script'][:20]} ...")
            query['flag']='cold'
            LOGGER.info(f"State: {state['state']} | {i}/{len(all_queries)} Query flag changed to cold")

    
    # Save the result to state var and set the cache flag to True
    state['results'][state['state']] = sql_scripts
    state['cache_flag'][state['state']] = True

    # Assign id to each query with the same hash as the query
    for raw_query, sql_script in zip(all_queries, state['results'][state['state']]):
        sql_script['id'] = raw_query['id']
        sql_script['flag'] = raw_query['flag']

    # Save the result to cache with the state name and {result} key
    save_json_data(state['cache_location'][state['state']], {"result":state['results'][state['state']]})
    LOGGER.info(f"State: {state['state']} | SQL script generation completed, saved the result to cache and set the cache flag to True")

    # Save the previous state results with proper hot/cold flag
    save_json_data(state['cache_location']['subj_query_generation'], {"result":subj_queries})
    save_json_data(state['cache_location']['stat_query_generation'], {"result":stat_queries})
    LOGGER.info(f"State: {state['state']} | Saved the previous state subj_query_generation and stat_query_generation results to cache with proper all cold flag")

    return state










############################################################################################
# Agent Function 8
# Name: sql_result_agent
# LLM-Call: No
# Description: It generates the SQL result by executing the SQL script generated in the previous step. It queries the data from the table using the SQL script.
############################################################################################

def sql_result_agent(state:AgentState):
    state['state'] = 'sql_result'
    LOGGER.info(f"State: {state['state']} | Initializing the agent to generate SQL result")

    #### TODO: Implement Selective Caching for the SQL result. We are currently avoiding the cache for the SQL result as it is not that expensive to run the SQL queries again.
    # Load the data from cache if the cache flag is set to True
    # cached_result = load_cached_results(state)
    # if cached_result:
    #     return state

    global TITLE_DATA_INMEM_DB
    
    # Previous state outputs
    sql_scripts = state['results']['sql_script_generation']

    # Execute the SQL scripts
    sql_results = []
    pass_count = 0
    fail_count = 0
    overlength_count = 0
    for i, sql_script in enumerate(sql_scripts):
        try:
            columns, data = TITLE_DATA_INMEM_DB.query_data(sql_script['sql_script'])
            sql_results.append({
                "id": sql_script['id'],
                "columns": columns,
                "data": data,
                "status": "success",
                "description": sql_script['description'],
                "sql_script": sql_script['sql_script']
            })
            if len(data) < 20:
                LOGGER.info(f"State: {state['state']} | {i}/{len(sql_scripts)} SQL script executed, {len(data)} rows returned")
                pass_count += 1
            else:
                LOGGER.warning(f"State: {state['state']} | {i}/{len(sql_scripts)} SQL script executed, {len(data)} rows returned. Too many rows returned, consider refining the query")
                sql_results[-1]['status'] = "overlength"
                overlength_count += 1

        except Exception as e:
            LOGGER.error(f"State: {state['state']} | {i}/{len(sql_scripts)} SQL script execution failed: {str(e)}. Skipping the query")
            sql_results.append({
                "id": sql_script['id'],
                "columns": [],
                "data": [],
                "status": "failed",
                "description": sql_script['description'],
                "sql_script": sql_script['sql_script']
            })
            fail_count += 1

    LOGGER.info(f"State: {state['state']} | SQL script execution completed, {pass_count} passed, {fail_count} failed, {overlength_count} overlength")

    # Save the result to state var and set the cache flag to True
    state['results'][state['state']] = sql_results
    state['cache_flag'][state['state']] = True

    # Save the result to cache with the state name and {result} key
    save_json_data(state['cache_location'][state['state']], {"result":state['results'][state['state']]})

    LOGGER.info(f"State: {state['state']} | SQL result generation completed, saved the result to cache and set the cache flag to True")

    return state
    
    













############################################################################################
# Agent Function 9
# Name: generate_bucket_query_agent
# LLM-Call: No
# Description: It generates the bucket queries. It generates the bucket queries from the SQL results generated in the previous step. 
# it mainly groups the queries into subjective and statistical queries. Then generate a question-answer pair for each query.
############################################################################################

def generate_bucket_query_agent(state: AgentState):
    state['state'] = 'bucket_query_generation'
    LOGGER.info(f"State: {state['state']} | Initializing the agent to generate bucket queries")

    # It is not an AI task, it is a rule-based task. So, we can directly write the code here.

    # Load previous state outputs
    old_summary = state['results']['refine_old_summaries'][0]
    subj_results = state['results']['subj_query_generation']
    stat_results = state['results']['stat_query_generation']
    sql_results = [result for result in state['results']['sql_result'] if result['status'] == 'success']

    # Subj IDs and  Stat IDs
    subj_ids = [query['id'] for query in subj_results]
    stat_ids = [query['id'] for query in stat_results]

    # Filter SQL results with subj and stat IDs
    sql_subj_results = [result for result in sql_results if result['id'] in subj_ids]
    sql_stat_results = [result for result in sql_results if result['id'] in stat_ids]

    # Bucket queries
    sql_subj_result_qa_s = "\n".join([f"Q{i}. {result['description']}:\n{pd.DataFrame(result['data'], columns=result['columns']).head()}" for i, result in enumerate(sql_subj_results)])
    sql_stat_result_qa_s = "\n".join([f"Q{i}. {result['description']}:\n{pd.DataFrame(result['data'], columns=result['columns']).head()}" for i, result in enumerate(sql_stat_results)])

    
    prompts = [[
        f"You are financial expert. Your task is to provide a DETAILED financial summary over P&L Trend and other financial metrics. You are provided with some structured questions and their answers. You need to generate the summary from the insights provided in the answers.",
        f"The meaning for the columns are as follows: BUIS means Business Unit, DATE means Day of calcualtion, NET means Net Profit/Loss, Factor such as IRDelta, IRGamma, FXDelta etc., PROF_LOSS means Profit or Loss, CUR means Currency, PF means Portfolio, DSK means Desk.",
        f"Here is a sample summary for reference (Just follow the pattern, not the exact values): {old_summary}",
        f"Here are the structured questions and their answers: {qa}",
        f"Generate a detailed financial summary based on the insights provided in the answers. The statistical terms such as average, max , min etc. should be summarized in a financial point of view not a statistical point of view. "
    ] for qa in [sql_subj_result_qa_s, sql_stat_result_qa_s]]


    # Update the state
    state['results'][state['state']] = prompts

    LOGGER.info(f"State: {state['state']} | Bucket queries generated")

    return state











############################################################################################
# Agent Function 10
# Name: generate_final_result
# LLM-Call: Yes
# Description: It generates the final result. It generates the final result from the bucket queries generated in the previous step.
# It genrates two typed of financial summaries, one from the subjective queries and one from the statistical queries.
############################################################################################`

def generate_final_result(state: AgentState):
    state['state'] = 'final_result'
    LOGGER.info(f"State: {state['state']} | Initializing the agent to generate final result")

    # Load the data from cache if the cache flag is set to True
    cached_result = load_cached_results(state)
    if cached_result:
        return state

    # Initialize the model
    gemini_simple_chat_engine = GeminiSimpleChatEngine(model_name=state['model']['model_name'], 
                                                   temperature=state['model']['temperature'],
                                                   max_output_tokens=2048,
                                                   systemInstructions=None,
                                                   max_retries=state['model']['max_retries'],
                                                   wait_time=state['model']['wait_time'])

    # Previous state outputs
    bucket_queries = state['results']['bucket_query_generation']

    # Generate the final result
    final_results = []
    for i, bucket_query in enumerate(bucket_queries):
        final_result = gemini_simple_chat_engine(bucket_query)
        final_results.append(final_result)
        LOGGER.info(f"State: {state['state']} | {i}/{len(bucket_queries)} Final result generated from the bucket query. {final_result[:30]}...")

    # Save the result to state var and set the cache flag to True
    state['results'][state['state']] = final_results
    state['cache_flag'][state['state']] = True

    # Save the result to cache with the state name and {result} key
    save_json_data(state['cache_location'][state['state']], {"result":state['results'][state['state']]})


    LOGGER.info(f"State: {state['state']} | Final result generation completed, saved the result to cache and set the cache flag to True")
    return state





############################################################################################
# Manual Change State
# Name: manual_change_state
# Description: It is used to manually change the state of the agent. It is used to manually change the state of the agent and update the results in the cache.
# It is used to manually change the state of the agent and update the results in the cache.
############################################################################################
def validate_schema(change_state_name: str, new_state_result: Any) -> bool:
    if change_state_name == "refine_old_summaries":
        return isinstance(new_state_result, list) and all(isinstance(item, str) for item in new_state_result)
    
    elif change_state_name == "sql_script_generation":
        expected_keys = {"description", "sql_script", "columns", "id", "flag"}
        return (
            isinstance(new_state_result, list) and
            all(
                isinstance(item, dict) and expected_keys == set(item.keys()) and
                isinstance(item["description"], str) and
                isinstance(item["sql_script"], str) and
                isinstance(item["columns"], list) and all(isinstance(col, str) for col in item["columns"]) and
                isinstance(item["id"], str) and
                isinstance(item["flag"], str)
                for item in new_state_result
            )
        )
    
    elif change_state_name in {"subj_query_generation", "stat_query_generation"}:
        expected_keys = {"query", "id", "flag"}
        return (
            isinstance(new_state_result, list) and
            all(
                isinstance(item, dict) and expected_keys == set(item.keys()) and
                isinstance(item["query"], str) and
                isinstance(item["id"], str) and
                isinstance(item["flag"], str)
                for item in new_state_result
            )
        )
    
    return False

def manual_change_state(old_state:AgentState, change_state_name:str, new_state_result:Dict[str, Any]):
    state_name_array = [
            'refine_old_summaries',
            'subj_query_generation',
            'stat_query_generation',
            'register_data',
            'sql_script_generation',
            'sql_result',
            'bucket_query_generation',
            'final_result'
    ]
    
    schema_valid = validate_schema(change_state_name, new_state_result)
    if not schema_valid:
        LOGGER.error(f"State: {change_state_name} | Manual change in state {change_state_name} not possible due to schema mismatch")
        return old_state
    
    i = 0
    for i in range(len(state_name_array)):
      if state_name_array[i] == change_state_name:
         break
      
    if change_state_name in ['refine_old_summaries', 'sql_script_generation']:
        old_state['results'][change_state_name] = new_state_result
        old_state['cache_flag'][change_state_name] = True
        LOGGER.info(f"State: {change_state_name} | Manual change in state {change_state_name} done both in RAM and Disk cache")
        save_json_data(old_state['cache_location'][change_state_name], {"result":new_state_result})
        for j in range(i+1, len(state_name_array)):
            old_state['results'][state_name_array[j]] = []
            old_state['cache_flag'][state_name_array[j]] = False
            save_json_data(old_state['cache_location'][state_name_array[j]], {"result":old_state['results'][state_name_array[j]]})
            LOGGER.info(f"State: {state_name_array[j]} | Manual RESET state {state_name_array[j]} done both in RAM and Disk cache")

    elif change_state_name in ['subj_query_generation', 'stat_query_generation']:
        prev_result = old_state['results'][change_state_name]
        id_to_content_mapping = {
            result['id']:result for result in prev_result
        }
        
        for result in new_state_result:
            if result['id'] not in id_to_content_mapping or (result['id'] in id_to_content_mapping and result['query']!=id_to_content_mapping[result['id']]['query']):
                result['id'] = hashlib.sha256(json.dumps(result).encode()).hexdigest()
                result['flag'] = 'hot'
                LOGGER.info(f"Conflict in ID found for the query: {result['query'][:20]}... New ID generated and flag set to hot")


        old_state['results'][change_state_name] = new_state_result
        old_state['cache_flag'][change_state_name] = True
        save_json_data(old_state['cache_location'][change_state_name], {"result":new_state_result})
        LOGGER.info(f"State: {change_state_name} | Manual change in state {change_state_name} done both in RAM and Disk cache")
    
    else:
        LOGGER.warning(f"State: {change_state_name} | Manual change in state {change_state_name} not possible")

    # Store the state
    save_json_data(old_state['cache_location']['state_config'], old_state)
    LOGGER.info(f"State: {change_state_name} | State config saved to disk cache")

    return old_state




############################################################################################
# Reset All State
# Name: reset_all_state
# Description: It is used to reset all the states to the initial state. It is used to reset all the states to the initial state.
############################################################################################

def reset_all_state():
    init_state_snap = {
            'state': 'start',
            'model':{
                'model_name':'gemini-2.0-flash-001',
                'temperature': 0.5,
                'max_output_tokens': 512,
                'max_retries':5,
                'wait_time':30
            },
            'results':{
                'refine_old_summaries':[],
                'subj_query_generation':[],
                'stat_query_generation':[],
                'register_data':[],
                'sql_script_generation':[],
                'sql_result':[],
                'bucket_query_generation':[],
                'final_result':[]
            },
            'cache_location':{
                "sample_summarized_pnl_commentaries":"../sample_data/sample_summarized_pnl_commentaries.json",
                "rule_based_title_comment_data":"../sample_data/rule_based_title_comment_data.json",

                "state_config":"../sample_data/cached/state_config.json",
                
                "refine_old_summaries":"./sample_data/cached/refine_old_summaries.json",
                "subj_query_generation":"./sample_data/cached/subj_query_generation.json",
                "stat_query_generation":"./sample_data/cached/stat_query_generation.json",
                "register_data":"./sample_data/cached/register_data.json",
                "sql_script_generation":"./sample_data/cached/sql_script_generation.json",
                "sql_result":"./sample_data/cached/sql_result.json",
                "bucket_query_generation":"./sample_data/cached/bucket_query_generation.json",
                "final_result":"./sample_data/cached/final_result.json"
            },
            'cache_flag':{
                "refine_old_summaries":False,
                "subj_query_generation":False,
                "stat_query_generation":False,
                "register_data":False,
                "sql_script_generation":False,
                "sql_result":False,
                "bucket_query_generation":False,
                "final_result":False
            }
        }
    save_json_data(init_state_snap['cache_location']['state_config'], init_state_snap)
    LOGGER.info(f"State: reset_all_state | All states reset to initial state")
    return init_state_snap



############################################################################################
# Load State
# Name: load_current_state
# Description: It is used to load the state from the disk cache. It is used to load the state from the disk cache.
############################################################################################

def load_current_state(sate_config_path):
    snap = load_json_data(sate_config_path)
    if snap is None:
        snap = reset_all_state()
    LOGGER.info(f"State: load_state | State loaded from disk cache")
    return snap