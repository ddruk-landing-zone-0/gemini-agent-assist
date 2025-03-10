from ..service.agent_build_service import MY_AGENT
from flask import Blueprint, jsonify, request, render_template
from app.service.hello_service import get_hello_message
from app.service.generic_utils import load_json_data, save_json_data
import json

agent_blueprint = Blueprint('agent', __name__)

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
        "sample_summarized_pnl_commentaries":"./sample_data/sample_summarized_pnl_commentaries.json",
        "rule_based_title_comment_data":"./sample_data/rule_based_title_comment_data.json",
        
        "refine_old_summaries":"./sample_data/cached/refine_old_summaries.json",
        "subj_query_generation":"./sample_data/cached/subj_query_generation.json",
        "stat_query_generation":"./sample_data/cached/stat_query_generation.json",
        "sql_script_generation":"./sample_data/cached/sql_script_generation.json",
        "sql_result":"./sample_data/cached/sql_result.json",
        "final_result":"./sample_data/cached/final_result.json"
    },
    'cache_flag':{
        "refine_old_summaries":True,
        "subj_query_generation":True,
        "stat_query_generation":True,
        "sql_script_generation":True,
        "sql_result":True,
        "final_result":True
    }
}

@agent_blueprint.route('/agent/continue-flow', methods=['GET'])
def agent():
    snap = MY_AGENT.continue_flow(init_state_snap);
    return jsonify(snap), 200




# Get the sample_summarized_pnl_commentaries 
@agent_blueprint.route('/agent/get-pre-loaded-data', methods=['GET'])
def get_pre_loaded_data():
    path_sample_summarized_pnl_commentaries = init_state_snap['cache_location']['sample_summarized_pnl_commentaries']

    sample_summarized_pnl_commentaries = load_json_data(path_sample_summarized_pnl_commentaries)

    if sample_summarized_pnl_commentaries is None:
        return jsonify({
            'message': 'Data not found'
        }), 404
    return jsonify({
        'sample_summarized_pnl_commentaries': sample_summarized_pnl_commentaries,
    }), 200

# Set the sample_summarized_pnl_commentaries
@agent_blueprint.route('/agent/set-pre-loaded-data', methods=['POST'])
def set_pre_loaded_data():
    path_sample_summarized_pnl_commentaries = init_state_snap['cache_location']['sample_summarized_pnl_commentaries']

    data = request.json
    save_json_data(path_sample_summarized_pnl_commentaries, data)

    return jsonify({
        'message': 'Data saved successfully'
    }), 200


# Get the steps
@agent_blueprint.route('/agent/get-steps', methods=['GET'])
def get_steps():
    steps = list(init_state_snap['results'].keys())

    # Return payload 
    result = {"results":{},"cache_flag":{}}

    # read data from cache
    for step in steps:
        path = init_state_snap['cache_location'].get(step, None)
        if path is not None:
            result["results"][step] = load_json_data(path)
        else:
            result["results"][step] = {
                "result": ["Handled by the internal system"]
            }

    result["cache_flag"] = init_state_snap['cache_flag']   

    return jsonify(result), 200