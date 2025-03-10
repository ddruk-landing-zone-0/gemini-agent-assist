from ..service.agent_build_service import MY_AGENT
from ..service.agent_functions import reset_all_state, load_current_state,manual_change_state
from flask import Blueprint, jsonify, request, render_template
from app.service.hello_service import get_hello_message
from app.service.generic_utils import load_json_data, save_json_data
import json

agent_blueprint = Blueprint('agent', __name__)
sate_config_path='sample_data/cached/state_config.json'


@agent_blueprint.route('/agent/continue-flow', methods=['GET'])
def agent():
    snap = load_current_state(sate_config_path)
    snap = MY_AGENT.continue_flow(snap);
    return jsonify(snap), 200



# Get the sample_summarized_pnl_commentaries 
@agent_blueprint.route('/agent/get-pre-loaded-data', methods=['GET'])
def get_pre_loaded_data():
    snap = load_current_state(sate_config_path)
    path_sample_summarized_pnl_commentaries = snap['cache_location']['sample_summarized_pnl_commentaries']

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
    snap = load_current_state(sate_config_path)
    path_sample_summarized_pnl_commentaries = snap['cache_location']['sample_summarized_pnl_commentaries']

    data = request.json
    save_json_data(path_sample_summarized_pnl_commentaries, data)

    return jsonify({
        'message': 'Data saved successfully'
    }), 200


# Get the steps
@agent_blueprint.route('/agent/get-steps', methods=['GET'])
def get_steps():
    snap = load_current_state(sate_config_path)
    steps = list(snap['results'].keys())

    # Return payload 
    result = {"results":{},"cache_flag":{}}

    # read data from cache
    for step in steps:
        path = snap['cache_location'].get(step, None)
        if path is not None:
            result["results"][step] = load_json_data(path)
        else:
            result["results"][step] = {
                "result": ["Handled by the internal system"]
            }

    result["cache_flag"] = snap['cache_flag']   

    return jsonify(result), 200



# Set the steps manually
@agent_blueprint.route('/agent/set-steps', methods=['POST'])
def set_steps():
    snap = load_current_state(sate_config_path)
    data = request.json

    old_state = snap
    change_state_name = data.get('change_state_name', None)
    new_state_result = data.get('new_state_result', None)

    if change_state_name is not None and new_state_result is not None:
        manual_change_state(old_state, change_state_name, new_state_result)
        return jsonify({
            'message': 'State changed successfully'
        }), 200
    else:
        return jsonify({
            'message': 'Invalid request'
        }), 400
