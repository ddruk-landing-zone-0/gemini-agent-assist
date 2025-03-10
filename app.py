from app.service.agent_build_service import MY_AGENT
from app.service.agent_functions import reset_all_state, load_current_state
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/debasmitroy/Desktop/programming/gemini-agent-assist/key.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "hackathon0-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"



snap = load_current_state(sate_config_path='sample_data/cached/state_config.json')
MY_AGENT.continue_flow(snap);

snap = MY_AGENT.get_recent_state_snap()
print(snap)