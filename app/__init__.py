from flask import Flask
from app.controller.hello_controller import hello_blueprint
from app.controller.agent_controller import agent_blueprint
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/debasmitroy/Desktop/programming/gemini-agent-assist/key.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "hackathon0-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

app = Flask(__name__)
app.register_blueprint(hello_blueprint)
app.register_blueprint(agent_blueprint)