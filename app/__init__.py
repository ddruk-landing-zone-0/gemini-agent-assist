from flask import Flask
from app.controller.hello_controller import hello_blueprint
from app.controller.agent_controller import agent_blueprint

app = Flask(__name__)
app.register_blueprint(hello_blueprint)
app.register_blueprint(agent_blueprint)