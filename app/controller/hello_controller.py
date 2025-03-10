from flask import Blueprint, jsonify, request, render_template
from app.service.hello_service import get_hello_message

hello_blueprint = Blueprint('hello', __name__)

@hello_blueprint.route('/hello', methods=['GET'])
def hello():
    return jsonify(get_hello_message()), 200

@hello_blueprint.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@hello_blueprint.route("/agent")
def predate():
    return render_template("predata.html")