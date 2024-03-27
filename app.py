from flask import Flask, render_template, session, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import threading
from queue import Queue
import text_scorer as tex_score
import ml_eval
from utils import set_socketio_instance, print_, update_job_status, update_job_result, update_job_values
from datetime import datetime, timedelta
import time
from config import *
import traceback
import hashlib
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app, cors_allowed_origins="*")
set_socketio_instance(socketio)

processing_queue = Queue()
busy_users = set()

user_activity_timestamps = {}

def allocate_ml_processing(user_id, data):
    global processing_queue, busy_users
    
    busy_users.add(user_id)
    processing_queue.put((user_id, data))
    print_("Added to processing queue...", user_id)
    print_("Processing queue size increased: " + str(processing_queue.qsize()))

def send_server_status(status=None, user_id=None):
    data = {
        'no_jobs': len(processing_queue.queue),
        'no_connections': len(user)
    }
    if status:
        data['status'] = status

    if user_id:
        socketio.emit('server_status', data, room=user_id)
    else:
        socketio.emit('server_status', data)


def handle_ml_processing(data, user_id=None):
    text = data["job_values"]["text"]
    

    update_job_status(user_id=user_id, update={"progress": 0.01, "updates": ["Inspecting text..."]}, job_id=data["job_id"])
    

    if text == "" or text is None or len(text) == 0 or text.isspace():
        print_("\nZero shot: Negative (AI Generated)\n", room=user_id)
        return None  
    

    eval_tex = tex_score.sentiment_analyzer.analyze_text(data)
    tex_score.print_eval_text(eval_tex, user_id)

    print("##################", eval_tex, "##################")


    if eval_tex["res"] == None:
        print_("\nZero shot: Positive (Human Generated)\n", room=user_id)
        update_job_status(user_id, update={"progress": 1, "updates": ["Completed", "Zero_Shot_Check (Positive)"], "status": "Completed"}, job_id=data["job_id"])
        
        job_res = {
            "score": 0.0,
        }
        update_job_result(user_id, update=job_res, job_id=data["job_id"])
        return {
            "score": (0.0),
            "label": "Negative",
            "text_input": "None",
            **eval_tex,
        }
    elif (eval_tex["res"] >= zero_shot_thresh and zero_shot):
        print_("\nZero shot: Positive (Human Generated)\n", room=user_id)
        update_job_status(user_id, update={"progress": 1, "updates": ["Completed", "Zero_Shot_Check (Positive)"], "status": "Completed"}, job_id=data["job_id"])
        
        job_res = {
            "score": (1.0 - float(eval_tex["res"]))
        }
        update_job_result(user_id, update=job_res, job_id=data["job_id"])
        
        return {
            "score": (1.0 - float(eval_tex["res"])),
            "label": "Negative",
            **eval_tex,
        }
    else:
        print_("\nZero shot: Negative (AI Generated)\n", room=user_id)
        if zero_shot_post:
            update_job_status(user_id=user_id, update={"progress": 0.3, "updates": ["Zero_Shot_Check (Negative)"]}, job_id=data["job_id"])
        else:
            update_job_status(user_id=user_id, update={"progress": 0.3, "updates": ["Skipping Zero_Shot_Check"]}, job_id=data["job_id"])
        
        score, result = ml_eval.evaluator.model_score(eval_tex["text_input"], user_id=user_id,data=data)
        update_job_status(user_id=user_id, update={"progress": 0.9, "updates": ["Wrapping up..."]}, job_id=data["job_id"])

        return {
            "score": float(score),
            "label": "Negative" if score >= 0.7 else "Positive" if score <= 0.55 else "Neutral",
            **eval_tex,
            **result 
            }


"""
// data = {
//   job_id: 0, // Job ID
//   user_id: server_status.value.user_id, // User ID
//   job_name: "", // Job Name
//   job_status: {
//     progress: 0,
//     updates: [], // Job Updates
//     status: "Pending", // Pending, Processing, Completed, Failed
//   },
//   job_values: {
//     text: "", // Text
//     zero_shot: false, // Zero Shot
//     text_check: true, // Text Check
//     url_ratio_check: true, // URL Ratio Check
//     emoji_ratio_check: true, // Emoji Check
//     specialChar_ratio_check: true, // Special Char Check

//     model_check: true, // Model Check
//     random_seed: 0, // Random Seed | 0 default/none

//     models: -1, // Model | -1 default
//     parallel: -1, // Parallel | -1 default
//   },

//   job_result: {
//     clean_text: "", // Clean Text
//     urls_score: 0, // URL Score
//     emoji_score: 0, // Emoji Score
//     specialChar_score: 0, // Special Char Score
//     text_score: 0, // Text Score

//     textblob_score: {
//       polarity: 0, // Polarity
//       subjectivity: 0, // Subjectivity
//     },

//     vader_score: {
//       pos: 0, // Positive
//       neg: 0, // Negative
//       neu: 0, // Neutral
//       compound: 0, // Compound
//     },
//     sentiment_score: 0, // Sentiment Score
//     sentiment_label: "", // Sentiment Label

//     profanity_check: false, // Profanity Check
//     profanity_prob: 0, // Profanity Probability
//     profanity_score: 0, // Profanity Score

//     one_shot_score: 0, // One Shot Score

//     model_score: {
//       model_avg: 0, // Model Average
//       execution_time: 0, // Execution Time
//       models: [
//         {
//           model: "", // Model
//           execution_type: "", // Execution Type
//           score: 0, // Score
//           time: 0, // Time
//           device: "", // Device
//         },
//       ],
//     },

//     score: 0, // Score

//     device: "", // Devices
//     parallel: 0, // max_parallel
//     max_length: 0, // Max Length
//     max_memory: 0, // Max Memory
//     execution_time: 0, // Time
//   }, // Job Result
// }; // job template
"""
def polling_thread():
    global processing_queue, busy_users, user_activity_timestamps

    while True:
        if not processing_queue.empty():
            try:
                user_id, data = processing_queue.get()
            except Exception as e:
                print_("Error occurred while getting data from processing queue: " + str(e))
    

            try:
                print_("Processing...", user_id)
                print("Data: ", data)

                jos_status = {
                    "progress": 0.02,
                    "updates": ["Processing..."], 
                    "status": "Processing",
                }

                update_job_status(user_id, jos_status, job_id=data["job_id"])
                
                send_server_status(status='Busy')

                result = handle_ml_processing(data, user_id)
                
                update_job_status(user_id, update={"progress": 0.95, "updates": ["Processed successfully"]}, job_id=data["job_id"])

                if result["score"] >= result_text_thresh[0]:
                    update_message = "The text is AI generated, with an 'AI-Written' Probability Score : ({:.2})".format(result["score"])
                elif result["score"] <= result_text_thresh[1]:
                    update_message = "The text is human written, with 'AI-Written' Probability Score : ({:.2})".format(result["score"])
                elif result["score"] <= result_text_thresh[2]:
                    update_message = "The text is most likely human written, with 'AI-Written' Probability Score : ({:.2})".format(result["score"])
                else:
                    update_message = "The text is likely human written, with 'AI-Written' Probability Score : ({:.2})".format(result["score"])

                update_job_status(user_id, update={"progress": 1, "updates": ["Completed", update_message], "status": "Completed"}, job_id=data["job_id"])

                print_("Result: " + str(result), user_id, "result")
                if not any([True for _, d in list(processing_queue.queue) if d["user_id"] == user_id]):
                    busy_users.remove(user_id)
                else:
                    print_("User still has pending jobs...", user_id)

                send_server_status('Free')

                print_("Processing queue size decreased: " + str(processing_queue.qsize()))
                print_("Done", user_id)
            except Exception as e:
                print_("Error occurred while processing data: " + str(e)
                       + "\n" + traceback.format_exc())
                update_job_status(user_id, update={"progress": -1, "updates": ["Failed", "An error occurred while processing the data. Please try again."], "status": "Failed"}, job_id=data["job_id"])
                try:
                    busy_users.remove(user_id)
                    send_server_status('Free')
                except Exception as e:
                    print("Error occurred while removing user from busy users: " + str(e))
        else:
            socketio.sleep(1)

        current_time = datetime.now()
        for user_id, last_activity_time in list(user_activity_timestamps.items()):
            if (current_time - last_activity_time).total_seconds() > 60*user_session_timeout:
                print_("User {} has been inactive for more than {} minute. Disconnecting...".format(user_id,user_session_timeout))
                socketio.server.disconnect(user_id)
                user_activity_timestamps.pop(user_id, None)
                try:
                    found_user = next((user_info for user_info in user if user_info[0] == user_id), None)
                    if found_user:
                        user.remove(found_user)
                except Exception as e:
                    print("Error occurred while removing user from user list: " + str(e))    
                    

polling_thread = threading.Thread(target=polling_thread)
polling_thread.start()

user = []

@socketio.on('connect')
def handle_connect():
    user_id = request.sid
    try:
        found_user = next((user_info for user_info in user if user_info[0] == user_id), None)
        if found_user:
            user.remove(found_user)
    except Exception as e:
        print("Error occurred while removing user from user list: " + str(e))
    
    session['user_id'] = user_id
    user_activity_timestamps[user_id] = datetime.now()
    print_('Connected', user_id)
    user.append([user_id, 0])
    ln = len(user)
    print_({
        'user_id': user[ln-1][0],
        'user_no': ln,
        'jobs': user[ln-1][1],
    }, user_id=user_id, event="user_info")
    print_("Total connections : " + str(ln))
    
    send_server_status()
    send_server_status('Connected',user_id=user_id)
    

@socketio.on('disconnect')
def handle_disconnect():
    user_id = session.get('user_id')
    try:
        found_user = next((user_info for user_info in user if user_info[0] == user_id), None)
        if found_user:
            user.remove(found_user)
    except Exception as e:
        print("Error occurred while removing user from user list: " + str(e))
        
    print_('Disconnected', user_id)

    ln = len(user)
    print_("Total connections : " + str(len(user)))
    send_server_status()
    try:
        ln = len(user)
        print_({
            'user_id': user_id,
            'user_no': ln,
            'jobs': 0,
        }, user_id=user_id, event="user_info")
        send_server_status(status='Disconnected', user_id=user_id)
    except Exception as e:
        print("Can send server status as connection terminated : " + str(e))
    


@socketio.on('values')
def handle_message(values):
    one_user_one_job = False
    print("################")
    print(values)
    print("################")
    data = values
    
    user_id = session.get('user_id')

    if user_id:
        if user_id in busy_users and one_user_one_job:
            print_('Server busy. Please wait...', room=user_id)
            print_("server busy...")
        else:
            allocate_ml_processing(user_id, data)
        user_activity_timestamps[user_id] = datetime.now()
    else:
        print_('Invalid user', request.sid)

@app.route('/')
def index():
    return render_template('index_ml.html')

req_users = []

def generate_user_id():
    if 'user_id' in session:
        return session['user_id']
    else:
        user = request.remote_addr + str(datetime.now())
        user_id = hashlib.md5(user.encode()).hexdigest()[:8]
        session['user_id'] = user_id
        return user_id

def error_response(message="Error", status_code=400):
    return {
            "message": message,
        }, status_code

def generate_job_data(job_num, user_id, text):
    job_data = {
        "job_id": job_num,
        "user_id": user_id,
        "job_name": "",
        "job_status": {
            "progress": 0,
            "updates": [],
            "status": "Pending",
        },
        "job_values": {
            "text": text,
            "zero_shot": False,
            "text_check": True,
            "url_ratio_check": True,
            "emoji_ratio_check": True,
            "specialChar_ratio_check": True,
            "model_check": True,
            "random_seed": 0,
            "models": -1,
            "parallel": -1,
        },
        "job_result": {
            "clean_text": "",
            "urls_score": 0,
            "emoji_score": 0,
            "specialChar_score": 0,
            "text_score": 0,
            "textblob_score": {
                "polarity": 0,
                "subjectivity": 0,
            },
        },
    }
    return job_data


@app.route('/api/job', methods=['GET', 'POST'])
def process_jobs():
    if request.method == 'GET':
        data = request.args.to_dict()
    elif request.method == 'POST':
        if request.headers.get('Content-Type') != 'application/json':
            return error_response("Invalid Content-Type", 400)
        data = request.json

    user_id = data.get("user_id")
    if user_id not in req_users: 
        return error_response("Invalid User", 400)

    text = data.get("text")
    if not text:
        return error_response("Invalid Text", 400)

    job_data = generate_job_data(0, user_id, text)

    user_activity_timestamps[user_id] = datetime.now()
    result = handle_ml_processing(job_data, user_id)
    if result:
        return jsonify(result), 200
    else:
        return error_response("Failed to process the request", 500)

@app.route('/api/token', methods=['GET'])
def generate_token():
    user_id = generate_user_id()
    req_users.append(user_id)
    print("Total users:", len(req_users), "User ID:", user_id)
    return {"user_id": user_id}, 200

@app.route('/api/remove', methods=['GET'])
def remove_user():
    user_id = request.args.get('user_id')
    if user_id in req_users:
        req_users.remove(user_id)
        return {"message": "User removed successfully"}, 200
    else:
        return error_response("User not found", 404)
    
@app.route('/api', methods=['GET', 'POST'])
def api():
    return {
            "message": "AI GENERATED TEXT DETECTION API",
            "title": "AI Generated Text Detection API",
            "description": "API to detect AI generated text using various NLP techniques and models.",
            "source": "https://github.com/jesvijonathan/AI-Generated-Text-Detection",
            "version": "0.1.2",
            "author": "Jesvi Jonathan",
            "endpoints": {
                "/api/job": {
                    "method": "GET/POST",
                    "description": "Process the job for the user",
                    "data": {
                        "user_id": "User ID",
                        "text": "Text data to process"
                    },
                    "response": "JSON Object with the result of the job"
                },
                "/api/token": {
                    "method": "GET",
                    "description": "Generate a new user token",
                    "response": "JSON Object with the user_id token"
                }
            },

        }, 200


def run_local():
    job_num = 0

    while True:
        input("Press enter to process the text from input.txt")
        try:
            with open('data/input.txt', 'r') as file:
                text = file.read().strip()
                if not text:
                    print("Invalid Text")
                    continue

            user_id = job_num
            job_data = generate_job_data(job_num, user_id, text)

            user_activity_timestamps[user_id] = datetime.now()
            result = handle_ml_processing(job_data, user_id)
            if result:
                print(result)
                # Perform actions with the result here
            else:
                print("Failed to process the request | None")

            job_num += 1
        except Exception as e:
            print("An error occurred:", str(e))
            traceback.print_exc()


if __name__ == '__main__':
    if local_mode:
        run_local()
    else:
        socketio.run(app, debug=debug_mode, use_reloader=use_reloader, host=host, port=port, log_output=log_output)

# Run the server using the command: python app.py
# Open the browser and go to the URL: http://localhost:5000/

# WebSocket Endpoints:
# Connect: ws://localhost:5000/socket.io/
# Disconnect: ws://localhost:5000/socket.io/
# Values: ws://localhost:5000/socket.io/

# HTTP Endpoint:
# Index: http://localhost:5000/
# Token: http://localhost:5000/api/token
# Jobs: http://localhost:5000/api/job
    # GET: http://localhost:5000/api/job?user_id=ef933904&text=hello%20world%20jesvi
    # POST: http://localhost:5000/api/job {"user_id": "ef933904", "text": "hello world jesvi"}
# Remove User: http://localhost:5000/api/remove
    # GET: http://localhost:5000/api/remove?user_id=ef933904