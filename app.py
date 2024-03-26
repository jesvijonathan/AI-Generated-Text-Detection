from flask import Flask, render_template, session, request
from flask_socketio import SocketIO
from flask_cors import CORS
import threading
from queue import Queue
import text_scorer as tex_score
import ml_eval
from utils import set_socketio_instance, print_, update_job_status, update_job_result, update_job_values
from datetime import datetime, timedelta
from config import *
import traceback

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
            "text": "None"
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
            "text": eval_tex["text_input"]
        }
    else:
        print_("\nZero shot: Negative (AI Generated)\n", room=user_id)
        if zero_shot_post:
            update_job_status(user_id=user_id, update={"progress": 0.3, "updates": ["Zero_Shot_Check (Negative)"]}, job_id=data["job_id"])
        else:
            update_job_status(user_id=user_id, update={"progress": 0.3, "updates": ["Skipping Zero_Shot_Check"]}, job_id=data["job_id"])
        
        result = ml_eval.evaluator.model_score(eval_tex["text_input"], user_id=user_id,data=data)
        update_job_status(user_id=user_id, update={"progress": 0.9, "updates": ["Wrapping up..."]}, job_id=data["job_id"])

        return {
            "score": float(result),
            "label": "Negative" if result >= 0.7 else "Positive" if result <= 0.55 else "Neutral",
            "text": eval_tex["text_input"]
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

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)


# Run the server using the command: python app.py
# Open the browser and go to the URL: http://localhost:5000/ or http://localhost:3000/

# WebSocket Endpoints:
# Connect: ws://localhost:5000/socket.io/
# Disconnect: ws://localhost:5000/socket.io/
# Values: ws://localhost:5000/socket.io/

# HTTP Endpoint:
# Index: http://localhost:5000/