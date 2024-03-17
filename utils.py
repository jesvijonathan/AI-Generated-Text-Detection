from flask_socketio import SocketIO

socketio = None

def set_socketio_instance(instance):
    global socketio
    socketio = instance

def print_(text="\n", user_id=None, event="message", room=None):
    global socketio

    if room:
        user_id = room
    

    if not socketio:
        user_id = "*" if user_id is None else user_id
        print(str(text))
        return

    if user_id == "*" or user_id is None:
        socketio.emit(event, text)
        print(str(text))
        return
    
    if user_id != "~":
        socketio.emit(event, text, room=user_id)
    else:
        print(str(text))

job_template = {
    "job_id": 0,  # Job ID
    "user_id": "",  # User ID
    "job_name": "",  # Job Name
    "job_status": {
        "progress": 0,
        "updates": [],  # Job Updates
        "status": "Pending",  # Pending, Processing, Completed, Failed
    },
    "job_values": {
        "text": "",  # Text
        "zero_shot": False,  # Zero Shot
        "text_check": True,  # Text Check
        "url_ratio_check": True,  # URL Ratio Check
        "emoji_ratio_check": True,  # Emoji Check
        "specialChar_ratio_check": True,  # Special Char Check
        "model_check": True,  # Model Check
        "random_seed": 0,  # Random Seed | 0 default/none
        "models": -1,  # Model | -1 default
        "parallel": -1,  # Parallel | -1 default
    },
    "job_result": {
        "clean_text": "",  # Clean Text
        "urls_score": 0,  # URL Score
        "emoji_score": 0,  # Emoji Score
        "specialChar_score": 0,  # Special Char Score
        "text_score": 0,  # Text Score
        "textblob_score": {
            "polarity": 0,  # Polarity
            "subjectivity": 0,  # Subjectivity
        },
        "vader_score": {
            "pos": 0,  # Positive
            "neg": 0,  # Negative
            "neu": 0,  # Neutral
            "compound": 0,  # Compound
        },
        "sentiment_score": 0,  # Sentiment Score
        "sentiment_label": "",  # Sentiment Label
        "profanity_check": False,  # Profanity Check
        "profanity_prob": 0,  # Profanity Probability
        "profanity_score": 0,  # Profanity Score
        "one_shot_score": 0,  # One Shot Score
        "model_score": {
            "model_avg": 0,  # Model Average
            "execution_time": 0,  # Execution Time
            "models": [
                {
                    "model": "",  # Model
                    "details": "",  # Details
                    "execution_type": "",  # Execution Type
                    "score": 0,  # Score
                    "time": 0,  # Time
                    "device": "",  # Device
                }
            ],
        },
        "score": 0,  # Score
        "device": "",  # Devices
        "parallel": 0,  # max_parallel
        "max_length": 0,  # Max Length
        "max_memory": 0,  # Max Memory
        "execution_time": 0,  # Time
    },  # Job Result
}  # job template



def update_job_status(user_id, update, job_id=None, datam=None):
    """Update Job Status for a user
    Args:
        user_id (str): User ID
        update (dict): Job Update
        job_id (int, optional): Job ID. Defaults to None.
        datam (dict, optional): Data. Defaults to None."""
    print("update_job_status", user_id, update, job_id, datam)
    global socketio
    if not socketio:
        return
    
    if job_id==None:
        if not datam or "job_id" not in datam:
            return
        job_id = datam["job_id"]
     
    job_status = {}
    
    for key in update:
        # print("key", key)
        if key in job_template["job_status"]:
            # print("key in job_template[job_status]", key)
            job_status[key] = update[key]
    
    print("job_status", job_status)
    socketio.emit("job_status", {"job_id":job_id, "job_status":job_status}, room=user_id)


def update_job_result(user_id, update, job_id=None, datam=None):
    """Update Job Result for a user
    Args:
        user_id (str): User ID
        update (dict): Job Update
        job_id (int, optional): Job ID. Defaults to None.
        datam (dict, optional): Data. Defaults to None."""
    print("update_job_result", user_id, update, job_id, datam)
    global socketio
    if not socketio:
        return
    
    if job_id==None:
        if not datam or "job_id" not in datam:
            return
        job_id = datam["job_id"]
     
    job_result = {}

    for key in update:
        # print("key", key)
        if key in job_template["job_result"]:
            # print("key in job_template[job_result]", key)
            job_result[key] = update[key]

    print("job_result", job_result)
    socketio.emit("job_result", {"job_id":job_id, "job_result":job_result}, room=user_id)

def update_job_values(user_id, update, job_id=None, datam=None):
    """Update Job Values for a user
    Args:
        user_id (str): User ID
        update (dict): Job Update
        job_id (int, optional): Job ID. Defaults to None.
        datam (dict, optional): Data. Defaults to None."""
    print("update_job_values", user_id, update, job_id, datam)
    global socketio
    if not socketio:
        return
    
    if job_id == None:
        if not datam or "job_id" not in datam:
            return
        job_id = datam["job_id"]
    job_values = {}
    for key in update:
        # print("key", key)
        if key in job_template["job_values"]:
            # print("key in job_template[job_values]", key)
            job_values[key] = update[key] 
   
    print("job_values", job_values)
    socketio.emit("job_values", {"job_id":job_id, "job_values":job_values}, room=user_id)