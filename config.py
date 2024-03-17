
max_len = 768
sortby="Sno" # ["Sno", "Model", "Device", "Prediction", "Execution Time"]

random_sel = 2
num_parallel = 2
paths_weights=[
        ('./models/debertav3large', './weights/17_ft1/weights_ep0',   
         1, "gpu"),
        ('./models/debertav3large', './weights/17_ft103/weights_ep0', 
         0, "gpu"),
        ('./models/debertav3large', './weights/19_ft1/weights_ep0',   
         0, "cpu"), #acc
        ('./models/debertav3large', './weights/19_ft103/weights_ep0', 
         0, "cpu"), #acc
        ('./models/debertalarge',   './weights/20_ft1/weights_ep0',   
         1, "gpu"),
        ('./models/debertalarge',   './weights/20_ft103/weights_ep0', 
         0, "cpu"),
    ]

sentiment_weight = 1.0
profanity_threshold = 0.77
profanity_weight_max = 1.1
profanity_weight_min = 0.5
emoji_weight = 0.8
specialChar_weight = 0.2
url_weight = 1.0

zero_shot_thresh = 0.7
zero_shot = True # pre model/processing

zero_shot_post = True

check_profanity = True
check_sentiment = True
check_text_score = True

# result_text_thresh = [0.5, 0.27, 0.4] # AI | Human | Likely Human | Updataed to work on memory based chat
result_text_thresh = [0.7, 0.27, 0.55] # AI | Human | Likely Human

user_session_timeout = 5

