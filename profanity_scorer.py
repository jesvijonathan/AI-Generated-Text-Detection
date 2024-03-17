import re
import demoji
import logging
from enum import Enum
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from profanity_check import predict, predict_prob
import numpy as np
from utils import print_, update_job_status, update_job_result, update_job_values
from config import *

class SentimentLabel(Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"

class SentimentAnalyzer:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.sentiment_weight = sentiment_weight
        self.profanity_threshold = profanity_threshold
        self.profanity_weight_max = profanity_weight_max
        self.profanity_weight_min = profanity_weight_min
        self.emoji_weight = emoji_weight
        self.specialChar_weight = specialChar_weight
        self.url_weight = url_weight

    def analyze_text(self, data):
        text = data["job_values"]["text"]
        print(data)
        
        update_job_status(user_id=data["user_id"], update={"progress": 0.06, "updates": ["Cleaning text..."]}, job_id=data["job_id"])
        clean_input = self._clean_input(text.lower())
        
        if not clean_input:
            return self._default_result(text)
        jos_res = {
            "clean_text": clean_input["text"],
            "urls_score": clean_input["url_ratio"],
            "emoji_score": clean_input["emoji_ratio"],
            "specialChar_score": clean_input["specialChar_ratio"],
        }
        update_job_result(user_id=data["user_id"], update=jos_res, job_id=data["job_id"])
        
        update_job_status(user_id=data["user_id"], update={"progress": 0.07, "updates": ["Analyzing text..."]}, job_id=data["job_id"])
        sentences = self._sentence_splitter(text)
        # "sentences": sentences

        update_job_status(user_id=data["user_id"], update={"progress": 0.08, "updates": ["Checking for profanity..."]}, job_id=data["job_id"])
        profanity_result = self._check_profanity(sentences)

        jos_res = {
            "profanity_check": profanity_result["is_profane"],
            "profanity_prob": profanity_result["profanity_prob"],
            "profanity_score": profanity_result["profanity_score"]
        }
            # "avg_prof_prob": np.mean(profanity_prob)
        update_job_result(user_id=data["user_id"], update=jos_res, job_id=data["job_id"])

        update_job_status(user_id=data["user_id"], update={"progress": 0.09, "updates": ["Calculating text scores..."]}, job_id=data["job_id"])
        text_scores = self._calculate_text_scores(clean_input)

        jos_res = {
            "text_score": text_scores
        }
        update_job_result(user_id=data["user_id"], update=jos_res, job_id=data["job_id"])
        
        update_job_status(user_id=data["user_id"], update={"progress": 0.1, "updates": ["Analyzing sentiment..."]}, job_id=data["job_id"])
        sentiment_scores = self._get_sentiment_scores(clean_input["text"])
        
        jos_res = {
            "textblob_score": {
                "polarity": sentiment_scores[0],
                "subjectivity": sentiment_scores[1]
            },
            "vader_score": {
                "compound": sentiment_scores[3],
                "neu": sentiment_scores[4],
                "pos": sentiment_scores[5],
                "neg": sentiment_scores[6]
            },
            "sentiment_score": sentiment_scores[0],
        }
        update_job_result(user_id=data["user_id"], update=jos_res, job_id=data["job_id"])

        update_job_status(user_id=data["user_id"], update={"progress": 0.18, "updates": ["Calculating final score..."]}, job_id=data["job_id"])
        final_score = self._calculate_final_score(clean_input, profanity_result, text_scores, sentiment_scores)
        res = final_score[1]

        jos_res = {
            "final_score": final_score[0],
            "sentiment_label": self._get_sentiment_label(final_score),
            "one_shot_score": res
        }

        update_job_result(user_id=data["user_id"], update=jos_res, job_id=data["job_id"])

        update_job_status(user_id=data["user_id"], update={"progress": 0.27, "updates": ["Initial Analysis Completed"]}, job_id=data["job_id"])
        return self._generate_output(clean_input, profanity_result, text_scores, sentiment_scores, final_score, res)


    def _clean_input(self, text):
        emojis = demoji.findall(text)
        text = demoji.replace_with_desc(text)
        # print(emojis)

        we_ratio = len(emojis) / len(text.split())
        text = demoji.replace(text, repl="")
        if not text:
            return None

        urls = re.findall(r'(http://|https://|www\.)\S+', text)
        wu_ratio = len(urls) / len(text.split())
        # print(urls)

        if wu_ratio > 0.0:
            text = re.sub(r'(http://|https://|www\.)\S+', '', text)

        if not text:
            return None

        ws_ratio = None
        if not emojis or len(emojis) == 0:
            special_chars = re.findall(r'[^a-zA-Z0-9\s]', text)
            if len(text.split()) > 0 and len(special_chars) <= len(text.split()):
                ws_ratio = len(special_chars) / len(text.split())
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            # print(special_chars)
            
        if not text:
            return None
    

        return {
            "text": text,
            "emojis": emojis,
            "emoji_ratio": we_ratio,
            "specialChar_ratio": ws_ratio,
            "url_ratio": wu_ratio,
            "urls": urls
        }

    def _sentence_splitter(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for i in range(len(sentences)):
            if len(sentences[i].split()) > 25:
                sentences[i] = re.findall(r'\b[\w\W]{1,25}\b', sentences[i])
            else:
                sentences[i] = [sentences[i]]
            # print(sentences[i])
        sentences = [sentence for sublist in sentences for sentence in sublist]
        return sentences

    def _check_profanity(self, sentences):
        for sentence in sentences:
            is_profane = predict([sentence])[0]
            profanity_prob = predict_prob([sentence])[0]
            # print(is_profane, profanity_prob, sentence)

            if is_profane and profanity_prob > self.profanity_threshold:
                return {
                    "is_profane": True,
                    "profanity_prob": np.mean(profanity_prob),            
                    "profanity_score": 1 if is_profane and profanity_prob > self.profanity_threshold else profanity_prob * self.profanity_weight_max,
                    "avg_prof_prob": np.mean(profanity_prob)
                }

        return {
            "is_profane": False,
            "profanity_prob": np.max(profanity_prob),
            "profanity_score": 1 if is_profane and profanity_prob > self.profanity_threshold else profanity_prob * self.profanity_weight_max,
            "avg_prof_prob": np.mean(profanity_prob)
        }

    def _get_sentiment_scores(self, text):
        blob = TextBlob(text)
        analyzer = SentimentIntensityAnalyzer()

        textblob_polarity, textblob_subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
        #print(textblob_polarity, textblob_subjectivity)
        vader_compound, vader_score = analyzer.polarity_scores(text)["compound"], analyzer.polarity_scores(text)["compound"]
        #print(vader_score, vader_compound)
        neutral, positive, negative = analyzer.polarity_scores(text)["neu"], analyzer.polarity_scores(text)["pos"], analyzer.polarity_scores(text)["neg"]
        return textblob_polarity, textblob_subjectivity, vader_score, vader_compound, neutral, positive, negative

    def _calculate_text_scores(self, clean_input):
        return sum(
            clean_input[key] * weight
            for key, weight in zip(["emoji_ratio", "specialChar_ratio", "url_ratio"],
                                   [self.emoji_weight, self.specialChar_weight, self.url_weight])
            if clean_input[key] is not None
        ) / 3

    def _calculate_final_score(self, clean_input, profanity_result, text_scores, sentiment_scores):

        res = 0.0

        if len(clean_input["text"].split()) <= 1:
            res = 1.0

        if profanity_result:
            if profanity_result["is_profane"] and profanity_result["profanity_prob"] >self.profanity_threshold and check_profanity:
                res = 1.0
            else:
                res += profanity_result["profanity_prob"] * self.profanity_weight_max

        if sentiment_scores[0] < 0 and sentiment_scores[0] > -1 and check_sentiment:
            res = 1.0
        else:
            res += sentiment_scores[0]

        if (clean_input["url_ratio"] > 0.8 or clean_input["emoji_ratio"] > 0.8) and check_text_score:
            res = 1.0
        else:
            res += (clean_input["url_ratio"] * self.url_weight + clean_input["emoji_ratio"] * self.emoji_weight) / 2

        # combined_score = sentiment_scores[0] + text_scores 
        weighed_prof = (profanity_result["profanity_score"] * self.profanity_weight_min)
        if weighed_prof < 1:
            weighed_prof = 0

        combined_score = sentiment_scores[0] + text_scores - weighed_prof
        
        # print(sentiment_scores[0], text_scores, -weighed_prof, combined_score)
        
        
        final_score = self._map_to_non_linear_range(combined_score)

        return final_score, res  


    def _map_to_non_linear_range(self, value):
        return ((value + 1) / 2 + 0.5) * 0.5

    def _generate_output(self, clean_input, profanity_result, text_scores, sentiment_scores, final_score, res=None):
        return {
        "text_input": clean_input["text"],
        "sentiment": {
            "textblob_polarity": sentiment_scores[0],
            "textblob_subjectivity": sentiment_scores[1],
            "vader_score": sentiment_scores[2],
            "vader_compound": sentiment_scores[3],
            "sentiment_score": sentiment_scores[0] * 1.0
        },
        "profanity": {
            "is_profane": bool(profanity_result["is_profane"]),
            "profanity_prob": profanity_result["profanity_prob"],
            "profanity_score": profanity_result["profanity_score"]
        },
        "final_score": final_score[0],
        "sentiment_label": self._get_sentiment_label(final_score),
        "text": {
            "emoji_ratio": clean_input["emoji_ratio"],
            "specialChar_ratio": clean_input["specialChar_ratio"],
            "url_ratio": clean_input["url_ratio"],
            "text_score": text_scores
        },
        "res": final_score[1] 
    }

    def _get_sentiment_label(self, final_score):
        if 0.56 < final_score[0] <= 1 or -0.56 < final_score[0] <= -1:
            return SentimentLabel.POSITIVE.value
        elif -0.44 < final_score[0] < 0.44:
            return SentimentLabel.NEGATIVE.value
        else:
            return SentimentLabel.NEUTRAL.value


    def _default_result(self, text):
        return {
            "text_input": text,
            "sentiment": {"textblob_polarity": None, "textblob_subjectivity": None, "vader_score": None,
                          "vader_compound": None, "sentiment_score": None},
            "profanity": {"is_profane": None, "profanity_prob": None, "profanity_score": None},
            "final_score": 1.0,
            "sentiment_label": SentimentLabel.NEUTRAL.value,
            "text": {"emoji_ratio": None, "specialChar_ratio": None, "url_ratio": None, "text_score": 1.0},
            "res": None
        }

# if __name__ == "__main__":
sentiment_analyzer = SentimentAnalyzer()

def print_eval_text(evaluation,user_id):
    print_(user_id=user_id)
    if evaluation:
        print_("Text: " + str(evaluation["text_input"]), user_id=user_id)
        print_(user_id=user_id)
        print_("URL Check:              " + str(evaluation["text"]["url_ratio"]), user_id=user_id)
        print_("Emoji Check:            " + str(evaluation["text"]["emoji_ratio"]), user_id=user_id)
        print_("Special Char Check:     " + str(evaluation["text"]["specialChar_ratio"]), user_id=user_id)
        print_("Text Score:             " + str(evaluation["text"]["text_score"]), user_id=user_id)
        print_(user_id=user_id)
        print_("TextBlob Polarity:      " + str(evaluation["sentiment"]["textblob_polarity"]), user_id=user_id)
        print_("TextBlob Subjectivity:  " + str(evaluation["sentiment"]["textblob_subjectivity"]), user_id=user_id)
        print_("VADER Score:            " + str(evaluation["sentiment"]["vader_score"]), user_id=user_id)
        print_("VADER Compound:         " + str(evaluation["sentiment"]["vader_compound"]), user_id=user_id)
        print_("Sentiment Score:        " + str(evaluation["sentiment"]["sentiment_score"]), user_id=user_id)
        print_(user_id=user_id)
        print_("Profanity Check:        " + str(evaluation["profanity"]["is_profane"]), user_id=user_id)
        print_("Profanity Probability:  " + str(evaluation["profanity"]["profanity_prob"]), user_id=user_id)
        print_("Profanity Score:        " + str(evaluation["profanity"]["profanity_score"]), user_id=user_id)
        print_(user_id=user_id)
        print_("Sentiment Check:        " + str(evaluation["sentiment_label"]), user_id=user_id)
        print_(user_id=user_id)
        print_("One Shot Human written: " + str(evaluation["res"]), user_id=user_id)
        print_("Final Score:            " + str(evaluation["final_score"]), user_id=user_id)
