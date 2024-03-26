import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
from prettytable import PrettyTable
from threading import Thread, Lock
import random
import time
import re
from utils import print_, update_job_status, update_job_result, update_job_values
from config import *

class ModelLoader:
    def __init__(self, model_path, weights_path, device="gpu"):
        self.model, self.tokenizer = self.load_model(model_path, weights_path, device)

    def load_model(self, model_path, weights_path, device="gpu"):
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model_name = model_path.split('/')[-1]
        weights_path_name = weights_path.split('/')[-1]
        mw_name = f"{model_name}-{weights_path_name}"
        
        model = DAIGTModel(model_path, config, tokenizer, pretrained=False)
        
        if torch.cuda.is_available() and (device != 'cpu' or device == 'cuda'):
            model_name = model_path.split('/')[-1]
            print_("Loading model {} on GPU".format(model_name))
            model.load_state_dict(torch.load(weights_path))
            model = model.cuda()
        else:
            print_("Loading model {} on CPU".format(mw_name)) 
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()

        return model, tokenizer

class DAIGTModel(nn.Module):
    def __init__(self, model_path, config, tokenizer, pretrained=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, config=config) if pretrained else AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        

    def forward_features(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return embeddings

    def forward(self, input_ids, attention_mask):
        embeddings = self.forward_features(input_ids, attention_mask)
        logits = self.classifier(embeddings)
        return logits

class ModelEvaluator:
    def __init__(self, models, num_parallel=1):
        self.models = models
        self.num_parallel = num_parallel
        self.table = PrettyTable()
        self.table.field_names = ["Sno", "Model", "Device", "Prediction", "Execution Time"]
        self.table.align["Model"] = "l"
        self.avg_pred_prob = 0.0
        self.user_id = "~"
        self.data = None
        self.job_res_model = []
        
    def _print(self, msg="\n", user_id="~"):
        if user_id != "~":
            print_(msg, user_id)
        else:
            print_(msg, self.user_id)

    def model_score(self, custom_text, user_id="~",data=None):
        
        self._print("\nModel Evaluation started..")

        self.job_res_model = []
        self.avg_pred_prob = 0.0
        self.data = None
        
        if not custom_text:
            self._print("\nNo text to process.")
            return
        
        if user_id:
            self.user_id = user_id

        if data:
            self.data = data
            update_job_status(user_id=self.user_id, update={"progress": 0.4, "updates": ["Model Evaluation started.."]}, job_id=data["job_id"])
        
        pred_probs = []
        lock = Lock()
        self.total_start_time = time.time()

        if self.num_parallel <= 1:
            self._evaluate_serial(custom_text, pred_probs)
        else:
            self._evaluate_parallel(custom_text, pred_probs, lock)
        

        self._print_results(pred_probs)
        job_res = {
            'score': self.avg_pred_prob,
        }
        update_job_result(user_id=self.user_id, update=job_res, job_id=self.data["job_id"])
        update_job_status(user_id=self.user_id, update={"progress": 0.8, "updates": ["Model Evaluation completed.."]}, job_id=self.data["job_id"])
        
        return self.avg_pred_prob

    def _evaluate_serial(self, custom_text, pred_probs):
        print_("Evaluating models serially...")
        update_job_status(user_id=self.user_id, update={"progress": 0.6, "updates": ["Evaluating models serially..."]}, job_id=self.data["job_id"])
        job_res_tmp = {
            "model_score": {
                "models": [], 
                "model_avg": 0,
                "execution_time": 0,
            },
            "score": 0,
            "device": "GPU" if all(next(model.model.parameters()).is_cuda for model in self.models) else "CPU" if all(not next(model.model.parameters()).is_cuda for model in self.models) else "Mix", 
            "parallel": self.num_parallel,
            "max_length": max_len,
            "max_memory": 0,

        }
        job_model = []
        for i, dataset in enumerate(self.models):
            self._print(f"Evaluating model {i + 1}...")
            update_job_status(user_id=self.user_id, update={"progress": 0.65, "updates": [f"Evaluating model {i + 1}..."]}, job_id=self.data["job_id"])
            model_start_time = time.time()
            pred_prob = self._predict_custom_text(dataset.model, dataset.tokenizer, custom_text)
            model_end_time = time.time()
            pred_probs.append(pred_prob)

            model_name = re.search(r'\(model\): (\w+)', str(dataset.model)).group(1) if re.search(r'\(model\): (\w+)', str(dataset.model)) else 'Unknown'
            
            self._print(f"Predicted probability (Model {i + 1}): {pred_prob}", f"\nExecution time: {model_end_time - model_start_time:.2f} seconds")
            model = {
                            "model": f"Model ({i + 1}: {model_name})",
                            "details": str(dataset.model), 
                            "execution_type": "Serial",
                            "score": pred_prob, 
                            "time": model_end_time - model_start_time, 
                            "device": "GPU" if next(dataset.model.parameters()).is_cuda else "CPU"
            }
            job_model.append(model)
            
            self.table.add_row([f"{i + 1}", f"Model {i + 1}", "GPU" if next(dataset.model.parameters()).is_cuda else "CPU", f"{pred_prob:.6f}", f"{model_end_time - model_start_time:.2f} seconds"])
            update_job_status(user_id=self.user_id, update={"progress": 0.7, "updates": [f"Model {i + 1} completed.."]}, job_id=self.data["job_id"])
        
        # job_res = {
        #     "model_score": {
        #         "model_avg": sum(pred_probs) / len(pred_probs),
        #         "execution_time": time.time() - self.total_start_time,
        #     }, 
        #         "score": pred_prob, 
        #         "device": "GPU" if next(dataset.model.parameters()).is_cuda else "CPU", 
        #         "parallel": self.num_parallel, 
        #         "max_length": 768, 
        #         "max_memory": 0, 
        # }
        job_res_tmp["model_score"]["models"] = job_model
        job_res_tmp["model_score"]["model_avg"] = sum(pred_probs) / len(pred_probs)
        job_res_tmp["model_score"]["execution_time"] = time.time() - self.total_start_time

        update_job_result(user_id=self.user_id, update=job_res_tmp, job_id=self.data["job_id"])
        
    def _evaluate_parallel(self, custom_text, pred_probs, lock):
        self._print(f"\nEvaluating models parallel in {self.num_parallel} threads...")
        update_job_status(user_id=self.user_id, update={"progress": 0.6, "updates": [("Evaluating models parallely in " + str(self.num_parallel) + " threads..")]}, job_id=self.data["job_id"])
        job_res_tmp = {
            "model_score": {
                "models": [], 
                "model_avg": 0,
                "execution_time": 0,
            },
            "score": 0,
            "device": "GPU" if all(next(model.model.parameters()).is_cuda for model in self.models) else "CPU" if all(not next(model.model.parameters()).is_cuda for model in self.models) else "Mix", 
            "parallel": self.num_parallel,
            "max_length": max_len,
            "max_memory": 0,
        }

        threads = []
        for i, dataset in enumerate(self.models):
            thread = Thread(target=self._model_score_threaded, args=(dataset.model, dataset.tokenizer, dataset, custom_text, pred_probs, lock, i))
            threads.append(thread)

        self._print("Starting threads...")
        for thread in threads:
            thread.start()

        self._print("\nWaiting for threads to complete...")
        for thread in threads:
            thread.join()

        self.avg_pred_prob = sum(pred_probs) / len(pred_probs)
        job_res_tmp["model_score"]["model_avg"] = self.avg_pred_prob
        job_res_tmp["model_score"]["execution_time"] = time.time() - self.total_start_time
        job_res_tmp["model_score"]["models"] = self.job_res_model
        print(self.job_res_model)
        print()
        print(job_res_tmp)
        print()
        print()

        update_job_status(user_id=self.user_id, update={"progress": 0.7, "updates": ["Model evaluation completed.."]}, job_id=self.data["job_id"])
        update_job_result(user_id=self.user_id, update=job_res_tmp, job_id=self.data["job_id"])


    def _model_score_threaded(self, model, tokenizer, dataset, custom_text, pred_probs, lock, i):
        self._print(f"Evaluating model {i + 1}...")
        update_job_status(user_id=self.user_id, update={"progress": 0.65, "updates": [f"Evaluating model {i + 1}..."]}, job_id=self.data["job_id"])
        model_name = re.search(r'\(model\): (\w+)', str(dataset.model)).group(1) if re.search(r'\(model\): (\w+)', str(dataset.model)) else 'Unknown'
           
        job_model = {
            "model": f"Model ({i + 1}: {model_name})",
            "details": str(dataset.model), 
            "execution_type": "Parallel", 
            "score": 0, 
            "time": 0, 
            "device": "GPU" if next(dataset.model.parameters()).is_cuda else "CPU"
            }
        model_start_time = time.time()
        pred_prob = self._predict_custom_text(model, tokenizer, custom_text)
        model_end_time = time.time()
        with lock:
            pred_probs.append(pred_prob)
        self._print(f"Predicted probability (Model {i + 1}): {pred_prob}", f"\nExecution time: {model_end_time - model_start_time:.2f} seconds")
        job_model["score"] = pred_prob
        job_model["time"] = model_end_time - model_start_time
        self.job_res_model.append(job_model)
        update_job_status(user_id=self.user_id, update={"progress": 0.65, "updates": [f"Model {i + 1} completed.."]}, job_id=self.data["job_id"])
        self.table.add_row([f"{i + 1}", f"Model {i + 1}", "GPU" if next(dataset.model.parameters()).is_cuda else "CPU", f"{pred_prob:.6f}", f"{model_end_time - model_start_time:.2f} seconds"])

    def _predict_custom_text(self, model, tokenizer, text):
        
        model.eval()

        tokenized = tokenizer(
            text=text,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )

        device = next(model.parameters()).device
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        pred_prob = torch.sigmoid(logits).cpu().item()
        return pred_prob    

    def _print_results(self, pred_probs,sortby="Sno"):
        self.avg_pred_prob = sum(pred_probs) / len(pred_probs)

        self._print(user_id=self.user_id)
        self._print(f"Average predicted probability: {self.avg_pred_prob}")
        self._print(f"Device: {'GPU' if all(next(model.model.parameters()).is_cuda for model in self.models) else 'CPU' if all(not next(model.model.parameters()).is_cuda for model in self.models) else 'Mix'}", f"\nExecution: {'Serial' if self.num_parallel <= 1 else f'Parallel ({self.num_parallel} threads)'}")
        total_execution_time = time.time() - self.total_start_time
        self._print(f"Total execution time: {total_execution_time:.2f} seconds")

        if sortby == "Sno":
            self.table.sortby = "Sno"
        elif sortby == "Device":
            self.table.sortby = "Device"
        elif sortby == "Prediction":
            self.table.sortby = "Prediction"
        elif sortby == "Time":
            self.table.sortby = "Execution Time"
        
        self.table.add_row([f"{len(self.models) + 1}", "Result", "GPU" if all(next(model.model.parameters()).is_cuda for model in self.models) else "CPU" if all(not next(model.model.parameters()).is_cuda for model in self.models) else "Mix", f"{self.avg_pred_prob:.6f}", f"{total_execution_time:.2f} seconds"])
        self._print(user_id=self.user_id)
        st = str("Results Summary " if sortby == "Sno" else "Results Summary (" + sortby + ") ")
        self._print(st)
        print(self.table)
        self._print(str(self.table))
        self._print(user_id=self.user_id)
        self.table.clear_rows()

def model(model_paths_weights=paths_weights, random_sel=-1):
    
    random_sel = len(model_paths_weights) if (random_sel == 0 or random_sel == -1) else random_sel

    new_model_paths_weights = [(model_path, weights_path, device) for model_path, weights_path, value, device in model_paths_weights if value == 1]

    model_paths_weights = new_model_paths_weights

    if random_sel >= len(model_paths_weights):
        print_(f"Warning: random_sel ({random_sel}) is larger than the number of available models. Using all available models.\n")
        random_sel = len(model_paths_weights)
    else:
        print_("Loadding Model(s)...")

    loaded_models = [ModelLoader(model_path, weights_path, device) for model_path, weights_path, device in model_paths_weights] if random_sel == -1 \
               else [ModelLoader(model_path, weights_path, device) for model_path, weights_path, device in random.sample(model_paths_weights, random_sel)]

    print_(len(loaded_models), "random" if random_sel < len(model_paths_weights) else "", "model(s) loaded\n")

    return loaded_models    

    


# def main():
#     import profanity_scorer as ps

#     zero_shot = True

#     while True:
#         input("Press enter to process the text from input.txt")
#         text = ""
#         with open('data/input.txt', 'r') as file:
#             text = file.read()
#             if not text:
#                 return
        
#         eval_tex = ps.sentiment_analyzer.analyze_text(text)
#         ps.print_eval_text(eval_tex)
        
#         if eval_tex["res"] >= 0.7 and zero_shot==True:
#             print_("\nZero shot : Positive (Human Genereated)\n")
#         else:
#             print_("\nZero shot : Negative (AI Generated)\n")
#             evaluator.model_score(eval_tex["text_input"])
 

print_()
print_("\n")
models = model(random_sel=random_sel)
evaluator = ModelEvaluator(models, num_parallel=num_parallel)



# if __name__ == "__main__":
#     main(use_dataset=False, num_parallel=2)
#     main() # you can use this to test the model alone



# json format | good naming convention
# define every return | return one single json
# option to disable running model | text_proc | functions , etc
# option to disable printing
# log every process
# define every args/get params
# all params preset | detective mode | analysis mode | zero-shot mode | Random mode | Recommendation mode | etc
# link with front-end 