from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.optimize import minimize
import os
import gspread
from google.oauth2.service_account import Credentials
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

cred_str = os.getenv("API_KEY")   # contains the entire JSON text
cred_dict = json.loads(cred_str) # convert string → dict

scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(cred_dict, scopes=scopes)
client = gspread.authorize(creds)

sheet_id = "1IFBqyjPGztl92D5A-pNssZNmviiEyoq16RPOecg30Kg"
sheet = client.open_by_key(sheet_id).sheet1

# Flask backend
app = Flask(__name__)
CORS(app)  # allow all origins

# Game backend state (same as original Colab logic)

# Fuzzy logic setup
performance_input = ctrl.Antecedent(np.arange(0,91,0.1), 'performance_input')
adjustment_output = ctrl.Consequent(np.arange(-15,15,0.1), 'adjustment_output')

performance_input['poor'] = fuzz.gaussmf(performance_input.universe,0,10)
performance_input['adequate'] = fuzz.gaussmf(performance_input.universe,30,5)
performance_input['good'] = fuzz.gaussmf(performance_input.universe,90,20)

adjustment_output['decrease'] = fuzz.gaussmf(adjustment_output.universe,-10,2)
adjustment_output['maintain'] = fuzz.gaussmf(adjustment_output.universe,0,3.5)
adjustment_output['increase'] = fuzz.gaussmf(adjustment_output.universe,10,2)

rule1 = ctrl.Rule(performance_input['poor'], adjustment_output['decrease'])
rule2 = ctrl.Rule(performance_input['adequate'], adjustment_output['maintain'])
rule3 = ctrl.Rule(performance_input['good'], adjustment_output['increase'])

difficulty_ctrl = ctrl.ControlSystem([rule1,rule2,rule3])
diff_change = ctrl.ControlSystemSimulation(difficulty_ctrl)

# IRT skill tracking
a = 1
target_p = 0.7

def adjust_difficulty(val):
    global difficulty, target_size, difficulty_distance, target_speed
    difficulty = round(val,2)
    difficulty = max(1,difficulty)
    difficulty = min(100,difficulty)
    target_size = 50 - 0.6*(difficulty-50)
    difficulty_distance = 30 + 4*difficulty
    target_speed = max(0, 1 + 0.025*(difficulty-50))

def calculate_difficulty(score_val):
    global diff_change, difficulty
    diff_change.input['performance_input'] = score_val
    diff_change.compute()
    adjust_difficulty(difficulty + diff_change.output['adjustment_output'])

def scale_difficulty(b_raw, min_raw=1, max_raw=100, min_scaled=-3, max_scaled=3):
    return min_scaled + (max_scaled - min_scaled)*(b_raw - min_raw)/(max_raw - min_raw)

def unscale_difficulty(b_scaled, min_raw=1, max_raw=100, min_scaled=-3, max_scaled=3):
    return min_raw + (b_scaled - min_scaled)*(max_raw - min_raw)/(max_scaled - min_scaled)

def prob_2pl(theta,a,b):
    return 1 / (1 + np.exp(-a*(theta-b)))

def neg_log_posterior(theta,a,b,x,mu0=0.0,sigma0=1.0):
    p = prob_2pl(theta,a,b)
    eps = 1e-12
    nll = -(x*np.log(p+eps) + (1-x)*np.log(1-p+eps))
    prior = 0.5*((theta-mu0)**2)/(sigma0**2)
    return nll + prior

def map_update_theta(theta_prior,a,b,x,mu0=0.0,sigma0=1.0):
    res = minimize(lambda th: neg_log_posterior(th,a,b,x,mu0,sigma0), x0=np.array([theta_prior]), method='L-BFGS-B')
    return float(res.x[0]) if res.success else float(theta_prior)

def select_difficulty(theta_hat,a,target_p=0.7,min_raw=1,max_raw=100):
    logit = np.log(target_p/(1-target_p))
    b_scaled = theta_hat - logit/a
    b_scaled = np.clip(b_scaled,-3,3)
    return unscale_difficulty(b_scaled,min_raw,max_raw)

def update_skill(success_extent, raw_difficulty, theta_hat, theta_prior):
  scaled_difficulty = scale_difficulty(raw_difficulty)
  return map_update_theta(theta_hat, a, scaled_difficulty, success_extent, mu0=theta_prior, sigma0=1)

# Flask routes

def preprocess_request(data):
    user_id = int(data["user_id"])
    difficulty = float(data["difficulty"])
    score = float(data["score"])
    theta_hat = float(data["theta_hat"])
    theta_prior = float(data["theta_prior"])
    global_task = int(data["global_task"])
    local_task = int(data["local_task"])

    success_extent = 1 + (score - 30) / 10
    success_extent = min(max(success_extent, 0), 1)

    new_skill = update_skill(success_extent, difficulty, theta_hat, theta_prior)

    return user_id, difficulty, score, theta_hat, success_extent, new_skill, global_task, local_task

@app.route("/")
def home():
    return "OK", 200

@app.route("/fuzzy_update", methods=["POST"])
def fuzzy_update():
    data = request.json
    user_id, difficulty, score, theta_hat, success_extent, new_skill, global_task, local_task = preprocess_request(data)
    sheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_task, difficulty, theta_hat, score, success_extent, "no" if local_task>1 else "yes", user_id])

    # --- Fuzzy logic adjustment ---
    diff_change.input["performance_input"] = score
    diff_change.compute()
    fuzzy_adjust = diff_change.output["adjustment_output"]

    new_difficulty = difficulty + fuzzy_adjust

    # Special condition if user_id is even (test the 1-up 1-down approach instead)
    if user_id % 2 == 0:
        if score == 30:
            new_difficulty = difficulty
        elif score < 30:
            new_difficulty = difficulty - 5
        else:
            new_difficulty = difficulty + 5

    new_difficulty = max(1, min(new_difficulty, 100))

    return jsonify({"new_difficulty": new_difficulty, "new_skill": new_skill, "new_scaled_skill": select_difficulty(new_skill, a, target_p, 1, 100)})

@app.route("/irt_update", methods=["POST"])
def irt_update():
    data = request.json
    user_id, difficulty, score, theta_hat, success_extent, new_skill, global_task, local_task = preprocess_request(data)
    if difficulty > -1:
        sheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_task, difficulty, theta_hat, score, success_extent, "no", user_id])
    
    new_difficulty = select_difficulty(new_skill if difficulty>-1 else theta_hat, a, target_p, 1, 100)

    return jsonify({"new_difficulty": new_difficulty, "new_skill": new_skill, "new_scaled_skill": new_difficulty})

if __name__ == "__main__":
    app.run(debug=True)    