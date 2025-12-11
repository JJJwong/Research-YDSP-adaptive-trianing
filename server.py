from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.optimize import minimize

# ----------------------------
# Flask backend
# ----------------------------
app = Flask(__name__)
CORS(app)  # allow all origins

# ----------------------------
# Game backend state (same as original Colab logic)
# ----------------------------

# Fuzzy logic setup
performance_input = ctrl.Antecedent(np.arange(0,91,0.1), 'performance_input')
adjustment_output = ctrl.Consequent(np.arange(-15,15,0.1), 'adjustment_output')

performance_input['poor'] = fuzz.gaussmf(performance_input.universe,0,10)
performance_input['adequate'] = fuzz.gaussmf(performance_input.universe,35,5)
performance_input['good'] = fuzz.gaussmf(performance_input.universe,90,20)

adjustment_output['decrease'] = fuzz.gaussmf(adjustment_output.universe,-10,2)
adjustment_output['maintain'] = fuzz.gaussmf(adjustment_output.universe,0,3)
adjustment_output['increase'] = fuzz.gaussmf(adjustment_output.universe,10,2)

rule1 = ctrl.Rule(performance_input['poor'], adjustment_output['decrease'])
rule2 = ctrl.Rule(performance_input['adequate'], adjustment_output['maintain'])
rule3 = ctrl.Rule(performance_input['good'], adjustment_output['increase'])

difficulty_ctrl = ctrl.ControlSystem([rule1,rule2,rule3])
diff_change = ctrl.ControlSystemSimulation(difficulty_ctrl)

# IRT skill tracking
a = 1.0
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

def update_skill(success_extent, raw_difficulty, theta_hat):
  scaled_difficulty = scale_difficulty(raw_difficulty)
  return map_update_theta(theta_hat, a, scaled_difficulty, success_extent, mu0=theta_hat, sigma0=1)

# ----------------------------
# Flask routes
# ----------------------------
@app.route("/fuzzy_update", methods=["POST"])
def fuzzy_update():
    data = request.json
    difficulty = float(data["difficulty"])
    score = float(data["score"])
    theta_hat = float(data["theta_hat"])
    success_extent = 0.5+(score-35)/40;
    success_extent = min(success_extent,1)
    success_extent = max(success_extent,0)
    
    new_skill = update_skill(success_extent, difficulty, theta_hat)

    # --- Fuzzy logic adjustment ---
    diff_change.input["performance_input"] = score
    diff_change.compute()
    fuzzy_adjust = diff_change.output["adjustment_output"]

    new_difficulty = difficulty + fuzzy_adjust
    new_difficulty = max(1, min(new_difficulty, 100))

    return jsonify({"new_difficulty": new_difficulty, "new_skill": new_skill})

@app.route("/irt_update", methods=["POST"])
def irt_update():
    data = request.json
    difficulty = float(data["difficulty"])
    score = float(data["score"])
    theta_hat = float(data["theta_hat"])
    success_extent = 0.5+(score-35)/40;
    success_extent = min(success_extent,1)
    success_extent = max(success_extent,0)

    new_skill = update_skill(success_extent, difficulty, theta_hat)

    new_difficulty = select_difficulty(theta_hat, a, target_p, 1, 100)

    return jsonify({"new_difficulty": new_difficulty, "new_skill": new_skill})
