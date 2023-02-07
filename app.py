import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from datetime import datetime, date
import pandas as pd


def query(amt, age, transaction_date, transaction_time, gender, category, state):
    q_dict = {'amt': 0, 'age': 0, 'daydate_transaction': 0, 'dayweek_transaction': 0, 'month_transaction': 0,
              'hour_transaction': 0, 'is_male': 0, 'ohe_food_dining': 0, 'ohe_gas_transport': 0, 'ohe_grocery_net': 0,
              'ohe_grocery_pos': 0, 'ohe_health_fitness': 0, 'ohe_home': 0, 'ohe_kids_pets': 0, 'ohe_misc_net': 0,
              'ohe_misc_pos': 0, 'ohe_personal_care': 0, 'ohe_shopping_net': 0, 'ohe_shopping_pos': 0, 'ohe_travel': 0,
              'ohe_CA': 0, 'ohe_CO': 0, 'ohe_CT': 0, 'ohe_DC': 0, 'ohe_DE': 0, 'ohe_FL': 0, 'ohe_GA': 0, 'ohe_AZ': 0,
              'ohe_HI': 0, 'ohe_IA': 0, 'ohe_ID': 0, 'ohe_IL': 0, 'ohe_IN': 0, 'ohe_KS': 0, 'ohe_KY': 0, 'ohe_LA': 0,
              'ohe_MA': 0, 'ohe_MD': 0, 'ohe_ME': 0, 'ohe_MI': 0, 'ohe_MN': 0, 'ohe_MO': 0, 'ohe_MS': 0, 'ohe_NY': 0,
              'ohe_MT': 0, 'ohe_NC': 0, 'ohe_ND': 0, 'ohe_NE': 0, 'ohe_NH': 0, 'ohe_NJ': 0, 'ohe_NM': 0, 'ohe_NV': 0,
              'ohe_OH': 0, 'ohe_OK': 0, 'ohe_OR': 0, 'ohe_PA': 0, 'ohe_RI': 0, 'ohe_SC': 0, 'ohe_SD': 0, 'ohe_TN': 0,
              'ohe_TX': 0, 'ohe_UT': 0, 'ohe_VA': 0, 'ohe_VT': 0, 'ohe_WA': 0, 'ohe_WI': 0, 'ohe_WV': 0, 'ohe_WY': 0,
              'ohe_AL': 0, 'ohe_AR': 0
              }

    q_dict['amt'] = amt
    q_dict['age'] = age
    if gender == 'Male':
        q_dict['is_male'] += 1
    else:
        pass
    transaction_date = datetime.strptime(transaction_date, '%Y-%m-%d')
    q_dict['daydate_transaction'] = transaction_date.day
    q_dict['dayweek_transaction'] = transaction_date.weekday()
    q_dict['month_transaction'] = transaction_date.month
    transaction_time = datetime.strptime(transaction_time, '%H:%M:%S').time()
    q_dict['hour_transaction'] = transaction_time.hour

    dict_shopping = {'Food and Dining': 'ohe_food_dining', 'Gas and Transport': 'ohe_gas_transport',
                     'Grovery Online Shopping': 'ohe_grocery_net', 'Grocery over POS': 'ohe_grocery_pos',
                     'Health and Fitness': 'ohe_health_fitness', 'Home': 'ohe_home', 'Kids and Pets': 'ohe_kids_pets',
                     'Misc. Online Shopping': 'ohe_misc_net', 'Misc. POS': 'ohe_misc_pos',
                     'Personal Care': 'ohe_personal_care',
                     'Shopping Online': 'ohe_shopping_net', 'Shopping over POS': 'ohe_shopping_pos',
                     'Travel': 'ohe_travel'}
    q_dict[dict_shopping[category]] += 1
    us_state_to_abbrev = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
                          "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
                          "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
                          "Kansas": "KS",
                          "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
                          "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
                          "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
                          "New Mexico": "NM",
                          "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
                          "Oklahoma": "OK",
                          "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
                          "South Dakota": "SD",
                          "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
                          "Washington": "WA",
                          "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
                          "American Samoa": "AS",
                          "Guam": "GU", "Northern Mariana Islands": "MP", "Puerto Rico": "PR",
                          "U.S. Virgin Islands": "VI",
                          "United States Minor Outlying Islands": "UM"}
    dict_state = {}
    for key, value in us_state_to_abbrev.items():
        dict_state[key] = str('ohe_') + value
    q_dict[dict_state[state]] += 1
    ser_query = pd.Series(q_dict)
    return ser_query





app = Flask(__name__)
model = pickle.load(open('clfmodel_rf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])



def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = features = [int(x) for x in query(1000, 23, '2020-05-21', '06:44:23', 'Male', 'Food and Dining', 'Alabama')]
    #features = [x for x in request.form.values()]
    amt = request.form.get("Amount")
    age = request.form.get("Age")
    transaction_date = request.form.get("Transaction Date")
    transaction_time = request.form.get("Transaction Time")
    gender = request.form.get("Gender")
    category = request.form.get("Category")
    state = request.form.get("State")
    int_features = query(amt, age, transaction_date, transaction_time, gender, category, state)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = int(prediction[0])
    if output == 1:
        return render_template('index.html', prediction_text='ALERT! This can be a fraudulent transaction'.format(output))
    else:
        return render_template('index.html', prediction_text='It looks like a fair transaction'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)