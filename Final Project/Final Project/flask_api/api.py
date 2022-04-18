from arima_prod import calc_change_ARIMA, fetch_nft_df
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from xgboost_prod import get_x_and_y
import xgboost
import sys

app = Flask(__name__)
df = None

@app.before_first_request
def before_first_request_func():
    global df
    print("loading file", file=sys.stderr)
    df = pd.read_pickle("https://nft-capstone.s3.us-west-1.amazonaws.com/df_all_values.pkl.gz")
    print("doing dummies", file=sys.stderr)
    df = pd.get_dummies(df, columns=['Category'])

@app.route('/predict_nft', methods=['GET'])
def predict_nft():
    args = request.args
    collection_name = args['collection_name']
    nft_name = args['nft_name']
    category = args['category']
    nft_id = args['nft_id']
    df_nft_id = fetch_nft_df(df, nft_name, nft_id)
    num_transac = len(df_nft_id)
    if num_transac <= 2:
        return "cannot make prediction for transaction"
    elif num_transac > 2 and num_transac <= 49:
        df_collection_nft_second = df_nft_id[(df_nft_id['First_item_Collection_cleaned'] == 0) & (df_nft_id['First_item_Unique_id_collection'] == 0)]
        df_collection_nft_second_sorted = df_collection_nft_second.sort_values(by=['Datetime_updated_seconds'])
        X, y, indices = get_x_and_y(df_collection_nft_second_sorted, 1)
        with open("xgboost.joblib", 'rb') as f:
            xgboost = joblib.load(f)
            prediction = xgboost.predict(X.loc[-1:])
            pred_str = ""
            if prediction[len(prediction) - 1] == 1:
                pred_str = "Positive"
            else:
                pred_str = "Negative"
            print(prediction, file=sys.stderr)
            return jsonify(prediction = pred_str, model = "xgboost")
    else:
        df_arima = df[['Price_USD', 'Datetime_updated', 'Collection_cleaned', 'Unique_id_collection']]
        return jsonify(prediction = calc_change_ARIMA(df_arima, nft_name, nft_id, False), model = 'arima')

app.run(host="0.0.0.0", port=8000, debug=True)
