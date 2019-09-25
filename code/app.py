import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS


def create_app():

    app = Flask(__name__)
    CORS(app)
    model = joblib.load('code/nba_model')

    @app.route('/api', methods=['POST'])
    def predict():
        # get the data
        data = request.get_json(force=True)

        predict_request = [data['all_nba'], data['all_star'], data['draft_yr'], data['pk'],
                           data['fg_percentage'], data['tp_percentage'], data['ft_percentage'],
                           data['minutes_per_game'], data['points_per_game'], data['trb_per_game'],
                           data['assists_per_game'], data['ws_per_game'], data['bpm'], data['vorp'],
                           data['attend_college']]

        array_pred_req = np.array(predict_request).reshape(1, -1)

        # predictions
        y_pred = model.predict(array_pred_req)

        # give it back
        output = {"y_pred": int(y_pred[0])}
        return jsonify(results=output)
    return app


if __name__ == '__main__':
    app.run()
