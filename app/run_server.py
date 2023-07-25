from flask import Flask, request, jsonify
import pandas as pd
import dill

# Загружаем обученные модели
with open('models/logreg_pipeline.dill', 'rb') as in_strm:
    model = dill.load(in_strm)

# Обработчики и запуск Flask
app = Flask(__name__)

@app.route("/", methods=["GET"])
def general():
    return "Welcome to prediction process"

@app.route('/predict', methods=['POST'])
def predict():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    description, company_profile, benefits, requirements = "", "", "", ""
    request_json = request.get_json()

    if request_json["description"]:
        description = request_json['description']

    if request_json["company_profile"]:
        company_profile = request_json['company_profile']

    if request_json["benefits"]:
        benefits = request_json['benefits']

    if request_json["requirements"]:
        requirements = request_json['requirements']

    #print(description)
    preds = model.predict_proba(pd.DataFrame({"description": [description],
                                              "company_profile": [company_profile],
                                              "benefits": [benefits],
                                              "requirements": [requirements]}))
    data["predictions"] = preds[:, 1][0]
    data["description"] = description
        # indicate that the request was a success
    data["success"] = True
    print('OK')

        # return the data dictionary as a JSON response
    return jsonify(data)


if __name__ == '__main__':
    app.run()