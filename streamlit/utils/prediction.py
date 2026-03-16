def predict(model, input_df):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability
