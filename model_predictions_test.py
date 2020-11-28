import machine_learning_models
import feature_extraction_image_preprocessing
import xray_only_model

import pandas as pd

if __name__ == "__main__":
    model_1 = machine_learning_models.load_keras_model("optimum_xray_less_dense.json","optimum_xray_less_dense.h5")
    model_2 = machine_learning_models.load_keras_model("optimum_xray_model.json","optimum_xray_model.h5")
    csv_x_data, csv_y_data = machine_learning_models.parse_csv_data("test.csv", False)
    raw_x_data = csv_x_data[:,0]
    raw_x_data = xray_only_model.format_images("resized_test", raw_x_data)
    x_data = raw_x_data.reshape(94,600,600,1)
    predictions_1 = model_1.predict(x_data)
    predictions_1 = predictions_1.reshape(94)
    predictions_2 = model_2.predict(x_data)
    predictions_2 = predictions_2.reshape(94)
    test_predictions = {"Filename": [], "Probability of COVID (Model 1)": [], "Probability of COVID (Model 2)": []}
    for idx in range(0,94):
        test_predictions["Filename"].append(csv_x_data[idx,0])
        test_predictions["Probability of COVID (Model 1)"].append(predictions_1[idx])
        test_predictions["Probability of COVID (Model 2)"].append(predictions_2[idx])
    prediction_dataframe = pd.DataFrame(test_predictions)
    prediction_dataframe.to_csv('Predictions.csv',index=False)