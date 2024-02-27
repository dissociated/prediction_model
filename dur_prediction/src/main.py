import preprocessing
import time_model
import utils

import os
from google.cloud import storage

# Get environment variables
project_id = os.environ.get('PROJECT_ID')
bucket_name = os.environ.get('GCS_BUCKET')
data_path = os.environ.get('DATA_PATH')
# seq_model_path = os.environ.get('SEQ_MODEL_PATH')
# time_model_path = os.environ.get('TIME_MODEL_PATH')

gcs = storage.Client(project=project_id)
bucket = gcs.bucket(bucket_name)

def main():
    data = utils.load_data_from_gcs(bucket, data_path)

    # 0. Get parameters and attributes
    N_UNIQUE_ACTS, MAX_SEQ_LENGTH, selected_attributes, activity_tokenizer, attribute_tokenizers = utils.get_params_tokenizers(data)

    # 1. Data Preprocessing
    data_sequences = preprocessing.preprocess_data(data, selected_attributes, activity_tokenizer, attribute_tokenizers)

    train_df, X_train, X_train_features, y_train_regression, test_df, X_test, X_test_features, y_test_regression, data_sequences = preprocessing.truncation_splitting(data_sequences, selected_attributes, N_UNIQUE_ACTS, MAX_SEQ_LENGTH)
    
    # 5. Train Regression Model for Duration Prediction
    regression_model, regression_history = time_model.train_regression_model(
        X_train, X_train_features, y_train_regression, 
        X_test, X_test_features, y_test_regression, 
        N_UNIQUE_ACTS
    )
    
    # # # 6. Duration Predictions
    # combined_df, sla, risk_accuracy = time_model.postprocess_predictions(
    #      regression_model, X_train_seq, X_train_additional, predicted_train_df, 
    #      X_test_seq, X_test_additional, predicted_test_df, activity_tokenizer, selected_attributes
    #  )
    
    regression_model.save(os.getenv("AIP_MODEL_DIR"))

if __name__ == "__main__":
    main()