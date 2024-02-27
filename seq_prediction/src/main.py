import preprocessing
import seq_model
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

    train_df, X_train, X_train_features, Y_train_onehot, test_df, X_test, X_test_features, Y_test_onehot, data_sequences = preprocessing.truncation_splitting(data_sequences, selected_attributes, N_UNIQUE_ACTS, MAX_SEQ_LENGTH)
    
    # 2. Sequence Model Creation & Training
    model, history = seq_model.create_and_train_model(
        X_train, X_train_features, Y_train_onehot, 
        X_test, X_test_features, Y_test_onehot, 
        N_UNIQUE_ACTS)
    
    # # 3. Generate Predictions from Sequence Model
    # predicted_df = seq_model.generate_predictions(
    #     model, X_train, X_train_features, train_df, 
    #     X_test, X_test_features, test_df, activity_tokenizer)
    
    model.save(os.getenv("AIP_MODEL_DIR"))
    utils.save_object_to_gcs(bucket, selected_attributes, 'lstm_tests/selected_attributes.pkl')
    utils.save_object_to_gcs(bucket, MAX_SEQ_LENGTH, 'lstm_tests/MAX_SEQ_LENGTH.pkl')
    
    utils.save_object_to_gcs(bucket, activity_tokenizer, 'lstm_tests/activity_tokenizer.pkl')
    utils.save_object_to_gcs(bucket, attribute_tokenizers, 'lstm_tests/attribute_tokenizers.pkl')

if __name__ == "__main__":
    main()