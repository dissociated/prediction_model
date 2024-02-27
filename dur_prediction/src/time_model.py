import pandas as pd
import numpy as np
import utils

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def duration_preprocessing(predicted_train_df, predicted_test_df, selected_attributes):
    # Sequences (predicted_sequence)
    X_train_seq = np.array(predicted_train_df['predicted_sequence'].tolist())
    X_test_seq = np.array(predicted_test_df['predicted_sequence'].tolist())

    # Additional Features (selected attributes)
    X_train_additional = predicted_train_df[selected_attributes].values
    X_test_additional = predicted_test_df[selected_attributes].values

    # Targets (remaining_duration)
    y_train_regression = predicted_train_df['remaining_duration'].values
    y_test_regression = predicted_test_df['remaining_duration'].values
    return X_train_seq, X_train_additional, y_train_regression, X_test_seq, X_test_additional, y_test_regression

def train_regression_model(X_train_seq, X_train_additional, y_train_regression, X_test_seq, X_test_additional, y_test_regression, N_UNIQUE_ACTS):
    # 1. Sequence Input and Processing
    sequence_input = Input(shape=(X_train_seq.shape[1],))
    embedded_sequences = Embedding(input_dim=N_UNIQUE_ACTS, output_dim=64)(sequence_input)
    lstm_out = LSTM(64, return_sequences=True)(embedded_sequences)
    lstm_out = LSTM(32)(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)  # Add dropout after LSTM

    # 2. Additional Features Input and Processing
    feature_input = Input(shape=(X_train_additional.shape[1],))
    dense_feature = Dense(64, activation='relu')(feature_input)
    dense_feature = Dropout(0.2)(dense_feature)  # Add dropout after first Dense layer
    dense_feature = Dense(32, activation='relu')(dense_feature)

    # 3. Combining the sequence and feature processing paths
    combined = Concatenate(axis=-1)([lstm_out, dense_feature])
    combined_dense = Dense(32, activation='relu')(combined)

    # 4. Regression Output Layer
    regression_output = Dense(1, activation='linear')(combined_dense)  # Linear activation for regression

    # Create and compile the model
    regression_model = Model(inputs=[sequence_input, feature_input], outputs=regression_output)
    regression_model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.01))

    # Train the model
    regression_history = regression_model.fit([X_train_seq, X_train_additional], y_train_regression,
                                            batch_size=32, epochs=50,
                                            validation_data=([X_test_seq, X_test_additional], y_test_regression))
    
    return regression_model, regression_history

def postprocess_predictions(regression_model, X_train_seq, X_train_additional, predicted_train_df, X_test_seq, X_test_additional, predicted_test_df, activity_tokenizer, selected_attributes):
    # 1. Predict the remaining duration for both train and test datasets
    predicted_remaining_train = regression_model.predict([X_train_seq, X_train_additional])
    predicted_remaining_test = regression_model.predict([X_test_seq, X_test_additional])

    # 2. Add the predicted remaining duration to the truncated duration
    predicted_train_df['predicted_total_duration'] = predicted_train_df['truncated_duration'] + predicted_remaining_train.flatten()
    predicted_test_df['predicted_total_duration'] = predicted_test_df['truncated_duration'] + predicted_remaining_test.flatten()

    # 3. If you want to combine both dataframes (train and test) for further analysis:
    combined_df = pd.concat([predicted_train_df, predicted_test_df], axis=0).reset_index(drop=True)
    combined_df['predicted_sequence'] = combined_df['predicted_sequence'].apply(lambda x: utils.decode_token_sequence(x, activity_tokenizer))
    
    sla = combined_df['total_duration'].median()

    combined_df['risk_predicted'] = combined_df['predicted_total_duration'].apply(lambda x: utils.get_risk(x, sla))
    combined_df['risk'] = combined_df['total_duration'].apply(lambda x: utils.get_risk(x, sla))
    combined_df['correct_prediction'] = combined_df['risk'] == combined_df['risk_predicted']
    accuracy_percentage = combined_df['correct_prediction'].mean() * 100

    return combined_df, sla, accuracy_percentage
