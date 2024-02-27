import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Bidirectional, RepeatVector, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def mask_acc(y_true, y_pred):
    # Mask for the true values (not equal to 0)
    mask = K.cast(K.max(y_true, axis=-1), K.floatx())
    
    # Calculate accuracy for the positions where mask is 1 and true value is not zero
    y_true_labels = K.cast(K.argmax(y_true, axis=-1), K.floatx())
    y_pred_labels = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
    
    # Exclude zero labels
    non_zero_mask = K.cast(K.greater(y_true_labels, 0), K.floatx())
    
    is_correct = K.cast(K.equal(y_true_labels, y_pred_labels), K.floatx()) * mask * non_zero_mask
    total_correct = K.sum(is_correct)
    total_values = K.sum(mask * non_zero_mask)
    
    return total_correct / total_values

def seq_acc(y_true, y_pred):
    # Convert predictions to label indices
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_true_labels = K.argmax(y_true, axis=-1)
    
    # Check if for each sample, the predicted sequence is equal to the true sequence
    correct_preds = K.all(K.equal(y_true_labels, y_pred_labels), axis=-1)
    
    # Compute the accuracy based on fully correct sequences
    accuracy = K.mean(correct_preds)
    return accuracy

def create_and_train_model(X_train, X_train_features, Y_train_onehot, X_test, X_test_features, Y_test_onehot, N_UNIQUE_ACTS):
    # 1. Sequence Input and Processing
    sequence_input = Input(shape=(X_train.shape[1],))
    embedded_sequences = Embedding(input_dim=N_UNIQUE_ACTS, output_dim=64)(sequence_input)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(embedded_sequences)
    lstm_out = Dropout(0.15)(lstm_out)  # Add dropout after LSTM

    # 2. Feature Input and Processing
    feature_input = Input(shape=(X_train_features.shape[1],))
    dense_feature = Dense(64, activation='relu')(feature_input)
    dense_feature = Dropout(0.15)(dense_feature)  # Add dropout after first Dense layer
    dense_feature = Dense(64, activation='relu')(dense_feature)
    repeated_feature = RepeatVector(X_train.shape[1])(dense_feature) 

    # 3. Combining the sequence and feature processing paths
    concatenated = Concatenate(axis=-1)([lstm_out, repeated_feature])
    combined_dense = Dense(64, activation='relu')(concatenated)

    # 4. Output Layer
    output = Dense(N_UNIQUE_ACTS, activation='softmax')(combined_dense)

    # Create and compile the model
    model = Model(inputs=[sequence_input, feature_input], outputs=output)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=[mask_acc, seq_acc])

    history = model.fit([X_train, X_train_features], 
                        Y_train_onehot, 
                        batch_size=32, 
                        epochs=10, 
                        validation_data=([X_test, X_test_features], Y_test_onehot))
    return model, history


def generate_predictions(model, X_train, X_train_features, train_df, X_test, X_test_features, test_df, activity_tokenizer):
    # Predict the sequences for the test set using the LSTM model
    predicted_sequences1 = model.predict([X_train, X_train_features])
    predicted_sequences2 = model.predict([X_test, X_test_features])

    predicted_activity_indices1 = [np.argmax(seq, axis=-1) for seq in predicted_sequences1]
    predicted_activity_indices2 = [np.argmax(seq, axis=-1) for seq in predicted_sequences2]

    train_df['predicted_sequence'] = predicted_activity_indices1
    test_df['predicted_sequence'] = predicted_activity_indices2

    combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    combined_df = combined_df[['traceId', 'predicted_sequence']].copy()

    def decode_token_sequence(token_sequence, tokenizer):
        return [tokenizer.index_word[token] for token in token_sequence if token in tokenizer.index_word]

    combined_df['predicted_sequence'] = combined_df['predicted_sequence'].apply(lambda x: decode_token_sequence(x, activity_tokenizer))

    return combined_df