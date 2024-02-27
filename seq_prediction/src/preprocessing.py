import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def preprocess_data(data, selected_attributes, activity_tokenizer, attribute_tokenizers):
    data['start'] = pd.to_datetime(data['start'])
    data['end'] = pd.to_datetime(data['end'])

    data['activity_duration'] = (data['end'] - data['start']).dt.total_seconds().astype(int)

    data['activity_list'] = data.groupby('traceId')['activity'].transform(lambda x: [x.tolist()] * len(x))
    data['activity_durations'] = data.groupby('traceId')['activity_duration'].transform(lambda x: [x.tolist()] * len(x))

    data['next_activity'] = data.groupby('traceId')['activity'].shift(-1)
    data['next_start'] = data.groupby('traceId')['start'].shift(-1)

    data['transition_duration'] = (data['next_start'] - data['end']).dt.total_seconds().fillna(0).astype(int)
    data['transition_durations'] = data.groupby('traceId')['transition_duration'].transform(lambda x: [x.tolist()] * len(x))

    data['total_duration'] = data.groupby('traceId')['activity_duration'].transform('sum') + data.groupby('traceId')['transition_duration'].transform('sum')

    data['activity_tokenized'] = [seq[0] for seq in activity_tokenizer.texts_to_sequences(data['activity'])]
    data['sequence_tokenized'] = activity_tokenizer.texts_to_sequences(data['activity_list'])
    data['next_activity_tokenized'] = [seq[0] if seq else -1 for seq in activity_tokenizer.texts_to_sequences(data['next_activity'].fillna('UNKNOWN'))]
    data['tokenized_transition'] = data['activity_tokenized'].astype(str) + "->" + data['next_activity_tokenized'].astype(str)

    for attr in selected_attributes:
        sequences = attribute_tokenizers[attr].texts_to_sequences(data[attr].astype(str).tolist())
        data[attr] = [seq[0] if seq else -1 for seq in sequences]

    data_sequences = data.drop_duplicates(subset='traceId')[['traceId', 'sequence_tokenized', 'activity_durations', 'transition_durations', 'total_duration'] + selected_attributes]
    return data_sequences

def truncation_splitting(data_sequences, selected_attributes, N_UNIQUE_ACTS, MAX_SEQ_LENGTH):
    def truncate_sequence(seq):
        if len(seq) > 1:
            trunc_point = np.random.randint(1, len(seq))
            truncated = seq[:trunc_point]
            remaining = seq[trunc_point:]
        else:
            truncated = seq
            remaining = []
            trunc_point = len(seq)
        return truncated, remaining, trunc_point

    def truncate_list(lst, trunc_points, offset=0):
        truncated = [item[:tp - offset] for item, tp in zip(lst, trunc_points)]
        remaining = [item[tp - offset:] for item, tp in zip(lst, trunc_points)]
        return truncated, remaining

    data_sequences[['truncated_tokenized', 'remaining_tokenized', 'trunc_point']] = data_sequences['sequence_tokenized'].apply(truncate_sequence).apply(pd.Series)

    data_sequences['truncated_durations'], data_sequences['remaining_durations'] = truncate_list(data_sequences['activity_durations'], data_sequences['trunc_point'])
    data_sequences['truncated_transitions'], data_sequences['remaining_transitions'] = truncate_list(data_sequences['transition_durations'], data_sequences['trunc_point'], 1)

    data_sequences['truncated_total_duration'] = data_sequences['truncated_durations'].apply(sum) + data_sequences['truncated_transitions'].apply(sum)
    data_sequences['remaining_total_duration'] = data_sequences['remaining_durations'].apply(sum) + data_sequences['remaining_transitions'].apply(sum)
    assert all(data_sequences['truncated_total_duration'] + data_sequences['remaining_total_duration'] == data_sequences['total_duration'])

    data_sequences = data_sequences.drop(columns=['truncated_durations', 'truncated_transitions', 'remaining_durations', 'remaining_transitions', 'trunc_point'])

    data_sequences['truncated_tokenized'] = pad_sequences(data_sequences['truncated_tokenized'], maxlen=MAX_SEQ_LENGTH, padding='post').tolist()
    data_sequences['remaining_tokenized'] = pad_sequences(data_sequences['remaining_tokenized'], maxlen=MAX_SEQ_LENGTH, padding='post').tolist()

    train_df, test_df = train_test_split(data_sequences, test_size=0.2, random_state=42)

    train_df = train_df.sort_values(by='traceId')
    test_df = test_df.sort_values(by='traceId')

    X_train_features = train_df[selected_attributes].values
    X_test_features = test_df[selected_attributes].values

    X_train = np.array(train_df['truncated_tokenized'].tolist())
    Y_train = np.array(train_df['remaining_tokenized'].tolist())

    X_test = np.array(test_df['truncated_tokenized'].tolist())
    Y_test = np.array(test_df['remaining_tokenized'].tolist())

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1], 1)

    Y_train_onehot = to_categorical(Y_train.squeeze(), num_classes=N_UNIQUE_ACTS)
    Y_test_onehot = to_categorical(Y_test.squeeze(), num_classes=N_UNIQUE_ACTS)

    return train_df, X_train, X_train_features, Y_train_onehot, test_df, X_test, X_test_features, Y_test_onehot, data_sequences