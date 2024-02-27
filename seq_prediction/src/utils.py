
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from io import BytesIO
import pickle 
import tensorflow as tf
import tempfile

def get_params_tokenizers(data):
    N_UNIQUE_ACTS = len(data['activity'].unique()) + 1 
    MAX_SEQ_LENGTH  = data.groupby('traceId')['activity'].count().max()

    exclude_columns = ["traceId", "index", "activity", "start", "end", "duration", "EVENTID", "VariantId"]
    attributes = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    selected_attributes = [attr for attr in attributes if attr not in exclude_columns]

    activity_tokenizer = Tokenizer(filters='', lower=False, split='Ω')
    activity_tokenizer.fit_on_texts(data['activity'].unique().tolist())

    attribute_tokenizers = {}
    for attr in selected_attributes:
        tok = Tokenizer(filters='', lower=False, split='Ω')
        tok.fit_on_texts(data[attr].astype(str).tolist())
        attribute_tokenizers[attr] = tok

    return N_UNIQUE_ACTS, MAX_SEQ_LENGTH, selected_attributes, activity_tokenizer, attribute_tokenizers

def decode_token_sequence(token_sequence, tokenizer):
    return [tokenizer.index_word[token] for token in token_sequence if token in tokenizer.index_word]

def get_risk(total_duration, sla):
    if (total_duration > sla) & (total_duration < sla*1.5):
        return 1
    elif (total_duration > sla*1.5) & (total_duration < sla*2):
        return 2
    elif total_duration > sla*2:
        return 3
    else:
        return 0
    
def load_from_gcs(bucket, filename):
    """Load data from Google Cloud Storage"""
    blob = bucket.blob(filename)
    
    # Load bytes from GCS
    serialized_data = blob.download_as_bytes()
    
    # Deserialize the data
    data = pickle.loads(serialized_data)
    return data

def load_model_from_gcs(bucket, source_filename):
    """Load a TensorFlow model directly from GCS without local storage."""
    blob = bucket.blob(source_filename)
    
    # Download model from GCS to a BytesIO buffer
    buffer = BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    
    # Load model from the buffer
    model = tf.keras.models.load_model(buffer)
    return model

def load_data_from_gcs(bucket, data_path):
    blob = bucket.blob(data_path)
    data_as_bytes = blob.download_as_bytes()
    data = pd.read_csv(BytesIO(data_as_bytes))
    return data

def save_object_to_gcs(bucket, object, destination_filename):
    """Serialize and save object to Google Cloud Storage"""
    blob = bucket.blob(destination_filename)
    
    # Serialize the object
    serialized_data = pickle.dumps(object)
    blob.upload_from_string(serialized_data)

def save_model_directly_to_gcs(model, bucket, destination_filename):
    """Save the TensorFlow model directly to GCS without local storage."""
    
    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
        # Serialize the model to the temp file
        tf.keras.models.save_model(model, tmp.name, save_format='h5')
        
        # Upload the temp file content to GCS
        blob = bucket.blob(destination_filename)
        blob.upload_from_filename(tmp.name, content_type='application/octet-stream')

