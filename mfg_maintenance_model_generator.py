from argparse import ArgumentParser
from os import path, environ, listdir

import pandas as pd
import json
import numpy as np

from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import joblib

features = ['Air_temperature', 'Process_temperature', 'Rotational_speed', 'Torque']

labels = 'Target'

def _model_(args, x_train, y_train, x_test, y_test):

    # define preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    label_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder())
    ])
    one_hot_code_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, [0, 1, 2, 3] ),
            ('categorical', label_transformer, []),
            ('one_hot_code', one_hot_code_transformer, [])
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ensemble.RandomForestClassifier(n_estimators=args.n_estimators,max_depth=args.max_depth))
    ])



    pipeline.fit(x_train, y_train)
    
    print("Training Accuracy: {:.3f}".format(pipeline.score(x_train,y_train)))
    print("Testing Accuracy: {:.3f}".format(pipeline.score(x_test,y_test)))
    
    return pipeline

def _load_data(file_path, channel):
    # Read the set of files and read them all into a single pandas dataframe
    input_files = [ path.join(file_path, file) for file in listdir(file_path) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(file_path, channel))
        
    raw_data = [ pd.read_csv(file, header=0, engine="python") for file in input_files ]
    df = pd.concat(raw_data)

    print(df.columns)

    x = df[features]
    y = df[[labels]]
    return x, y

def _parse_args():
    parser = ArgumentParser()

    # Hyper parameters
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default= environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--test', type=str, default=environ.get('SM_CHANNEL_TESTING'))
    
    parser.add_argument('--hosts', type=list, default=json.loads(environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


def model_fn(model_dir):
    """
    Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    model = joblib.load(path.join(model_dir, "mfg_maintenance_model.joblib"))
    return model

def input_fn(data, content_type):
    """
    input_fn handles data in JSON formats.
    Args:
        data: the request payload serialized in the content_type format
        content_type: the request content_type
    Returns: processed data
    """
    if content_type == 'application/json':
        # Read the raw input data as json.
        df = pd.read_json(data)
        if len(df.columns) == len(features) + 1:
            # labeled example
            df.columns = features + [labels]
        elif len(df.columns) == len(features):
            # Unlabeled example
            df.columns = features
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))

def predict_fn(data, model):
    """


    Args:
        data: input data for prediction by input_fn
        model: model loaded in memory by model_fn

    Returns: a prediction
        Return a two-dimensional NumPy array where the first row is predictions
        and the second row is prediction probability
    """
    predictions = model.predict(data)
    # get prediction probability
    pred_probs = model.predict_proba(data)
    # get max of prediction probability
    pred_probs = np.array(pred_probs.max(axis=1))
    return np.vstack((predictions, pred_probs))

def output_fn(prediction, content_type):
    print('Output_fn')

    if content_type == "application/json":
        # first row is prediction
        predictions = prediction[0]
        # second row is prediction probability
        pred_probs = prediction[1]

        instances = []
        for index, prediction in enumerate(predictions):
            instance = {}
            instance['prediction'] = prediction
            instance['pred_prob'] = pred_probs[index]
            instances.append(instance)
        # Serialize to JSON
        return json.dumps(instances)
    else:
        raise ValueError("{} content type is not supported by this script.".format(content_type))


if __name__ == '__main__':
    
    args, unknown = _parse_args()

    train_data, train_labels = _load_data(args.train,'train')
    eval_data, eval_labels = _load_data(args.test,'test')

    model = _model_(args, train_data, train_labels, eval_data, eval_labels)

    if args.current_host == args.hosts[0]:
        joblib.dump(model, path.join(args.model_dir, "mfg_maintenance_model.joblib"))