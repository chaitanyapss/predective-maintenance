#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import boto3

from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import sagemaker

# SageMaker SKLearn Estimator
from sagemaker.sklearn.estimator import SKLearn
from sklearn.metrics import mean_squared_error


# In[25]:


training_data_uri = 's3://sagemakerdemotest2/predictive_maintenance_final.csv'
testing_data_uri = 's3://sagemakerdemotest2/predictive_maintenance_final.csv'
model_path_uri = 's3://ie-auto/maintenance/datawrangler/model/'
bucket = 'sagemakerdemotest2'
prefix = 'model'
script_file = 'mfg_maintenance_model_generator.py'


# In[26]:


#!pygmentize 'vehicle_maintenance_model_generator.py'
s3 = boto3.client('s3')
#obj = s3.get_object(Bucket=bucket, Key=script_file)
#file_content = obj['Body'].read().decode('utf-8')

#print(file_content)


# In[27]:


instance_type='ml.m5.xlarge'


# In[28]:


try:
    sagemaker_role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    sagemaker_role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20231117T140215')['Role']['Arn']


# In[29]:


sess = sagemaker.Session()
sklearn_estimator = SKLearn(
    entry_point=script_file,  # Path to your script
    role= sagemaker_role,
    instance_count=1,
    instance_type= instance_type,
    framework_version='0.23-1',  # Specify the scikit-learn version
    base_job_name='asset-maintenance',
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    sagemaker_session=sess
)


# In[30]:


sklearn_estimator.fit({'training':training_data_uri,
               'testing':testing_data_uri})


# In[31]:


sklearn_estimator.latest_training_job.job_name


# In[32]:


sklearn_estimator.model_data


# In[33]:


predictor = sklearn_estimator.deploy(initial_instance_count=1, instance_type=instance_type)


# In[ ]:




