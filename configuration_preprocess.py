from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

# This def is used to preprocess the configuration data before it is passed to the model
# The configuration data is preprocessed using the ColumnTransformer class from sklearn
# The configuration data is split into numerical and categorical data
# The numerical data is scaled using the StandardScaler class from sklearn
# The categorical data is encoded using the OneHotEncoder class from sklearn
def configuration_preprocess_before_model_training(configurations):
        
    df = pd.DataFrame(configurations)
    categorical_config = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    numerical_config = df.select_dtypes(exclude=['object', 'bool']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_config),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_config)
        ]
    )
    
    return preprocessor

# Every time when sampling configurations, it is reccomended to call this function
def configuration_preprocess_before_sampling(config_space, theta):
    theta.update({key: config_space.get_hyperparameter(key).default_value 
                  for key in config_space if key not in theta})
    return theta



