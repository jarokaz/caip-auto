# Lint as: python2, python3
"""Preprocess code to be exported.

Usage:
This code snippet can be used in two ways:
1. Command line preprocessing.
  You can call instance_generator.py from the command line with three arguments:
    1) raw_data_string: this argument is mandatory. It will be a line of data
                        represented as a comma-separated string.
    2) metadata: this argument is optional. It will be the full file path of
                  the 'metadata.json' file. If not specified, the program will
                  look for a 'metadata.json' file in the same directory as this
                  code snippet.
  The exact command will be:

    python instance_generator.py --raw_data_string <RAW_DATA_STRING>
                                 --metadata <META_DATA>

2. Use the CMLEPreProcessor module in your own code.
  This approach gives you the control of processing as many data points as you
  like instead of just one line of data. To use the CMLEPreProcessor module,
  see the example below:

  import json
  from instance_generator import CMLEPreProcessor

  # Blank between two commas stands for missing data.
  # Number stands for numerical data. Strings stand for categorical data.
  raw_data_string = '1,Google,,CMLE,,Codeless,Rocks'
  preprocessor = CMLEPreProcessor()
  processed_data_list = preprocessor.transform_string_instance(raw_data_string)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher


class CMLEPreProcessor(object):
  """A preprocessor class to preprocess the data at prediction time.
  """

  def __init__(self, metadata_file=None):
    """Initializes a CMLEDataPreprocessor with meta data.

    The metadata will be set to a dictionary with the following format:
    self._metadata = {
      'target_algorithm':'XGboost',
      'target_column: {
        'type': 'regression'
      },
      feature_columns: {
        'col_1': {
          'type': 'numerical',
          'statistics: {
            'mean': 0.0,
            'variance': 1.0,
          }
        },
        col_2: {
          'type': 'categorical',
          'treatment': 'vocabulary',
          'num_category': 3,
          'mapping':{
            'cat_a': 0,
            'cat_b': 1,
            'cat_c': 2
          }
        }
      }
    }

    Args:
      metadata_file: full file path of the meta data.
    """
    if not metadata_file:
      current_filepath = os.path.realpath(__file__)
      metadata_file = os.path.join(
          os.path.dirname(current_filepath), 'metadata.json')
    with open(metadata_file, 'r') as file_handler:
      self._metadata = json.load(file_handler)

  def _is_number(self, string_value):
    """Check if string_value is a number.

    Args:
      string_value: input string value/
    Returns:
      True if  string_value is a number, False otherwise.
    """

    is_number = True
    try:
      float(string_value)
    except ValueError:
      is_number = False
    return is_number

  def _impute_missing_values(self, data_column, column_mapping):
    """Imputes missing values for numeric columns.

    Numeric: imputes with mean value.
    Categorical: leaves it intact and sets nan to all zeros while encoding.
    Args:
      data_column: a pandas series representing a DataFrame column
      column_mapping: a dictionary of the column mapping.

    Returns:
      data_column: a pandas series with the missing values imputed.
    """
    if column_mapping['type'] == 'numerical':
      if data_column.isnull().values.any():
        mean = column_mapping['statistics']['mean']
        data_column.fillna(mean, inplace=True)
    else:
      # If feature is categorical, do nothing
      pass

    return data_column

  def _onehot_encode(self, data_column, category_mapping):
    """One-hot encode a non-numeric feature.

    Args:
      data_column: a pandas series representing a DataFrame column
      category_mapping: dictionary of category index mapping

    Returns:
      encoded_columns: A pandas DataFrame of
      the one-hot encoding columns
    """
    categories = list(category_mapping.keys())
    categories.sort()
    one_hot_matrix = np.zeros((len(data_column), len(categories)), dtype=np.int)

    for row_index in range(len(data_column)):
      category = str(data_column[row_index]).strip()
      if category in category_mapping:
        column_index = category_mapping[category]
        one_hot_matrix[row_index, column_index] = 1

    encoded_columns = pd.DataFrame(one_hot_matrix, columns=categories)
    return encoded_columns

  def _hash(self, data_column, num_hash_features):
    """Convert a categorical feature to numerical representation using hashing.

    Args:
      data_column: a pandas series representing a DataFrame column
      num_hash_features: the number of hashing features

    Returns:
      hash_columns: a pandas DataFrame of the hashed columns
    """
    hasher = FeatureHasher(n_features=num_hash_features, input_type='string')
    data_column = data_column.fillna('null')
    hashed_matrix = hasher.transform(data_column).toarray()
    hash_columns = pd.DataFrame(hashed_matrix)
    return hash_columns

  def _transform_data(self, data_frame):
    """Transform the data given a mapping.

    Numerical features:  Imputing missing values with mean.
    Categorical features: One_hot encoding if not a TensorFlow estimator
    Target feature: one_hot encoding if a classification and
    if not a TensorFlow estimator

    Args:
      data_frame: a pandas DataFrame of training, evaluation, or test data

    Returns:
      transformed_data_frame: a pandas DataFrame after applying transformations
    """
    mapping = self._metadata
    feature_mapping = mapping['feature_columns']

    transformed_data_frame = pd.DataFrame()

    # Process feature columns
    for column in data_frame.columns:
      data_column = data_frame[column]
      if feature_mapping[column]['type'] == 'numerical':
        # Impute missing values
        data_column = self._impute_missing_values(data_column,
                                                  feature_mapping[column])
        transformed_data_frame[column] = data_column.astype(float)
      else:
        if feature_mapping[column]['treatment'] in ['vocabulary', 'identity']:
          # One-hot encode
          encoded_columns = self._onehot_encode(
              data_column, feature_mapping[column]['mapping'])
          for col in encoded_columns.columns:
            col_name = '{}_{}'.format(column, col)
            transformed_data_frame[col_name] = encoded_columns[col]
        elif feature_mapping[column]['treatment'] == 'hashing':
          # Process with FeatureHasher
          hashed_columns = self._hash(data_column,
                                      feature_mapping[column]['hash_buckets'])
          for col in hashed_columns.columns:
            col_name = '{}_{}'.format(column, col)
            transformed_data_frame[col_name] = hashed_columns[col]
        else:
          # Ignore row_key and constant features
          pass

    return transformed_data_frame

  def transform_list_instance(self, raw_data_list):
    """Transforms the raw data list according to metadata.

    Args:
      raw_data_list: a single data point represented as a list.

    Returns:
      A list of processed data where all the entries will be numerical.

    """
    header = list(self._metadata['feature_columns'].keys())
    header.sort()
    data_frame = pd.DataFrame([raw_data_list], columns=header)
    transformed_data_frame = self._transform_data(data_frame)

    return list(transformed_data_frame.values[0])

  def transform_string_instance(self, raw_data_str):
    """Transforms the raw data string according to metadata.

    Args:
      raw_data_str: a single data point represented as a string and separeted by
        comma(',').

    Raises:
      IndexError: In the given raw_data_list, if the number of features does not
                  match the number of features in the training data.
      ValueError: If the data type of raw feature does not match the data type
                  of the corresponding column in the training data.

    Returns:
      A list of processed features.
    """

    raw_data_list = [value.strip() for value in raw_data_str.split(',')]

    if len(self._metadata['feature_columns']) != len(raw_data_list):
      raise IndexError('Make sure that your input data schema matches '
                       'the metadata provided. Do not include a column '
                       'for the target feature in your input data.')

    columns = list(self._metadata['feature_columns'].keys())
    columns.sort()

    for i in range(len(raw_data_list)):
      if not raw_data_list[i]:
        raw_data_list[i] = None
        continue

      raw_feature = raw_data_list[i]
      column = columns[i]
      column_type = self._metadata['feature_columns'][column]['type']

      if column_type == 'numerical':
        if not self._is_number(raw_feature):
          raise ValueError('Expecting numerical value in column %d but '
                           'categorical value is present.' % i)

      elif column_type == 'categorical':
        treatment = self._metadata['feature_columns'][column]['treatment']
        if raw_feature.isdigit() and treatment == 'vocabulary':
          raise ValueError('Expecting categorical value in column %d but '
                           'numerical value is present.' % i)
        if not self._is_number(raw_feature) and treatment == 'identity':
          raise ValueError('Expecting numerical value in column %d but '
                           'categorical value is present.' % i)

    return self.transform_list_instance(raw_data_list)


def get_params():
  """Gets the parameter from command line.

  Returns:
    A dictionary consists of raw data string and meta data file path.
  """
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument(
      '--raw_data_string',
      type=str,
      required=True,
      help=('A single data point represented as a string and separated by '
            'comma. For example it should have the following format:\n'
            'raw_data_string = \'Google,CMLE,4,all\'\n'
            'In this case, column 1, 2 and 4 are categorical and column 3 is '
            'numerical'))
  parser.add_argument(
      '--metadata',
      type=str,
      default=None,
      help=('Path to the meta data json file. If not specified, then it will '
            'look for \'meta-data.json\' under the same directory as '
            '\'preprocess.py\''))

  args = parser.parse_args()
  return vars(args)


def main():
  """Main routine to preprocess a data point.

  Prints the processed data to the console.
  """
  params = get_params()
  pre_processor = CMLEPreProcessor(params['metadata'])
  print(pre_processor.transform_string_instance(params['raw_data_string']))


if __name__ == '__main__':
  main()
