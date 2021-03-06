import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages, categories



def clean_data(messages, categories):
    """Combines the tabular data by associating messages with their respective categories.
    
    Args:
        messages (pd.DataFrame): DataFrame of the messages extracted from csv.
        categories (pd.DataFrame): DataFrame of the output categories also extracted from csv
    
    Returns:
        pd.DataFrame: Cleaned data containing the merged result.
    """
    categories = categories.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split('-').str[0].values
    categories.columns = category_colnames
    for column in categories:
        # set each value to be number following the '-'
        categories[column] = categories[column].str.split('-').str[1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

        # convert values to binary.
        categories[column] = np.where(categories[column] > 0, 1, 0)

    # we observe that the rows in the messages and categories map one-to-one
    # i.e, the n_th row of messages matches up with the n_th row of categories
    # but the ids are duplicated in both tables, which means merging on ids will give us fake datapoints
    # this is why we choose to do a simple concat rather than a merge
    df = pd.concat([messages, categories], axis=1)
    df = df.drop_duplicates()

    return df



def save_data(df, database_filename):
    """Save the given dataframe into sqlite database.
    
    Args:
        df (pd.DataFrame): DataFrame that is to be saved
        database_filename (TYPE): Name of database file in which to save the data.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()