import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    ## Loading each csv file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    print('Messages: {}'.format(messages.shape))
    print('Categories: {}'.format(categories.shape))

    ## Split and expand categories
    categories = categories['categories'].str.split(pat=';', expand=True)

    row = categories.iloc[0, :]
    category_columns = row.map(lambda x: x.split('-')[0])
    categories.columns = category_columns

    ## Convert category values to just numeric 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    ## Remain only 0 and 1
    categories.replace(2, 1, inplace=True)

    ## Concat two things and return
    result = pd.concat([messages, categories], axis=1)
    print('Merged data: {}'.format(result.shape))

    return result
    
        
    
def clean_data(df):
    ## Drop duplicates          
    df_cleaned = df.drop_duplicates()

    return df_cleaned
          
          
def save_data(df, database_filename):
    """Save the DataFrame data into the database"""
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response_df', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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