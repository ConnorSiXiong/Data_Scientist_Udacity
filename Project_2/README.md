# Udacity-Data-Scientist

## Project 2


## Disaster Response Pipeline Project

In this project, the disaster data collected from Figure Eight was processed and analysed aiming at classifying messages for response and relief agencies.

These data were preprocessed into different categories and were saved in the disaster_categories.csv, so it is not necessary to apply NLP.
#

## Libraries/Dependencies

1. Install the dependencies first

- ```pip install -r requirements.txt```

2. Run the ETL pipleline

- ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```

  The process data will be saved in ```data/DisasterResponse.db```

3. Machine Learning Inference

- ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```

4. Launch the app
- ```python run.py```

#

# Structure
```
project_2
│   README.md
│   requirements.txt
│
└───app
│   │   run.py  # Flask file that runs app
│   │───templates
│       │   go.html  # classification result page of web app
│       │   master.html  # classification result page of web app
│   
└───data
│    │   process_data.py
│    │   disaster_categories.csv
│    │   disaster_messages.csv
│    │   DisasterResponse.db (sample, not upload)  # database to save clean data to
│   
└───models
│    │   train_classifier.py
│    │   classifier.pkl (sample, not upload)  # saved model
```

#
## Note:
The file paths may be different in different operation system, you may need to modify it manually.