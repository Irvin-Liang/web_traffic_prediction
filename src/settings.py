import pandas as pd

START_INDEX = 8
PREDICT_DATE = [str(t.date()) for t in pd.date_range('2017-01-01', periods=60)]
DATA_FILENAME = '../data/train_1_parsed.csv'