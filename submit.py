import pandas as pd

def submit(y_predict, model_name):
    df = pd.read_csv('./input/sample_submission.csv')
    for i, c in enumerate(df.columns[1:]):
        df[c] = y_predict[:, i]
    # save to file
    df.to_csv('./submission/{}.csv'.format(model_name), index=None)