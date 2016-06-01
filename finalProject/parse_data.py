# Parse data from Reuters
import eikon as ek

ek.set_app_id('D79B714919385CD9D749D1')
import pandas as pd

from IPython import embed

def parse_data(start_date, end_date, *args):

    comm_list = args
    print comm_list
    list_dfs = []
    for k in comm_list:
        dic = ek.get_timeseries([k], start_date=start_date, end_date=end_date, interval='daily', fields=['CLOSE'])
        print "Finished ", k
        df = dic[k]  # put from dic into df
        df.columns = [k]  # rename columns to RICs code
        list_dfs += [df]

    # Outer join
    oj = pd.concat(list_dfs, axis=1, join='outer')

    # Do some data prep on missing values
    df = oj.copy()

    ## NaN indicator column
    # for k in df:
    #     s = 'NaN.' + k
    #     df[s] = df[k].isnull()

    # forward fill NaN's with previous value
    df = df.fillna(method='pad')

    # fill with zeroes what doesn't have a past value
    df = df.fillna(0)

    return df

# if __name__ == "__main__":
#     import sys
#     parse_data(list(sys.argv[1]))