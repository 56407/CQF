# Parse data from Reuters

def parse_data(start_date, end_date, comm_list):

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

if __name__ == "__main__":
    # import sys
    # parse_data(list(sys.argv[1]))

    from datetime import datetime
    import eikon as ek

    ek.set_app_id('D79B714919385CD9D749D1')
    import pandas as pd

    start_date = datetime(2009, 1, 1)
    end_date = datetime(2016, 5, 31)
    df2 = parse_data(start_date, end_date, ["Cc1", "Sc1", "Wc1"])
