import pandas as pd

def interpolate_dates(df: pd.DataFrame, x: str, y: list[str], interval = 'D', method = 'linear'):
    final_df = pd.DataFrame()
    for column in y:
        idf = pd.DataFrame(df[[x, column]]).set_index([x])
        interpolated = idf.resample(interval).interpolate(method = method)
        final_df = pd.concat([final_df, interpolated], axis=1)
        # print(interpolated)
    final_df[x] = final_df.index
    final_df.index = range(len(final_df))

    return final_df