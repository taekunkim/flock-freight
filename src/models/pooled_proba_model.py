def generate_probability_pipeline(df):
    if not [col in df.cols for col in df_X].mean() == 1: AssertionError("DataFrame does not contain required columns")

    pipeline = ...

    return pipeline