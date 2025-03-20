def transform_data(df):
    print("Transforming data...")
    df["new_feature"] = df["existing_feature"] * 2
    return df
