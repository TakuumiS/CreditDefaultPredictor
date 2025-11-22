import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def main():

    # Pandas options
    pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
    pd.options.display.max_columns = None

    # Read data
    df = pd.read_excel("./data/default of credit card clients.xls", skiprows = 1)
    target = df['default payment next month']

    # EDA
    unique_marriages = df['MARRIAGE'].drop_duplicates()
    print("Marriage Values: ", sorted(unique_marriages))

    unique_sex = df['SEX'].drop_duplicates()
    print("Sex Values: ", sorted(unique_sex))

    unique_education = df['EDUCATION'].drop_duplicates()
    print("Education Values: ", sorted(unique_education))

    # Data Preprocessing
    df.drop(columns=['ID', 'default payment next month'], axis=1, inplace=True)

    # EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: Marital status (1=married, 2=single, 3=others)
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

    # Scale values
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df.to_csv("./data/preprocessed_data.csv", index=False)



if __name__ == "__main__":
    main()