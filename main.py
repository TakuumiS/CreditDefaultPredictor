import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():

    # Pandas options
    pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.colheader_justify', 'left')
    pd.set_option('display.precision', 4)


    # Read data
    df = pd.read_excel("./data/default of credit card clients.xls", skiprows = 1)
    target = df['default payment next month']

    # # EDA
    # unique_marriages = df['MARRIAGE'].drop_duplicates()
    # print("Marriage Values: ", sorted(unique_marriages))
    #
    # unique_sex = df['SEX'].drop_duplicates()
    # print("Sex Values: ", sorted(unique_sex))
    #
    # unique_education = df['EDUCATION'].drop_duplicates()
    # print("Education Values: ", sorted(unique_education))

    # Data Preprocessing
    df.drop(columns=['ID', 'default payment next month'], axis=1, inplace=True)

    # EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: Marital status (1=married, 2=single, 3=others)
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

    # Export preprocessed data
    df.to_csv("./data/preprocessed_data.csv", index=False)

    # Build and train model
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)
    print(f"Data split: {len(x_train)} train, {len(x_test)} test samples.\n")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    rf = RandomForestClassifier()
    rf.fit(x_train_scaled, y_train)
    y_pred = rf.predict(x_test_scaled)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Pre-Optimized Model Metrics:")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")







if __name__ == "__main__":
    main()