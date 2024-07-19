import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_data():
    housing_train = pd.read_csv('resource/asnlib/publicdata/dev/train_data1.csv')
    labels_train = pd.read_csv('resource/asnlib/publicdata/dev/train_label1.csv')
    housing_test = pd.read_csv('resource/asnlib/publicdata/dev/test_data1.csv')
    labels_test = pd.read_csv('resource/asnlib/publicdata/dev/test_label1.csv')

    # print(housing_train.shape)
    # print(housing_test.shape)
    # print(housing_train.head())
    # print(housing_test.head())
    # print(housing_train.describe())
    # print(housing_train.columns)
    # print(np.bincount(np.unique(labels_train)).sum())
    print(np.unique(housing_train.TYPE))
    print((housing_test.TYPE == 'Mobile house for sale').sum())
    return housing_train, labels_train, housing_test, labels_test
    # read_data()


def read_data():
    housing_train = pd.read_csv('resource/asnlib/publicdata/dev/train_data1.csv')
    labels_train = pd.read_csv('resource/asnlib/publicdata/dev/train_label1.csv')
    housing_test = pd.read_csv('resource/asnlib/publicdata/dev/test_data1.csv')
    labels_test = pd.read_csv('resource/asnlib/publicdata/dev/test_label1.csv')
    return housing_train, labels_train, housing_test, labels_test


def preprocess_data(housing_train, housing_test, labels_train, labels_test):
    drop_columns = ['LONGITUDE', 'LATITUDE', 'FORMATTED_ADDRESS', 'LONG_NAME', 'STREET_NAME',
                    'ADMINISTRATIVE_AREA_LEVEL_2', 'MAIN_ADDRESS', 'STATE', 'ADDRESS', 'BROKERTITLE', 'LOCALITY',
                    'SUBLOCALITY']

    combined_train_data = pd.concat([housing_train, labels_train.reset_index(drop=True)], axis=1)
    combined_test_data = pd.concat([housing_test, labels_test.reset_index(drop=True)], axis=1)

    combined_train_data = combined_train_data[combined_train_data['TYPE'] != 'Pending']
    combined_test_data = combined_test_data[combined_test_data['TYPE'] != 'Pending']

    combined_train_data.reset_index(drop=True, inplace=True)
    combined_test_data.reset_index(drop=True, inplace=True)

    combined_train_data = combined_train_data.drop(columns=drop_columns)
    combined_test_data = combined_test_data.drop(columns=drop_columns)
    # combined_train_data = combined_train_data.dropna()
    # combined_test_data = combined_test_data.dropna()
    numeric_columns = combined_train_data.select_dtypes(include=[np.number]).columns
    combined_train_data[numeric_columns] = combined_train_data[numeric_columns].fillna(
        combined_train_data[numeric_columns].median())
    combined_test_data[numeric_columns] = combined_test_data[numeric_columns].fillna(
        combined_train_data[numeric_columns].median())

    # combined_train_data = pd.get_dummies(combined_train_data, columns=['TYPE'], drop_first=True, dummy_na=False)
    # combined_test_data = pd.get_dummies(combined_test_data, columns=['TYPE'], drop_first=True, dummy_na=False)
    combined_train_data = pd.get_dummies(combined_train_data, columns=['TYPE'], drop_first=True)
    combined_test_data = pd.get_dummies(combined_test_data, columns=['TYPE'], drop_first=True)

    combined_train_data, combined_test_data = combined_train_data.align(combined_test_data, join='outer', axis=1,
                                                                        fill_value=0)

    labels_train = combined_train_data['BEDS'].astype(int)
    labels_test = combined_test_data['BEDS'].astype(int)
    housing_train = combined_train_data.drop('BEDS', axis=1)
    housing_test = combined_test_data.drop('BEDS', axis=1)

    return housing_train, labels_train, housing_test, labels_test


def encode_labels(y_train, y_val, y_test):
    all_labels = np.concatenate([y_train, y_val, y_test])
    unique_classes = np.unique(all_labels)
    unique_classes.sort()
    class_to_index = {original: encoded for encoded, original in enumerate(unique_classes)}

    # Label encoding
    y_train_encoded = np.array([class_to_index[label] for label in y_train])
    y_val_encoded = np.array([class_to_index[label] for label in y_val])
    y_test_encoded = np.array([class_to_index[label] for label in y_test])

    # One-hot encoding
    def one_hot_encode(labels, num_classes):
        num_samples = len(labels)
        onehot_labels = np.zeros((num_samples, num_classes))
        for i, label in enumerate(labels):
            onehot_labels[i, label] = 1
        return onehot_labels

    num_classes = len(unique_classes)
    y_train_onehot = one_hot_encode(y_train_encoded, num_classes)
    y_val_onehot = one_hot_encode(y_val_encoded, num_classes)
    y_test_onehot = one_hot_encode(y_test_encoded, num_classes)

    return y_train_onehot, y_val_onehot, y_test_onehot, num_classes


def split_data(X, y, validation_split=0.2):
    split_index = int(len(X) * (1 - validation_split))
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_val = X[split_index:]
    y_val = y[split_index:]

    return X_train, y_train, X_val, y_val


def convert_bools_to_floats(X):
    return X.astype(float)


def standardize_features(X, mean=None, std=None):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_replaced = np.where(std == 0, 1, std)
    X_standardized = (X - mean) / std_replaced
    return X_standardized, mean, std


def load_and_preprocess_data():
    housing_train, labels_train, housing_test, labels_test = read_data()
    housing_train, labels_train, housing_test, labels_test = preprocess_data(housing_train, housing_test, labels_train,
                                                                             labels_test)

    X_train = housing_train.to_numpy()
    y_train = labels_train.to_numpy().squeeze()
    X_test = housing_test.to_numpy()
    y_test = labels_test.to_numpy().squeeze()

    X_train = convert_bools_to_floats(X_train)
    X_test = convert_bools_to_floats(X_test)

    X_train, train_mean, train_std = standardize_features(X_train)
    X_test = (X_test - train_mean) / np.where(train_std == 0, 1, train_std)

    y_train_encoded, y_val_encoded, y_test_encoded, num_classes = encode_labels(y_train, y_train, y_test)

    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train_encoded = y_train_encoded[indices]

    X_train, y_train_encoded, X_val, y_val_encoded = split_data(X_train, y_train_encoded)

    return X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded, num_classes


def graph_type_street_name_correlations(housing_train):
    counter=housing_train['TYPE'].value_counts()
    plt.bar(counter.index,counter,color='red')
    plt.xticks(rotation=90)

    plt.xlabel('type of realty')
    plt.ylabel('counting')
    plt.title('number of the realty count')

    plt.show()

    counter=housing_train['STREET_NAME'].value_counts()
    plt.bar(counter.index[:20],counter[:20],color='red')
    plt.xticks(rotation=90)

    plt.xlabel('type of realty')
    plt.ylabel('counting')
    plt.title('number of the realty count')

    plt.show()
