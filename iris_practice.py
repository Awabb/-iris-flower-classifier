import pandas as pd
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    # 1. Load the flowers and get a practice table with fake mistakes.
    iris = load_iris(as_frame=True)
    records = make_messy(iris.data)
    records["flower_type"] = iris.target
    records = pd.concat([records, records.head()], ignore_index=True)
    print("Messy records (first five)")
    print(records.head().to_string(index=False))

    # 2. Remove repeated records and clean up the measurement columns.
    unique_records = records.drop_duplicates()
    removed_count = len(records) - len(unique_records)
    print("\nRepeated records removed:", removed_count)

    measurements = unique_records.drop(columns="flower_type")
    features = clean_measurements(measurements)
    labels = unique_records["flower_type"]
    print("\nMissing measurements after parsing")
    print(features.isna().sum().to_string())

    # 3. Split the clues and answers into a study set and a quiz set.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    print("\nFlowers for training:", len(x_train))
    print("Flowers for testing:", len(x_test))

    # 4. Learn the middle values from the study set and reuse them on the quiz.
    training_medians = x_train.median()
    x_train = x_train.fillna(training_medians)
    x_test = x_test.fillna(training_medians)

    # 5. Compare a simple guess with a decision tree on the same quiz.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_accuracy = baseline.score(x_test, y_test)
    print(f"Most-common-species baseline: {baseline_accuracy:.1%}")

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    tree_accuracy = accuracy_score(y_test, predictions)
    print(f"Decision tree accuracy: {tree_accuracy:.1%}")

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    results = pd.DataFrame(matrix, index=iris.target_names, columns=iris.target_names)
    print("\nConfusion matrix (rows = actual, columns = predicted)\n", results)

    # 6. Prepare one new flower the same way before asking for a prediction.
    new_measurements = ["6.1 cm", "", "4.7", "1.4"]
    new_flower = pd.DataFrame([new_measurements], columns=features.columns)
    new_flower = clean_measurements(new_flower)
    new_flower = new_flower.fillna(training_medians)
    new_predictions = model.predict(new_flower)
    species = iris.target_names[new_predictions[0]]
    print("\nExample measurements:", new_measurements)
    print("Predicted species:", species)


def make_messy(measurements):
    """Add fake data-entry mistakes to a copy of the measurements."""
    messy = measurements.astype(str)
    # range(start, stop, step) picks rows at regular intervals.
    for row in range(0, len(messy), 9):
        messy.loc[row, "sepal length (cm)"] = ""
    for row in range(1, len(messy), 10):
        width = messy.loc[row, "sepal width (cm)"]
        messy.loc[row, "sepal width (cm)"] = width + " cm "
    for row in range(2, len(messy), 11):
        messy.loc[row, "petal length (cm)"] = "oops"
    for row in range(3, len(messy), 12):
        messy.loc[row, "petal width (cm)"] = "-1"
    return messy


def clean_measurements(measurements):
    """Read centimeters and treat blanks or invalid measurements as missing."""
    cleaned = measurements.copy()
    for column in cleaned.columns:
        values = cleaned[column].str.strip()
        values = values.str.removesuffix("cm")
        # errors="coerce" makes anything that can't become a number missing.
        numbers = pd.to_numeric(values, errors="coerce")
        cleaned[column] = numbers.where(numbers > 0)
    return cleaned


if __name__ == "__main__":
    main()
