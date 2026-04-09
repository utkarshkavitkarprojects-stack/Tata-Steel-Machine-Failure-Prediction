import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def preprocess_data(df, target_col='Machine failure', test_size=0.2, random_state=42):
    """
    Preprocesses the dataset:
    - handles missing values and duplicates
    - encodes categorical variables
    - splits train/test
    - applies SMOTE on training data only

    Parameters:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        test_size (float): Test split ratio
        random_state (int): Random seed

    Returns:
        X_train_resampled, X_test, y_train_resampled, y_test, feature_names
    """

    df = df.copy()

    print(f"✅ Loaded data with shape: {df.shape}")

    # -----------------------------
    # Handle missing values
    # -----------------------------
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df = df.dropna()
        print("🧹 Dropped rows with missing values")
    else:
        print("✅ No missing values found")

    # -----------------------------
    # Remove duplicate rows
    # -----------------------------
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"🧹 Dropped duplicate rows: {duplicate_count}")
    else:
        print("✅ No duplicates found")

    # -----------------------------
    # Separate features and target
    # -----------------------------
    X = df.drop(columns=["Machine_failure"])
    y = df["Machine_failure"]

    # -----------------------------
    # Encode categorical variables
    # -----------------------------
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print(f"🔠 Encoded categorical columns: {categorical_cols}")
    else:
        print("✅ No categorical columns to encode")

    feature_names = X.columns.tolist()

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    print(f"📊 X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"📊 y_train distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"📊 y_test distribution:\n{y_test.value_counts(normalize=True)}")

    # -----------------------------
    # Apply SMOTE only on training data
    # -----------------------------
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("⚖️ Applied SMOTE on training data")
    print(f"📊 X_train after SMOTE: {X_train_resampled.shape}")
    print(f"📊 y_train after SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")

    return X_train_resampled, X_test, y_train_resampled, y_test