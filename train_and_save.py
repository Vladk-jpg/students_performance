"""Тренирует RandomForest на onti-students-performance и сохраняет артефакты для UI."""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = Path(__file__).parent / "onti-students-performance"
ART_DIR = Path(__file__).parent / "artifacts"
ART_DIR.mkdir(exist_ok=True)

CAT_COLS = [
    "НАПРАВЛЕНИЕ", "ГОД", "АТТЕСТАЦИЯ", "ДИСЦИПЛИНА",
    "Пол", "Статус", "Категория обучения", "Форма обучения",
    "Образование", "Что именно закончил",
]
NUM_COLS = ["КУРС", "СЕМЕСТР", "number1", "number2"]
FEATURES = CAT_COLS + NUM_COLS


def load_studs_info():
    df = pd.read_csv(DATA_DIR / "studs_info.csv").rename(
        columns={"   ": "number2", "   number": "number1"}
    )
    df = df.drop(columns=["Дата выпуска", "Шифр", "направление (специальность)", "Дата выдачи"])
    df = df.drop_duplicates(["STD_ID"])

    df["number1"] = df["number1"].fillna(df["number1"].mean())
    df["number2"] = df["number2"].fillna(df["number2"].mean())
    for c in ["Пол", "Статус", "Категория обучения", "Форма обучения", "Образование", "Что именно закончил"]:
        df[c] = df[c].fillna(df[c].value_counts().keys()[0])
    return df


def build_dataset():
    X = pd.read_csv(DATA_DIR / "X_train.csv").drop(columns=["Unnamed: 0"])
    y = pd.read_csv(DATA_DIR / "y_train.csv")["mark"]

    studs = load_studs_info()
    merged = X.merge(studs, how="left", on="STD_ID")
    mask = merged.notna().all(axis=1)
    merged = merged[mask].reset_index(drop=True)
    y = y[mask.values].reset_index(drop=True)
    return merged[FEATURES], y, studs


def main():
    X, y, studs = build_dataset()
    print(f"Dataset: {X.shape}, target: {y.shape}")

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_enc = X.copy()
    X_enc[CAT_COLS] = encoder.fit_transform(X[CAT_COLS])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_enc, y, test_size=0.1, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_leaf=3,
        n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_valid).clip(0, 5)
    print(f"R²:  {r2_score(y_valid, pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_valid, pred):.4f}")

    categories = {col: list(cats) for col, cats in zip(CAT_COLS, encoder.categories_)}

    joblib.dump(model, ART_DIR / "model.pkl")
    joblib.dump(encoder, ART_DIR / "encoder.pkl")
    joblib.dump({
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "features": FEATURES,
        "categories": categories,
        "studs_info": studs.to_dict(orient="records"),
    }, ART_DIR / "meta.pkl")
    print(f"Saved artifacts to {ART_DIR}")


if __name__ == "__main__":
    main()
