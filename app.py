import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

ART_DIR = Path(__file__).parent / "artifacts"


@st.cache_resource
def load_artifacts():
    meta = joblib.load(ART_DIR / "meta.pkl")
    if meta.get("model_type") == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(ART_DIR / "catboost_model.cbm"))
        encoder = None
    else:
        model = joblib.load(ART_DIR / "model.pkl")
        encoder = joblib.load(ART_DIR / "encoder.pkl")
    return model, encoder, meta


st.set_page_config(page_title="Прогноз оценки студента", page_icon="🎓", layout="centered")
st.title("Прогноз оценки студента")

try:
    model, encoder, meta = load_artifacts()
except FileNotFoundError:
    st.error("Не найдены артефакты")
    st.stop()

_r2 = meta.get("metrics", {}).get("r2_score")
_cap = meta.get("best_name", "model") + (f" — R²={_r2:.3f} (honest holdout)" if _r2 is not None else "")
st.caption(_cap)

cats = meta["categories"]
cat_cols = meta["cat_cols"]
features = meta["features"]
history_cols = meta.get("history_cols", [])
global_mean = meta.get("global_mean", 4.0)
stud_full = meta.get("stud_full", {})
stud_count_full = meta.get("stud_count", {})
disc_full = meta.get("disc_full", {})
dir_full = meta.get("dir_full", {})

LABELS = {
    "Пол": {"М": "Мужской", "Ж": "Женский"},
    "Статус": {"СТ": "Студент", "ВЫП": "Выпускник"},
    "Форма обучения": {"О": "Очная", "З": "Заочная", "В": "Вечерняя"},
    "Категория обучения": {"Б": "Бюджет", "П": "Платно", "БП": "Бюджет/Платно"},
    "Образование": {"НВ": "Неполное высшее", "В": "Высшее", "С": "Среднее", "СС": "Среднее специальное"},
}


def label(col: str, value):
    return LABELS.get(col, {}).get(value, value)


def select_labeled(col: str, options, **kwargs):
    return st.selectbox(col, options, format_func=lambda v: label(col, v), **kwargs)


studs_records = meta["studs_info"]
studs_by_id = {r["STD_ID"]: r for r in studs_records}

st.subheader("Студент")
mode = st.radio(
    "Источник данных о студенте",
    ["По STD_ID из справочника", "Ввести вручную"],
    horizontal=True,
)

stud_data = {}
chosen_id = None
n1 = 0.0
n2 = 0.0
if mode == "По STD_ID из справочника":
    ids = sorted(studs_by_id.keys())
    chosen_id = st.selectbox("STD_ID", ids, index=0)
    stud_data = studs_by_id[chosen_id]
    n1 = float(stud_data.get("number1", 0.0) or 0.0)
    n2 = float(stud_data.get("number2", 0.0) or 0.0)
    cols = st.columns(3)
    cols[0].metric("Пол", label("Пол", stud_data["Пол"]))
    cols[1].metric("Статус", label("Статус", stud_data["Статус"]))
    cols[2].metric("Форма обучения", label("Форма обучения", stud_data["Форма обучения"]))
    if chosen_id in stud_full:
        c1, c2 = st.columns(2)
        c1.metric("Средняя оценка студента", f"{stud_full[chosen_id]:.2f}")
        c2.metric("Записей в истории", int(stud_count_full.get(chosen_id, 0)))
        st.caption("warm-start: модель использует историю студента")
    else:
        st.caption("cold-start: студент не встречался в train, история = глобальное среднее")
else:
    c1, c2 = st.columns(2)
    stud_data["Пол"] = select_labeled("Пол", cats["Пол"])
    stud_data["Статус"] = select_labeled("Статус", cats["Статус"])
    stud_data["Категория обучения"] = select_labeled("Категория обучения", cats["Категория обучения"])
    stud_data["Форма обучения"] = select_labeled("Форма обучения", cats["Форма обучения"])
    stud_data["Образование"] = select_labeled("Образование", cats["Образование"])
    stud_data["Что именно закончил"] = c2.selectbox("Что именно закончил", cats["Что именно закончил"])
    n1 = float(c1.number_input("number1", value=0.0, step=1.0))
    n2 = float(c2.number_input("number2", value=0.0, step=1.0))
    st.caption("cold-start: для нового студента история = глобальное среднее")

st.subheader("Курс / дисциплина")
c1, c2 = st.columns(2)
direction = c1.selectbox("НАПРАВЛЕНИЕ", cats["НАПРАВЛЕНИЕ"])
year = c2.selectbox("ГОД", cats["ГОД"])
attestation = c1.selectbox("АТТЕСТАЦИЯ", cats["АТТЕСТАЦИЯ"])
discipline = c2.selectbox("ДИСЦИПЛИНА", cats["ДИСЦИПЛИНА"])
course = c1.slider("КУРС", 1, 6, 2)
semester = c2.slider("СЕМЕСТР", 1, 12, 3)

if st.button("Предсказать оценку", type="primary", use_container_width=True):
    row = {
        "НАПРАВЛЕНИЕ": direction,
        "ГОД": year,
        "АТТЕСТАЦИЯ": attestation,
        "ДИСЦИПЛИНА": discipline,
        "Пол": stud_data["Пол"],
        "Статус": stud_data["Статус"],
        "Категория обучения": stud_data["Категория обучения"],
        "Форма обучения": stud_data["Форма обучения"],
        "Образование": stud_data["Образование"],
        "Что именно закончил": stud_data["Что именно закончил"],
        "КУРС": float(course),
        "СЕМЕСТР": float(semester),
        "number1": float(n1),
        "number2": float(n2),
        "stud_mean_grade": float(stud_full.get(chosen_id, global_mean)) if chosen_id else global_mean,
        "stud_count": float(stud_count_full.get(chosen_id, 0)) if chosen_id else 0.0,
        "discipline_mean_grade": float(disc_full.get(discipline, global_mean)),
        "direction_mean_grade": float(dir_full.get(direction, global_mean)),
    }
    df = pd.DataFrame([row])[features]
    if meta.get("model_type") == "catboost":
        pred = float(model.predict(df)[0])
    else:
        df[cat_cols] = encoder.transform(df[cat_cols])
        pred = float(model.predict(df)[0])
    pred_clipped = max(0.0, min(5.0, pred))

    st.success(f"### Прогноз: **{pred_clipped:.2f}** / 5")
    rounded = round(pred_clipped)

    with st.expander("Детали входа модели"):
        st.json(row)

st.caption(f"{_cap}")
