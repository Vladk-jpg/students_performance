"""Streamlit UI для предсказания оценки студента.

Запуск:  streamlit run app.py
"""
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
st.title("🎓 Прогноз оценки студента")

try:
    model, encoder, meta = load_artifacts()
except FileNotFoundError:
    st.error("Не найдены артефакты. Сначала прогоните `result-0-41.ipynb` Restart & Run All.")
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
if mode == "По STD_ID из справочника":
    ids = sorted(studs_by_id.keys())
    chosen_id = st.selectbox("STD_ID", ids, index=0)
    stud_data = studs_by_id[chosen_id]
    cols = st.columns(3)
    cols[0].metric("Пол", stud_data["Пол"])
    cols[1].metric("Статус", stud_data["Статус"])
    cols[2].metric("Форма обучения", stud_data["Форма обучения"])
    if chosen_id in stud_full:
        c1, c2 = st.columns(2)
        c1.metric("Средняя оценка студента", f"{stud_full[chosen_id]:.2f}")
        c2.metric("Записей в истории", int(stud_count_full.get(chosen_id, 0)))
        st.caption("✅ warm-start: модель использует историю студента")
    else:
        st.caption("❄️ cold-start: студент не встречался в train, история = глобальное среднее")
else:
    c1, c2 = st.columns(2)
    stud_data["Пол"] = c1.selectbox("Пол", cats["Пол"])
    stud_data["Статус"] = c2.selectbox("Статус", cats["Статус"])
    stud_data["Категория обучения"] = c1.selectbox("Категория обучения", cats["Категория обучения"])
    stud_data["Форма обучения"] = c2.selectbox("Форма обучения", cats["Форма обучения"])
    stud_data["Образование"] = c1.selectbox("Образование", cats["Образование"])
    stud_data["Что именно закончил"] = c2.selectbox("Что именно закончил", cats["Что именно закончил"])
    st.caption("❄️ cold-start: для нового студента история = глобальное среднее")

st.subheader("Курс / дисциплина")
c1, c2 = st.columns(2)
direction = c1.selectbox("НАПРАВЛЕНИЕ", cats["НАПРАВЛЕНИЕ"])
year = c2.selectbox("ГОД", cats["ГОД"])
attestation = c1.selectbox("АТТЕСТАЦИЯ", cats["АТТЕСТАЦИЯ"])
discipline = c2.selectbox("ДИСЦИПЛИНА", cats["ДИСЦИПЛИНА"])
course = c1.slider("КУРС", 1, 6, 2)
semester = c2.slider("СЕМЕСТР", 1, 12, 3)

if st.button("🔮 Предсказать оценку", type="primary", use_container_width=True):
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
    label = {2: "неуд", 3: "удовлетворительно", 4: "хорошо", 5: "отлично"}.get(rounded, "—")
    st.info(f"Округлённо: **{rounded}** ({label})")

    st.progress(min(1.0, pred_clipped / 5.0))
    with st.expander("Детали входа модели"):
        st.json(row)

st.caption(f"⚠️ {_cap}. Не для реального выставления оценок.")
