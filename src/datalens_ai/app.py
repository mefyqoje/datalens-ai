"""Streamlit entry point for the DataLens AI web application."""

import pandas as pd
import streamlit as st

from datalens_ai.config import APP_TITLE
from datalens_ai.data_loader import load_csv
from datalens_ai.modeling import (
    ALGORITHM_LABELS,
    available_algorithms,
    train_and_evaluate,
)
from datalens_ai.profiling import profile_dataset
from datalens_ai.visualizations import (
    CORRELATION,
    FREQUENCY,
    HISTOGRAM,
    SCATTER,
    available_visualizations,
    create_visualization,
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1220;
        color: #e5edf5;
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #0e1726;
        border-right: 1px solid #263449;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: #0b1220;
    }
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],
    label,
    [data-testid="stText"] {
        color: #e5edf5;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9;
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stFileUploader"] label {
        color: #cbd5e1;
    }
    [data-testid="stFileUploader"],
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: #121c2d;
        border-radius: 12px;
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input {
        background-color: #172337;
        color: #e5edf5;
    }
    [data-baseweb="select"] *,
    [data-baseweb="input"] *,
    [data-testid="stNumberInput"] input {
        color: #e5edf5;
    }
    [data-testid="stTabs"] button {
        color: #cbd5e1;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #63d6b3;
    }
    [data-testid="stAlert"] p {
        color: #e5edf5;
    }
    [data-testid="stMetric"] {
        background: #121c2d;
        border: 1px solid #263449;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }
    [data-testid="stMetricLabel"] {
        color: #a9b8cb;
        font-size: 0.9rem;
    }
    [data-testid="stMetricValue"] {
        color: #f1f5f9;
    }
    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
    div[data-testid="stForm"] {
        background: #121c2d;
        border: 1px solid #263449;
        border-radius: 14px;
        padding: 1.25rem;
    }
    .hero {
        background: linear-gradient(115deg, #13263c 0%, #155447 100%);
        border-radius: 18px;
        padding: 1.6rem 2rem;
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2rem;
        margin: 0 0 0.35rem 0;
    }
    .hero p {
        color: #d7eee8;
        margin: 0;
        font-size: 1rem;
    }
    .section-note {
        color: #a9b8cb;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
    [data-testid="stDownloadButton"] button {
        border-color: #45c4a0;
        color: #9ae6d0;
        background: #172337;
    }
    [data-testid="stDownloadButton"] button:hover {
        border-color: #76e0c1;
        color: #f1f5f9;
        background: #1d3046;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #121c2d;
        border-color: #263449;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📊 DataLens AI")
    st.caption("Исследуйте данные и находите закономерности")
    st.divider()
    st.markdown("### Загрузите данные")
    uploaded_file = st.file_uploader(
        "CSV-файл",
        type=["csv"],
        help="Поддерживаются CSV-файлы. Файл обрабатывается в текущем сеансе.",
    )
    if uploaded_file is None:
        st.info("Загрузите CSV, чтобы открыть обзор, графики и машинное обучение.")
    else:
        st.success(f"Файл готов: {uploaded_file.name}")
        st.caption(f"Размер файла: {uploaded_file.size / 1024:.1f} КБ")
    st.divider()
    st.caption("Поддержка: CSV · EDA · визуализации · базовое ML")

st.markdown(
    f"""
    <div class="hero">
        <h1>{APP_TITLE}</h1>
        <p>Путь от CSV-файла к понятным выводам — в одном рабочем пространстве.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.subheader("Начните с загрузки данных")
    st.markdown(
        """
        Загрузите CSV-файл через панель слева. DataLens AI покажет структуру и
        качество данных, поможет построить графики и оценить базовую модель
        машинного обучения.
        """
    )
    first_step, second_step, third_step = st.columns(3)
    with first_step.container(border=True):
        st.markdown("#### 01 · Загрузите")
        st.write("Добавьте CSV-файл — содержимое появится в этом сеансе.")
    with second_step.container(border=True):
        st.markdown("#### 02 · Исследуйте")
        st.write("Проверьте статистику, пропуски, дубликаты и распределения.")
    with third_step.container(border=True):
        st.markdown("#### 03 · Смоделируйте")
        st.write("Выберите целевой столбец и сравните модель с базовым алгоритмом.")
else:
    try:
        dataset = load_csv(uploaded_file)
    except (
        UnicodeDecodeError,
        OSError,
        pd.errors.ParserError,
        ValueError,
    ) as error:
        st.error(f"Не удалось загрузить CSV-файл: {error}")
    else:
        if dataset.empty or len(dataset.columns) == 0:
            st.warning("В файле нет строк с данными. Загрузите CSV с заголовками и строками.")
        else:
            profile = profile_dataset(dataset)
            st.caption(
                f"Файл: **{uploaded_file.name}** · "
                f"{len(dataset):,} строк · {len(dataset.columns):,} столбцов"
                .replace(",", " ")
            )
            metrics = st.columns(4)
            metrics[0].metric("Строки", f"{profile['rows']:,}".replace(",", " "))
            metrics[1].metric("Столбцы", profile["columns"])
            metrics[2].metric("Пропуски", f"{profile['missing_values']:,}".replace(",", " "))
            metrics[3].metric("Дубликаты", f"{profile['duplicates']:,}".replace(",", " "))

            overview_tab, charts_tab, modeling_tab, data_tab = st.tabs(
                ["Обзор", "Визуализации", "Машинное обучение", "Данные"]
            )

            with overview_tab:
                st.subheader("Сводка по набору данных")
                st.markdown(
                    '<p class="section-note">Основные статистические показатели и качество столбцов.</p>',
                    unsafe_allow_html=True,
                )
                st.markdown("#### Описательная статистика")
                st.dataframe(
                    profile["summary"],
                    hide_index=True,
                    use_container_width=True,
                )
                st.markdown("#### Типы и полнота данных")
                st.dataframe(
                    profile["columns_info"],
                    hide_index=True,
                    use_container_width=True,
                )

            with charts_tab:
                st.subheader("Визуализации")
                st.markdown(
                    '<p class="section-note">Выберите тип графика и признаки для исследования.</p>',
                    unsafe_allow_html=True,
                )
                chart_types = available_visualizations(dataset)
                chart_type = st.selectbox("Тип графика", chart_types)
                numeric_columns = dataset.select_dtypes(include="number").columns.tolist()
                if chart_type == HISTOGRAM:
                    column = st.selectbox("Числовой признак", numeric_columns)
                    figure = create_visualization(dataset, chart_type, column=column)
                elif chart_type == FREQUENCY:
                    column = st.selectbox("Признак", dataset.columns.tolist())
                    figure = create_visualization(dataset, chart_type, column=column)
                elif chart_type == SCATTER:
                    x_column, y_column = st.columns(2)
                    x = x_column.selectbox("Ось X", numeric_columns)
                    y = y_column.selectbox(
                        "Ось Y",
                        numeric_columns,
                        index=1 if len(numeric_columns) > 1 else 0,
                    )
                    figure = create_visualization(
                        dataset, chart_type, x_column=x, y_column=y
                    )
                else:
                    figure = create_visualization(dataset, CORRELATION)
                figure.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#121c2d",
                    plot_bgcolor="#121c2d",
                    margin=dict(l=20, r=20, t=55, b=20),
                    font=dict(family="sans-serif", color="#dbe5f0"),
                )
                st.plotly_chart(figure, use_container_width=True)

            with modeling_tab:
                st.subheader("Обучение модели")
                st.markdown(
                    '<p class="section-note">Сравните случайный лес с простым базовым алгоритмом на отложенной выборке.</p>',
                    unsafe_allow_html=True,
                )
                with st.form("model_training_form"):
                    target_column = st.selectbox(
                        "Что будем предсказывать?",
                        dataset.columns.tolist(),
                        help="Выберите столбец, значение которого модель должна предсказать.",
                    )
                    suggested_task = (
                        "regression"
                        if pd.api.types.is_numeric_dtype(dataset[target_column])
                        else "classification"
                    )
                    task_type = st.selectbox(
                        "Тип задачи",
                        ["classification", "regression"],
                        index=["classification", "regression"].index(suggested_task),
                        format_func=lambda value: (
                            "Классификация" if value == "classification" else "Регрессия"
                        ),
                    )
                    algorithms = st.multiselect(
                        "Алгоритмы для сравнения",
                        options=available_algorithms(task_type),
                        default=available_algorithms(task_type),
                        format_func=lambda value: ALGORITHM_LABELS[value],
                        help="Линейная модель, дерево решений и случайный лес.",
                    )
                    test_fraction = st.slider(
                        "Доля данных для проверки",
                        min_value=0.1,
                        max_value=0.4,
                        value=0.2,
                        step=0.05,
                    )
                    train_model = st.form_submit_button(
                        "Обучить и сравнить", type="primary", use_container_width=True
                    )

                if train_model:
                    try:
                        result = train_and_evaluate(
                            dataset,
                            target_column,
                            task_type,
                            test_size=test_fraction,
                            algorithms=algorithms,
                        )
                    except (ValueError, TypeError) as error:
                        st.error(f"Не удалось обучить модель: {error}")
                    else:
                        st.success("Обучение завершено")
                        st.caption(
                            f"Обучающих строк: {result['train_rows']}; "
                            f"проверочных строк: {result['test_rows']}. "
                            "Строки с пропущенной целевой переменной исключены."
                        )
                        metric_labels = {
                            "accuracy": "Accuracy",
                            "f1_weighted": "F1 (weighted)",
                            "mae": "MAE",
                            "rmse": "RMSE",
                            "r2": "R²",
                        }
                        comparison = pd.DataFrame(
                            {
                                "Метрика": [
                                    metric_labels[name]
                                    for name in result["baseline_metrics"]
                                ],
                                **{
                                    ALGORITHM_LABELS[algorithm]: [
                                        metrics[name] for name in result["baseline_metrics"]
                                    ]
                                    for algorithm, metrics in result[
                                        "model_metrics"
                                    ].items()
                                },
                                "Базовая модель": list(result["baseline_metrics"].values()),
                            }
                        )
                        st.dataframe(
                            comparison, hide_index=True, use_container_width=True
                        )
                        st.caption(
                            "Для Accuracy, F1 и R² больше — лучше; для MAE и RMSE — меньше."
                        )

            with data_tab:
                st.subheader("Предпросмотр данных")
                st.markdown(
                    '<p class="section-note">Просматривайте строки и скачивайте обработанную таблицу в CSV.</p>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "Скачать CSV",
                    data=dataset.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_export.csv",
                    mime="text/csv",
                )
                st.dataframe(
                    dataset,
                    use_container_width=True,
                    height=520,
                )
