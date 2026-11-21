from pathlib import Path

import flet as ft
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "iris.data"

COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]

PERSIAN_NAMES = {
    "Iris-setosa": "زنبق ستوسا",
    "Iris-versicolor": "زنبق ورسیکالر",
    "Iris-virginica": "زنبق ویرجینیکا",
}
def load_model():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"فایل iris.data پیدا نشد:\n{CSV_PATH}"
        )

    df = pd.read_csv(
        CSV_PATH,
        header=None,
        names=COLUMNS,
    )

    # تبدیل ویژگی‌ها به عدد؛ دادهٔ خراب باعث خطای واضح می‌شود
    for column in COLUMNS[:-1]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[COLUMNS[:-1]].isna().any().any():
        raise ValueError("در ویژگی‌های عددی دادهٔ خالی وجود دارد.")

    X = df[COLUMNS[:-1]]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy
def main(page: ft.Page):
    page.title = "Iris Predictor | پیش‌بینی گل زنبق"
    page.window.width = 720
    page.window.height = 850
    page.padding = 24
    page.bgcolor = "#101014"
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    yellow = "#FACC15"
    card_color = "#1E1E26"
    white = "#F5F5F5"
    sepal_length = ft.TextField(
        label="طول کاسبرگ | Sepal length",
        value="5.1",
        keyboard_type=ft.KeyboardType.NUMBER,
        border_color=yellow,
        focused_border_color=yellow,
        text_size=15,
    )

    sepal_width = ft.TextField(
        label="عرض کاسبرگ | Sepal width",
        value="3.5",
        keyboard_type=ft.KeyboardType.NUMBER,
        border_color=yellow,
        focused_border_color=yellow,
        text_size=15,
    )

    petal_length = ft.TextField(
        label="طول گلبرگ | Petal length",
        value="1.4",
        keyboard_type=ft.KeyboardType.NUMBER,
        border_color=yellow,
        focused_border_color=yellow,
        text_size=15,
    )

    petal_width = ft.TextField(
        label="عرض گلبرگ | Petal width",
        value="0.2",
        keyboard_type=ft.KeyboardType.NUMBER,
        border_color=yellow,
        focused_border_color=yellow,
        text_size=15,
    )
    message = ft.Text(
        "",
        color="#FF6B6B",
        text_align=ft.TextAlign.CENTER,
    )

    result_text = ft.Text(
        "",
        size=22,
        weight=ft.FontWeight.BOLD,
        color=yellow,
        text_align=ft.TextAlign.CENTER,
    )

    result_card = ft.Container(
        content=result_text,
        visible=False,
        bgcolor=card_color,
        border=ft.Border.all(2, yellow),
        border_radius=18,
        padding=20,
        alignment=ft.Alignment.CENTER,
        width=500,
    )
    def predict_clicked(e):
        message.value = ""
        result_card.visible = False

        try:
            values = [
                float(sepal_length.value),
                float(sepal_width.value),
                float(petal_length.value),
                float(petal_width.value),
            ]

            if any(value <= 0 for value in values):
                raise ValueError("همهٔ اندازه‌ها باید بزرگ‌تر از صفر باشند.")

            model, accuracy = load_model()

            sample = pd.DataFrame(
                [values],
                columns=COLUMNS[:-1],
            )

            prediction = model.predict(sample)[0]
            persian_name = PERSIAN_NAMES.get(prediction, prediction)

            result_text.value = (
                f"{persian_name}\n{prediction}"
            )
            result_card.visible = True

            accuracy_text.value = (
                f"دقت مدل روی دادهٔ آزمایشی: {accuracy:.2%}\n"
                "این عدد، میزان اطمینانِ پیش‌بینی همین گل نیست."
            )

        except ValueError as error:
            message.value = (
                "ورودی نامعتبر است. چهار عدد مثبت وارد کن.\n"
                f"جزئیات: {error}"
            )
        except Exception as error:
            message.value = f"خطا در اجرای برنامه: {error}"

        page.update()
    accuracy_text = ft.Text(
        "",
        color="#B8B8C0",
        size=13,
        text_align=ft.TextAlign.CENTER,
    )

    title = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(
                    "🌼 IRIS CLASSIFIER",
                    color=yellow,
                    size=15,
                    weight=ft.FontWeight.BOLD,
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.Text(
                    "پیش‌بینی گونهٔ گل زنبق",
                    color=white,
                    size=28,
                    weight=ft.FontWeight.BOLD,
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.Text(
                    "چهار اندازه را وارد کن و گونهٔ گل را پیش‌بینی کن",
                    color="#B8B8C0",
                    size=14,
                    text_align=ft.TextAlign.CENTER,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
        ),
        bgcolor=card_color,
        border=ft.Border.all(1, yellow),
        border_radius=22,
        padding=24,
        width=600,
    )

    predict_button = ft.Button(
        content="🔍 پیش‌بینی گونه | Predict",
        on_click=predict_clicked,
        bgcolor=yellow,
        color="#101014",
        expand=True,
    )

    inputs_card = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(
                    "اندازه‌های گل | Flower measurements",
                    size=18,
                    weight=ft.FontWeight.BOLD,
                    color=white,
                    text_align=ft.TextAlign.CENTER,
                ),
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
                predict_button,
                message,
            ],
            spacing=14,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor=card_color,
        border_radius=22,
        padding=24,
        width=600,
    )

    page.add(
        ft.Column(
            controls=[
                title,
                inputs_card,
                result_card,
                accuracy_text,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=18,
            width=600,
        )
    )
ft.run(main)
