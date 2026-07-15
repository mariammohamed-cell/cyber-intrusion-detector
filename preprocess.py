import pandas as pd
import numpy as np
import joblib

# تحميل الملفات المحفوظة من النوتبوك
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")
encryption_mode = joblib.load("encryption_mode.pkl")


def preprocess_input(raw: dict) -> pd.DataFrame:
    """
    راخد dictionary فيه بيانات session واحدة جديدة من اليوزر،
    وبيرجع DataFrame جاهز يتدخل مباشرة على model.predict()

    مفاتيح الـ dict المطلوبة:
    network_packet_size, protocol_type, login_attempts, session_duration,
    encryption_used, ip_reputation_score, failed_logins, browser_type,
    unusual_time_access
    """
    df = pd.DataFrame([raw])

    # 1. لو اليوزر سايب encryption_used فاضي -> نحط الـ mode المحفوظ
    df["encryption_used"] = df["encryption_used"].replace("", np.nan)
    df["encryption_used"] = df["encryption_used"].fillna(encryption_mode)

    # 2. نفس الـ feature engineering اللي في النوتبوك
    df["packet_per_duration"] = df["network_packet_size"] / df["session_duration"]
    df["failed_login_ratio"] = df["failed_logins"] / df["login_attempts"]

    # 3. get_dummies زي ما حصل بالظبط في التدريب
    df_encoded = pd.get_dummies(df, drop_first=True)

    # 4. محاذاة الأعمدة مع أعمدة التدريب (أي عمود ناقص = صفر)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    # 5. log1p على session_duration (زي ما اتعمل في التدريب)
    df_encoded["session_duration"] = np.log1p(df_encoded["session_duration"])

    # 6. Scaling
    df_scaled = pd.DataFrame(
        scaler.transform(df_encoded), columns=df_encoded.columns
    )

    return df_scaled
