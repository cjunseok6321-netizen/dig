import io
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

st.set_page_config(page_title="극지식물 개체별 생육 차이 분석", page_icon="🌱", layout="wide")

# 한글 폰트 (깨짐 방지)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

FONT = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"
COLORS = {"상위 25%": "#d9480f", "중간 50%": "#adb5bd", "하위 25%": "#1c7ed6", "전체": "#868e96"}
UNIT = {"생중량": "g", "잎 수": "장", "지상부 길이": "mm", "지하부 길이": "mm"}
GROWTH_COLS = ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
# 분석에 쓰는 송도고 데이터 (main.py와 같은 폴더)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV = os.path.join(BASE_DIR, "송도고_환경데이터.csv")
DEFAULT_GROWTH = os.path.join(BASE_DIR, "송도고_생육결과데이터.csv")
ENV_LABEL = {"temperature": "온도(℃)", "humidity": "습도(%)", "ph": "pH", "ec": "EC(mS/cm)"}


# ---------------------------------------------------------------- 데이터 읽기
def read_csv(data):
    # 바이트를 한 번만 꺼내 두고 인코딩을 바꿔 가며 읽기
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


def id_text(v):
    # 개체번호를 56.0이 아니라 원래 값(56, A1)으로 표시
    try:
        f = float(v)
        if f == int(f):
            return str(int(f))
    except Exception:
        pass
    return str(v)


@st.cache_data(show_spinner=False)
def load_df(data):
    df = std_cols(read_csv(data))
    if "개체번호" in df.columns:
        df["개체번호"] = df["개체번호"].map(id_text)
    return df


def has_num(df, col, n=2):
    return df is not None and col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].dropna().shape[0] >= n


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def style(fig, title=None, height=380):
    fig.update_layout(font=dict(family=FONT), title=title, height=height, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- 데이터 (송도고 CSV를 한 번만 읽기)
def read_default(path):
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read()
    return None


env_df, growth_df = None, None
with st.spinner("데이터를 불러오는 중..."):
