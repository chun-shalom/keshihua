import os
import chardet
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import dash_daq as daq
import requests
import io
from pathlib import Path

# ===== 1) 基础配置 =====
# 优先用远程 CSV（环境变量 CSV_URL 提供），否则读取本地 data/ 目录
CSV_URL = os.getenv("CSV_URL", "").strip()
CSV_PATH = Path(__file__).parent / "data" / "doc_risk_scores_k7.csv"

RED = "#C25759"
RED_LIGHT = "#E69191"
BLUE = "#599CB4"
BLUE_LIGHT = "#92B5CA"

# ===== 2) 读数工具 =====
def read_csv_smart(path_or_url) -> pd.DataFrame:
    # 如果是 URL，就在线读取
    if isinstance(path_or_url, str) and (path_or_url.startswith("http://") or path_or_url.startswith("https://")):
        resp = requests.get(path_or_url, timeout=30)
        resp.raise_for_status()
        enc = chardet.detect(resp.content[:50000]).get("encoding") or "utf-8"
        buf = io.BytesIO(resp.content)
        return pd.read_csv(buf, encoding=enc, low_memory=False)
    # 否则当作本地文件
    with open(path_or_url, "rb") as f:
        enc = chardet.detect(f.read(50_000)).get("encoding") or "utf-8"
    return pd.read_csv(path_or_url, encoding=enc, low_memory=False)

# ===== 3) 加载与整理数据 =====
raw = read_csv_smart(CSV_URL if CSV_URL else CSV_PATH)
raw.columns = [c.strip() for c in raw.columns]

required = {"company", "RD_0", "RD_1", "RD_2", "RD_3", "RD_4", "RD_5", "RD_6"}
missing = required - set(raw.columns)
if missing:
    raise ValueError(f"CSV 缺少必要列：{missing}")

if "year" not in raw.columns:
    raw["year"] = "全部"
if "industry" not in raw.columns:
    raw["industry"] = "未指定"

raw["市场风险"]   = raw["RD_0"]
raw["信用风险"]   = raw["RD_1"]
raw["操作风险"]   = raw["RD_2"] + raw["RD_5"]
raw["法律合规风险"] = raw["RD_3"]
raw["技术风险"]   = raw["RD_4"] + raw["RD_6"]

wide = raw[["company", "year", "industry",
            "市场风险", "信用风险", "操作风险", "法律合规风险", "技术风险"]].copy()

CATS = ["市场风险", "信用风险", "操作风险", "法律合规风险", "技术风险"]
YEARS = list(pd.unique(wide["year"]))
YEARS.sort()

wide["综合"] = wide[CATS].mean(axis=1)

def top_companies(year, n=8):
    dfy = wide[wide["year"] == year].sort_values("综合", ascending=False)
    return dfy["company"].head(n).tolist()

# ===== 4) Dash 应用 =====
app = Dash(__name__, title="可视化风险仪表盘", suppress_callback_exceptions=True)
server = app.server  # 关键：给 Gunicorn 用

theme_colors = {
    'light': {'background': 'white', 'text': 'black'},
    'dark': {'background': '#2E2E2E', 'text': 'white'}
}

app.layout = html.Div(
    style={"fontFamily": "Microsoft YaHei, SimHei, Arial"},
    children=[
        daq.ToggleSwitch(
            id='theme-switch',
            label="深色模式",
            value=False,
            labelPosition='top',
            color="lightgray"
        ),
        html.Div(id="main-content")
    ]
)

# ===== 5) 主内容根据主题变化 =====
@app.callback(
    Output("main-content", "children"),
    Input("theme-switch", "value")
)
def toggle_theme(is_dark):
    theme = theme_colors['dark'] if is_dark else theme_colors['light']
    text_color = theme['text']
    bg_color = theme['background']

    return html.Div(
        style={"backgroundColor": bg_color, "color": text_color, "padding": "10px"},
        children=[
            html.H2("上市公司风险可视化仪表盘",
                    style={"textAlign": "center", "margin": "12px 0", "color": text_color}),

            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                            "alignItems": "center", "justifyContent": "space-between", "marginBottom": "6px"},
                     children=[
                         html.Div(children=[
                             html.Div("选择年份"),
                             dcc.Dropdown(id="year",
                                          options=[{"label": str(y), "value": y} for y in YEARS],
                                          value=YEARS[-1],
                                          clearable=False,
                                          style={"minWidth": "160px"})
                         ]),
                         html.Div(children=[
                             html.Div("选择公司（雷达图）"),
                             dcc.Dropdown(id="company",
                                          options=[], value=None, clearable=False,
                                          style={"minWidth": "220px"})
                         ]),
                         html.Div(children=[
                             html.Div("选择行业"),
                             dcc.Dropdown(id="industry-filter",
                                          options=[], value=None, clearable=True,
                                          style={"minWidth": "220px"})
                         ]),
                         html.Div(children=[
                             html.Div("对比公司（可多选）"),
                             dcc.Dropdown(id="companies_compare",
                                          options=[], value=[], multi=True,
                                          style={"minWidth": "320px"})
                         ]),
                         html.Div(children=[
                             html.Div("对比维度"),
                             dcc.Dropdown(id="compare_metric",
                                          options=[{"label": "综合（各类别均值）", "value": "综合"}] +
                                                  [{"label": c, "value": c} for c in CATS],
                                          value="综合", clearable=False,
                                          style={"minWidth": "220px"})
                         ]),
                     ]),

            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
                     children=[
                         dcc.Graph(id="radar", config={"displaylogo": False}),
                         dcc.Graph(id="bars", config={"displaylogo": False}),
                     ]),

            html.Div(children=[
                html.H3("热力图"),
                dcc.Graph(id="heatmap")
            ])
        ]
    )

# ===== 6) 下拉选项联动 =====
@app.callback(
    Output("company", "options"),
    Output("company", "value"),
    Output("industry-filter", "options"),
    Output("industry-filter", "value"),
    Output("companies_compare", "options"),
    Output("companies_compare", "value"),
    Input("year", "value"),
)
def _options_by_year(year):
    dfy = wide[wide["year"] == year].copy()
    opts = [{"label": company, "value": company} for company in dfy["company"].unique()]
    if dfy.empty:
        return [], None, [], None, [], []
    default_company = dfy.sort_values("综合", ascending=False)["company"].iloc[0]
    default_compare = top_companies(year, n=min(8, len(dfy)))
    industry_opts = [{"label": industry, "value": industry} for industry in dfy["industry"].unique()]
    return opts, default_company, industry_opts, None, opts, default_compare

# ===== 7) 雷达图 =====
@app.callback(
    Output("radar", "figure"),
    Input("year", "value"),
    Input("company", "value"),
    Input("theme-switch", "value"),
)
def _radar(year, company, is_dark):
    theme = theme_colors['dark'] if is_dark else theme_colors['light']
    text_color = theme['text']
    bg_color = theme['background']

    dfy = wide[(wide["year"] == year) & (wide["company"] == company)]
    if dfy.empty:
        r = [0] * len(CATS)
    else:
        r = dfy.iloc[0][CATS].tolist()
    theta = CATS + [CATS[0]]
    rv = r + [r[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=rv, theta=theta, fill="toself",
        name=company, line=dict(color=RED, width=3),
        fillcolor="rgba(194,87,89,0.20)"
    ))
    fig.update_layout(
        title=f"{company} 风险画像（{year}）",
        polar=dict(radialaxis=dict(visible=True, tickfont=dict(size=10, color=text_color))),
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        showlegend=False
    )
    return fig

# ===== 8) 多公司对比条形图 =====
@app.callback(
    Output("bars", "figure"),
    Input("year", "value"),
    Input("companies_compare", "value"),
    Input("compare_metric", "value"),
    Input("theme-switch", "value"),
)
def _bars(year, companies, metric, is_dark):
    theme = theme_colors['dark'] if is_dark else theme_colors['light']
    text_color = theme['text']
    bg_color = theme['background']

    dfy = wide[wide["year"] == year].copy()
    dfy = dfy[dfy["company"].isin(companies)]
    if dfy.empty:
        return go.Figure()

    dfy = dfy.sort_values(metric, ascending=True)
    fig = px.bar(
        dfy, x=metric, y="company", orientation="h",
        color=np.where(dfy[metric] >= 0, "正", "负"),
        color_discrete_map={"正": RED, "负": BLUE},
        text=dfy[metric].round(2)
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title=f"公司对比（{year}，维度：{metric}）",
        xaxis_title="风险得分",
        yaxis_title="公司",
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        legend_title_text=None
    )
    return fig

# ===== 9) 热力图 =====
@app.callback(
    Output("heatmap", "figure"),
    Input("year", "value"),
    Input("theme-switch", "value"),
)
def _heatmap(year, is_dark):
    theme = theme_colors['dark'] if is_dark else theme_colors['light']
    text_color = theme['text']
    bg_color = theme['background']

    dfy = wide[wide["year"] == year].copy()
    if dfy.empty:
        return go.Figure()
    fig = px.imshow(dfy[CATS].T, color_continuous_scale="RdBu_r",
                    labels=dict(x="公司", y="风险类别", color="得分"))
    fig.update_layout(
        title=f"{year} 风险维度热力图",
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor=bg_color,
        font=dict(color=text_color)
    )
    return fig

# ===== 10) 启动入口 =====
if __name__ == "__main__":
    app.run_server(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False
    )
