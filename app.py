import html
import os
import platform
import re
from datetime import datetime
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

COLUMNS = ["Date", "User", "Message"]

DATE_LINE = re.compile(r"^-+\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일")
MESSAGE_LINE = re.compile(
    r"^\[(.+?)\]\s*\[(오전|오후)\s*(\d{1,2}):(\d{2})\]\s?(.*)$"
)


def decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_kakao_txt(text: str) -> pd.DataFrame:
    """카카오톡 대화 내보내기 텍스트를 Date, User, Message 표로 변환한다."""
    current_date = None
    rows: list[dict] = []

    for line in text.splitlines():
        date_match = DATE_LINE.match(line.strip())
        if date_match:
            year, month, day = (int(part) for part in date_match.groups())
            current_date = datetime(year, month, day)
            continue

        message_match = MESSAGE_LINE.match(line)
        if message_match and current_date is not None:
            user, ampm, hour_text, minute_text, message = message_match.groups()
            hour = int(hour_text)
            minute = int(minute_text)
            if ampm == "오후" and hour != 12:
                hour += 12
            elif ampm == "오전" and hour == 12:
                hour = 0
            rows.append(
                {
                    "Date": current_date.replace(hour=hour, minute=minute),
                    "User": user.strip(),
                    "Message": message,
                }
            )
            continue

        if rows and current_date is not None:
            previous = rows[-1]["Message"]
            rows[-1]["Message"] = f"{previous}\n{line}" if previous else line

    frame = pd.DataFrame(rows, columns=COLUMNS)
    if not frame.empty:
        frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


def load_csv(raw: bytes) -> pd.DataFrame:
    last_error = None
    frame = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            frame = pd.read_csv(BytesIO(raw), encoding=encoding)
            break
        except UnicodeDecodeError as error:
            last_error = error
    if frame is None:
        raise ValueError(f"CSV 인코딩을 읽지 못했습니다: {last_error}")

    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        found = ", ".join(map(str, frame.columns))
        raise ValueError(
            "CSV에는 Date, User, Message 컬럼이 있어야 합니다. "
            f"없는 컬럼: {', '.join(missing)}. 현재 컬럼: {found}"
        )

    frame = frame.loc[:, COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["User"] = frame["User"].fillna("").astype(str)
    frame["Message"] = frame["Message"].fillna("").astype(str)
    return frame


BAR_COLORS = [
    "#E8A317",
    "#3E527A",
    "#D4654A",
    "#2F8F7A",
    "#6B5BD2",
    "#C44E78",
    "#3D7EDB",
    "#8A6A4F",
]


def message_lengths(chat: pd.DataFrame) -> pd.Series:
    return chat["Message"].fillna("").astype(str).str.len()


def build_participant_stats(chat: pd.DataFrame) -> pd.DataFrame:
    """참여자별 메시지 수, 총 글자 수, 평균 메시지 길이를 계산한다."""
    counted = chat.assign(_len=message_lengths(chat))
    counted = counted[counted["User"].astype(str).str.strip() != ""]
    stats = (
        counted.groupby("User", as_index=False)
        .agg(메시지=("_len", "size"), 총글자=("_len", "sum"), 평균길이=("_len", "mean"))
        .sort_values(["메시지", "총글자"], ascending=[False, False])
        .reset_index(drop=True)
    )
    stats["평균길이"] = stats["평균길이"].round(1)
    return stats


def longest_message_row(chat: pd.DataFrame) -> pd.Series:
    counted = chat.assign(_len=message_lengths(chat))
    counted = counted[counted["User"].astype(str).str.strip() != ""]
    ordered = counted.sort_values(["_len", "Date"], ascending=[False, True], na_position="last")
    return ordered.iloc[0]


def message_count_figure(stats: pd.DataFrame) -> go.Figure:
    plot_df = stats.sort_values("메시지", ascending=True)
    count = len(plot_df)
    colors = [BAR_COLORS[(count - 1 - index) % len(BAR_COLORS)] for index in range(count)]
    max_count = int(plot_df["메시지"].max()) if count else 1

    figure = go.Figure(
        go.Bar(
            x=plot_df["메시지"],
            y=plot_df["User"],
            orientation="h",
            marker=dict(color=colors, cornerradius=8, line=dict(width=0)),
            text=[f"{int(value):,}" for value in plot_df["메시지"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:,}개<extra></extra>",
        )
    )
    figure.update_layout(
        title=dict(text="참여자별 메시지 수", x=0, xanchor="left", font=dict(size=18, color="#2C2C2C")),
        xaxis=dict(
            title="메시지 수",
            range=[0, max_count * 1.22],
            showgrid=True,
            gridcolor="#F0E8DA",
            zeroline=False,
            tickformat=",",
            tickfont=dict(color="#6B645B"),
            title_font=dict(color="#6B645B"),
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            automargin=True,
            tickfont=dict(color="#2C2C2C", size=14),
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif", color="#2C2C2C", size=14),
        margin=dict(l=24, r=72, t=64, b=56),
        height=max(340, 78 * count + 140),
        bargap=0.42,
        showlegend=False,
    )
    figure.update_traces(textfont=dict(color="#2C2C2C", size=13))
    return figure


KPI_ACCENTS = ["#E8A317", "#3E527A", "#2F8F7A", "#D4654A"]


def accent_bar(color: str) -> None:
    st.markdown(
        (
            f"<div style='height:4px;border-radius:999px;background:{color};"
            "margin:0 0 8px 0'></div>"
        ),
        unsafe_allow_html=True,
    )


def conversation_span(chat: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int | None]:
    dated = messages_with_datetime(chat)
    if dated.empty:
        return None, None, None
    start = dated["Date"].min()
    end = dated["Date"].max()
    days = int((end.normalize() - start.normalize()).days) + 1
    return start, end, max(days, 1)


def render_participant_stats(chat: pd.DataFrame) -> None:
    stats = build_participant_stats(chat)
    if stats.empty:
        st.info("참여자 이름을 찾지 못했습니다.")
        return

    total_messages = int(stats["메시지"].sum())
    st.subheader("참여자별 통계")
    row_size = min(len(stats), 3)
    for start in range(0, len(stats), row_size):
        chunk = stats.iloc[start : start + row_size]
        columns = st.columns(len(chunk))
        for column, (_, row), color in zip(columns, chunk.iterrows(), BAR_COLORS[start:], strict=False):
            share = (int(row["메시지"]) / total_messages * 100) if total_messages else 0
            with column:
                with st.container(border=True):
                    name = html.escape(str(row["User"]))
                    st.markdown(
                        (
                            f"<div style='font-size:1.05rem;font-weight:700;color:{color};"
                            f"margin-bottom:4px'>{name}</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                    accent_bar(color)
                    count_col, length_col, char_col = st.columns(3)
                    count_col.metric(
                        "메시지",
                        f"{int(row['메시지']):,}",
                        f"{share:.1f}%",
                        delta_color="off",
                        delta_arrow="off",
                    )
                    length_col.metric("평균 길이", f"{row['평균길이']:.1f}자")
                    char_col.metric("총 글자", f"{int(row['총글자']):,}")

    st.plotly_chart(
        message_count_figure(stats),
        width="stretch",
        config={"displayModeBar": False},
    )

    longest = longest_message_row(chat)
    sent_at = longest["Date"]
    sent_label = sent_at.strftime("%Y-%m-%d %H:%M") if pd.notna(sent_at) else "시간 없음"
    st.subheader("가장 긴 메시지")
    with st.container(border=True):
        accent_bar("#C44E78")
        person_col, length_col, time_col = st.columns(3)
        person_col.metric("보낸 사람", str(longest["User"]))
        length_col.metric("글자 수", f"{int(longest['_len']):,}자")
        time_col.metric("보낸 시각", sent_label)
        st.text(str(longest["Message"]))


def render_overview(chat: pd.DataFrame) -> None:
    participants = sorted(name for name in chat["User"].astype(str).unique() if str(name).strip())
    lengths = message_lengths(chat)
    average_length = float(lengths.mean()) if len(lengths) else 0.0
    start, end, days = conversation_span(chat)
    daily_average = (len(chat) / days) if days else None

    st.subheader("주요 지표")
    message_col, period_col, daily_col, length_col = st.columns(4)
    cards = [
        (message_col, KPI_ACCENTS[0], "💬 전체 메시지", f"{len(chat):,}개", f"참여자 {len(participants)}명"),
        (
            period_col,
            KPI_ACCENTS[1],
            "📅 대화 기간",
            f"{days:,}일" if days else "없음",
            f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d}" if start is not None and end is not None else "날짜 없음",
        ),
        (
            daily_col,
            KPI_ACCENTS[2],
            "📈 일평균 메시지",
            f"{daily_average:.1f}개" if daily_average is not None else "없음",
            "대화 기간으로 나눈 수",
        ),
        (length_col, KPI_ACCENTS[3], "✏️ 평균 길이", f"{average_length:.1f}자", "메시지 1개 기준"),
    ]
    for column, color, label, value, note in cards:
        with column:
            with st.container(border=True):
                accent_bar(color)
                st.metric(label, value, note, delta_color="off", delta_arrow="off")

    render_participant_stats(chat)

    st.subheader("데이터 미리보기")
    st.caption("처음 10개 행")
    preview = chat.head(10).copy()
    preview["Date"] = preview["Date"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(preview, width="stretch", hide_index=True)


WEEKDAY_SHORT = ["월", "화", "수", "목", "금", "토", "일"]
WEEKDAY_FULL = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
PEAK_COLOR = "#E8A317"
REST_COLOR = "#D5DCE6"
PLOT_FONT = dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif", color="#2C2C2C", size=14)


def messages_with_datetime(chat: pd.DataFrame) -> pd.DataFrame:
    """Date 컬럼을 datetime으로 파싱하고, 시각이 없는 행은 뺀다."""
    dated = chat.copy()
    dated["Date"] = pd.to_datetime(dated["Date"], errors="coerce")
    return dated.dropna(subset=["Date"])


def hourly_counts(dated: pd.DataFrame) -> pd.Series:
    counts = dated["Date"].dt.hour.value_counts()
    return counts.reindex(range(24), fill_value=0).astype(int)


def weekday_counts(dated: pd.DataFrame) -> pd.Series:
    counts = dated["Date"].dt.dayofweek.value_counts()
    return counts.reindex(range(7), fill_value=0).astype(int)


def monthly_counts(dated: pd.DataFrame) -> pd.Series:
    counts = dated["Date"].dt.to_period("M").value_counts().sort_index()
    if counts.empty:
        return counts.astype(int)
    months = pd.period_range(counts.index.min(), counts.index.max(), freq="M")
    return counts.reindex(months, fill_value=0).astype(int)


def busiest(counts: pd.Series) -> tuple[int, list]:
    if counts.empty or int(counts.max()) == 0:
        return 0, []
    peak = int(counts.max())
    keys = [key for key, value in counts.items() if int(value) == peak]
    return peak, keys


def style_time_figure(figure: go.Figure, title: str, height: int) -> go.Figure:
    figure.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=18, color="#2C2C2C")),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=PLOT_FONT,
        margin=dict(l=56, r=40, t=64, b=56),
        height=height,
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", font=dict(color="#2C2C2C", family=PLOT_FONT["family"])),
    )
    figure.update_xaxes(
        showgrid=False,
        automargin=True,
        tickfont=dict(color="#6B645B"),
        title_font=dict(color="#6B645B"),
    )
    figure.update_yaxes(
        title="메시지 수",
        showgrid=True,
        gridcolor="#F0E8DA",
        zeroline=False,
        tickformat=",",
        automargin=True,
        tickfont=dict(color="#6B645B"),
        title_font=dict(color="#6B645B"),
    )
    return figure


def pad_category_axis(figure: go.Figure, count: int) -> go.Figure:
    """첫·마지막 눈금 글자가 차트 가장자리에서 잘리지 않게 좌우 여백을 둔다."""
    if count > 1:
        figure.update_xaxes(range=[-0.75, count - 0.25])
    return figure


def bar_colors(values: pd.Series) -> list[str]:
    peak = int(values.max()) if len(values) else 0
    return [PEAK_COLOR if int(value) == peak and peak > 0 else REST_COLOR for value in values]


def hourly_figure(counts: pd.Series) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=[f"{hour}시" for hour in counts.index],
            y=counts.values,
            marker=dict(color=bar_colors(counts), cornerradius=4, line=dict(width=0)),
            hovertemplate="%{x}<br>%{y:,}개<extra></extra>",
        )
    )
    figure.update_xaxes(tickangle=0)
    styled = style_time_figure(figure, "시간대별 메시지 분포", 420)
    peak = int(counts.max()) if len(counts) else 1
    styled.update_yaxes(range=[0, max(peak * 1.12, 1)])
    return pad_category_axis(styled, len(counts))


def weekday_figure(counts: pd.Series) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=WEEKDAY_SHORT,
            y=counts.values,
            marker=dict(color=bar_colors(counts), cornerradius=6, line=dict(width=0)),
            text=[f"{int(value):,}" for value in counts.values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}요일<br>%{y:,}개<extra></extra>",
        )
    )
    figure.update_traces(textfont=dict(color="#2C2C2C", size=13))
    peak = int(counts.max()) if len(counts) else 1
    styled = style_time_figure(figure, "요일별 메시지 분포", 420)
    styled.update_yaxes(range=[0, peak * 1.28])
    return pad_category_axis(styled, len(counts))


def monthly_figure(counts: pd.Series) -> go.Figure:
    labels = [f"{period.year}년 {period.month}월" for period in counts.index]
    figure = go.Figure(
        go.Scatter(
            x=labels,
            y=counts.values,
            mode="lines+markers",
            line=dict(color=PEAK_COLOR, width=3),
            marker=dict(size=10, color="#3E527A", line=dict(width=2, color="#FFFFFF")),
            fill="tozeroy",
            fillcolor="rgba(232, 163, 23, 0.16)",
            hovertemplate="%{x}<br>%{y:,}개<extra></extra>",
        )
    )
    styled = style_time_figure(figure, "월별 메시지 추이", 420)
    peak = int(counts.max()) if len(counts) else 1
    styled.update_yaxes(range=[0, max(peak * 1.2, 1)])
    return pad_category_axis(styled, len(counts))


def render_time_patterns(chat: pd.DataFrame) -> None:
    dated = messages_with_datetime(chat)
    dropped = len(chat) - len(dated)

    st.subheader("시간대별 분석")
    if dated.empty:
        st.info("Date를 시간으로 읽은 메시지가 없어 시간대 분석을 건너뛰었습니다.")
        return
    if dropped:
        st.caption(f"날짜를 읽지 못한 메시지 {dropped:,}개는 시간대 분석에서 제외했습니다.")

    hours = hourly_counts(dated)
    weekdays = weekday_counts(dated)
    months = monthly_counts(dated)
    hour_peak, busy_hours = busiest(hours)
    day_peak, busy_days = busiest(weekdays)

    hour_col, day_col = st.columns(2)
    hour_col.metric(
        "가장 활발한 시간대",
        ", ".join(f"{hour}시" for hour in busy_hours),
        f"{hour_peak:,}개",
        delta_color="off",
    )
    day_col.metric(
        "가장 활발한 요일",
        ", ".join(WEEKDAY_FULL[day] for day in busy_days),
        f"{day_peak:,}개",
        delta_color="off",
    )

    st.plotly_chart(hourly_figure(hours), width="stretch")
    day_chart, month_chart = st.columns(2)
    with day_chart:
        st.plotly_chart(weekday_figure(weekdays), width="stretch")
    with month_chart:
        st.plotly_chart(monthly_figure(months), width="stretch")


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z]{2,}")
NUMBER_ONLY_PATTERN = re.compile(r"^[\d\s:./-]+$")
SYSTEM_EXACT = {"사진", "이모티콘", "동영상", "보이스톡", "페이스톡", "음성메시지", "연락처", "지도"}
SYSTEM_PREFIXES = ("보이스톡", "페이스톡", "파일:", "파일 :", "삭제된 메시지")
WORD_COLORS = ["#E8A317", "#3E527A", "#D4654A", "#2F8F7A", "#6B5BD2", "#C44E78"]


def korean_font_path() -> str | None:
    """워드클라우드용 한글 폰트. Windows는 malgun.ttf, Mac은 AppleGothic."""
    if platform.system() == "Windows":
        candidates = [r"C:\Windows\Fonts\malgun.ttf"]
    elif platform.system() == "Darwin":
        candidates = [
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/System/Library/Fonts/AppleGothic.ttf",
            "/Library/Fonts/AppleGothic.ttf",
        ]
    else:
        candidates = []
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def analyzable_text(text: object) -> str | None:
    """시스템 메시지, URL만 있는 메시지, 숫자만 있는 메시지는 분석에서 뺀다."""
    cleaned = str(text or "").replace("\u200b", "").strip()
    if not cleaned:
        return None
    if cleaned in SYSTEM_EXACT or cleaned.startswith(SYSTEM_PREFIXES):
        return None
    if NUMBER_ONLY_PATTERN.fullmatch(cleaned):
        return None
    without_url = URL_PATTERN.sub(" ", cleaned)
    without_url = re.sub(r"\s+", " ", without_url).strip()
    if not without_url:
        return None
    return without_url


def message_tokens(text: object) -> list[str]:
    kept = analyzable_text(text)
    if kept is None:
        return []
    return TOKEN_PATTERN.findall(kept)


def word_counts(messages: pd.Series) -> pd.Series:
    tokens: list[str] = []
    for text in messages.fillna(""):
        tokens.extend(message_tokens(text))
    if not tokens:
        return pd.Series(dtype="int64")
    return pd.Series(tokens, dtype="string").value_counts()


def word_color(word: str, **_kwargs: object) -> str:
    return WORD_COLORS[sum(ord(char) for char in word) % len(WORD_COLORS)]


def build_wordcloud(counts: pd.Series, font_path: str) -> object:
    cloud = WordCloud(
        font_path=font_path,
        width=1400,
        height=700,
        background_color="white",
        max_words=80,
        prefer_horizontal=0.95,
        relative_scaling=0.5,
        min_font_size=18,
        margin=8,
        color_func=word_color,
        random_state=7,
    )
    return cloud.generate_from_frequencies(counts.to_dict()).to_image()


def rank_bar_figure(counts: pd.Series, title: str) -> go.Figure:
    ordered = counts.sort_values(ascending=True)
    peak = int(ordered.max()) if len(ordered) else 1
    colors = [PEAK_COLOR if int(value) == peak else "#D5DCE6" for value in ordered]
    figure = go.Figure(
        go.Bar(
            x=ordered.values,
            y=ordered.index.astype(str),
            orientation="h",
            marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
            text=[f"{int(value):,}" for value in ordered.values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:,}회<extra></extra>",
        )
    )
    figure.update_traces(textfont=dict(color="#2C2C2C", size=13))
    figure.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=18, color="#2C2C2C")),
        xaxis=dict(
            title="횟수",
            range=[0, peak * 1.25],
            showgrid=True,
            gridcolor="#F0E8DA",
            zeroline=False,
            tickformat=",",
            tickfont=dict(color="#6B645B"),
            title_font=dict(color="#6B645B"),
        ),
        yaxis=dict(title="", showgrid=False, automargin=True, tickfont=dict(color="#2C2C2C", size=14)),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=PLOT_FONT,
        margin=dict(l=16, r=56, t=64, b=40),
        height=max(360, 42 * len(ordered) + 120),
        showlegend=False,
    )
    return figure


def reaction_counts(chat: pd.DataFrame) -> pd.DataFrame:
    users = sorted(name for name in chat["User"].astype(str).str.strip().unique() if name)
    tallies = {user: {"ㅋ": 0, "ㅎ": 0} for user in users}
    for user, text in zip(chat["User"].astype(str), chat["Message"].fillna("").astype(str), strict=False):
        name = user.strip()
        if name not in tallies or analyzable_text(text) is None:
            continue
        if "ㅋ" in text:
            tallies[name]["ㅋ"] += 1
        if "ㅎ" in text:
            tallies[name]["ㅎ"] += 1
    return pd.DataFrame(
        [{"참여자": user, "ㅋ": tallies[user]["ㅋ"], "ㅎ": tallies[user]["ㅎ"]} for user in users]
    )


def reaction_figure(counts: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_bar(
        name="ㅋ 포함",
        x=counts["참여자"],
        y=counts["ㅋ"],
        marker=dict(color=PEAK_COLOR, cornerradius=6, line=dict(width=0)),
        text=[f"{int(value):,}" for value in counts["ㅋ"]],
        textposition="outside",
        hovertemplate="%{x}<br>ㅋ 포함 %{y:,}개<extra></extra>",
    )
    figure.add_bar(
        name="ㅎ 포함",
        x=counts["참여자"],
        y=counts["ㅎ"],
        marker=dict(color="#3E527A", cornerradius=6, line=dict(width=0)),
        text=[f"{int(value):,}" for value in counts["ㅎ"]],
        textposition="outside",
        hovertemplate="%{x}<br>ㅎ 포함 %{y:,}개<extra></extra>",
    )
    peak = int(counts[["ㅋ", "ㅎ"]].max().max()) if len(counts) else 1
    figure.update_layout(
        title=dict(text="ㅋ / ㅎ 포함 메시지", x=0, xanchor="left", font=dict(size=18, color="#2C2C2C")),
        barmode="group",
        bargap=0.28,
        xaxis=dict(title="", tickfont=dict(color="#2C2C2C", size=14)),
        yaxis=dict(
            title="메시지 수",
            range=[0, max(peak * 1.25, 1)],
            showgrid=True,
            gridcolor="#F0E8DA",
            zeroline=False,
            tickformat=",",
            tickfont=dict(color="#6B645B"),
            title_font=dict(color="#6B645B"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=PLOT_FONT,
        margin=dict(l=16, r=16, t=80, b=40),
        height=420,
    )
    return figure


def render_word_usage(chat: pd.DataFrame) -> None:
    st.subheader("자주 쓰는 단어")
    st.caption("1글자, 사진·이모티콘·동영상·보이스톡 같은 시스템 메시지, URL, 숫자만 있는 메시지는 제외했습니다.")

    counts = word_counts(chat["Message"])
    if counts.empty:
        st.info("조건에 맞는 단어가 없습니다.")
        return

    st.plotly_chart(rank_bar_figure(counts.head(20), "전체 메시지 단어 TOP 20"), width="stretch")

    font_path = korean_font_path()
    if font_path is None:
        st.error("한글 워드클라우드 폰트를 찾지 못했습니다. Windows는 malgun.ttf, Mac은 AppleGothic이 필요합니다.")
    else:
        st.image(build_wordcloud(counts, font_path), width="stretch", alt="전체 대화 워드클라우드")

    st.subheader("참여자별 자주 쓰는 단어")
    named = chat[chat["User"].astype(str).str.strip() != ""]
    participants = sorted(named["User"].astype(str).str.strip().unique())
    for start in range(0, len(participants), 2):
        columns = st.columns(2)
        for column, user in zip(columns, participants[start : start + 2], strict=False):
            personal = word_counts(named.loc[named["User"].astype(str).str.strip() == user, "Message"])
            with column:
                if personal.empty:
                    st.markdown(f"**{user}**")
                    st.caption("조건에 맞는 단어가 없습니다.")
                else:
                    st.plotly_chart(rank_bar_figure(personal.head(10), f"{user} TOP 10"), width="stretch")

    reactions = reaction_counts(chat)
    st.subheader("ㅋ / ㅎ 포함 메시지")
    st.caption("ㅋㅋ, ㅋㅋㅋ처럼 ㅋ이 들어간 메시지와 ㅎ이 들어간 메시지를 참여자별로 비교합니다.")
    if reactions.empty:
        st.info("비교할 참여자가 없습니다.")
        return
    k_total = int(reactions["ㅋ"].sum())
    h_total = int(reactions["ㅎ"].sum())
    k_col, h_col = st.columns(2)
    k_col.metric("ㅋ 포함 메시지", f"{k_total:,}개")
    h_col.metric("ㅎ 포함 메시지", f"{h_total:,}개")
    st.plotly_chart(reaction_figure(reactions), width="stretch")


NEW_CONVERSATION_GAP = pd.Timedelta(hours=6)
SLOW_REPLY_GAP = pd.Timedelta(hours=1)


def format_duration(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    if total < 60:
        return f"{total}초"
    minutes, remain = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}분" if remain == 0 else f"{minutes}분 {remain}초"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}시간" if minutes == 0 else f"{hours}시간 {minutes}분"


def format_reply_seconds(seconds: float) -> str:
    rounded = max(int(round(seconds)), 0)
    if rounded < 60:
        return "1분 이내"
    return f"{rounded}초 ({format_duration(seconds)})"


def shorten_message(text: object, limit: int = 60) -> str:
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def reply_events(chat: pd.DataFrame) -> pd.DataFrame:
    """화자가 바뀐 구간만 답장으로 센다.

    같은 사람의 연속 메시지는 마지막 메시지 시각을 기준으로 삼고,
    그 시각부터 6시간 이상 지난 응답은 새 대화로 보고 제외한다.
    """
    columns = ["답장한 사람", "상대", "초", "시각", "메시지"]
    dated = messages_with_datetime(chat)
    dated = dated[dated["User"].astype(str).str.strip() != ""].copy()
    if dated.empty:
        return pd.DataFrame(columns=columns)

    dated["User"] = dated["User"].astype(str).str.strip()
    dated["Message"] = dated["Message"].fillna("").astype(str)
    dated = dated.sort_values("Date", kind="mergesort")

    rows: list[dict] = []
    previous_user = None
    previous_time = None
    for user, when, message in zip(dated["User"], dated["Date"], dated["Message"], strict=True):
        if previous_user is None:
            previous_user = user
            previous_time = when
            continue
        if user == previous_user:
            previous_time = when
            continue
        gap = when - previous_time
        if pd.Timedelta(0) <= gap < NEW_CONVERSATION_GAP:
            rows.append(
                {
                    "답장한 사람": user,
                    "상대": previous_user,
                    "초": gap.total_seconds(),
                    "시각": when,
                    "메시지": message,
                }
            )
        previous_user = user
        previous_time = when
    return pd.DataFrame(rows, columns=columns)


def reply_summary(events: pd.DataFrame, participants: list[str]) -> pd.DataFrame:
    rows = []
    for name in participants:
        mine = events[events["답장한 사람"] == name] if not events.empty else events
        slow = int((mine["초"] >= SLOW_REPLY_GAP.total_seconds()).sum()) if not mine.empty else 0
        rows.append(
            {
                "참여자": name,
                "평균초": float(mine["초"].mean()) if not mine.empty else None,
                "답장횟수": int(len(mine)),
                "읽씹": slow,
            }
        )
    return pd.DataFrame(rows)


def reply_speed_figure(summary: pd.DataFrame) -> go.Figure:
    plotted = summary[summary["답장횟수"] > 0].sort_values("평균초", ascending=False)
    fastest = float(plotted["평균초"].min()) if len(plotted) else 0
    colors = [PEAK_COLOR if float(value) == fastest else REST_COLOR for value in plotted["평균초"]]
    minutes = plotted["평균초"] / 60
    peak = float(minutes.max()) if len(minutes) else 1
    figure = go.Figure(
        go.Bar(
            x=minutes,
            y=plotted["참여자"],
            orientation="h",
            marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
            text=[format_duration(value) for value in plotted["평균초"]],
            textposition="outside",
            cliponaxis=False,
            customdata=[int(round(value)) for value in plotted["평균초"]],
            hovertemplate="<b>%{y}</b><br>평균 %{customdata:,}초<extra></extra>",
        )
    )
    figure.update_traces(textfont=dict(color="#2C2C2C", size=13))
    figure.update_layout(
        title=dict(text="참여자별 평균 답장 시간", x=0, xanchor="left", font=dict(size=18, color="#2C2C2C")),
        xaxis=dict(
            title="평균 분",
            range=[0, peak * 1.45],
            showgrid=True,
            gridcolor="#F0E8DA",
            zeroline=False,
            tickfont=dict(color="#6B645B"),
            title_font=dict(color="#6B645B"),
        ),
        yaxis=dict(title="", showgrid=False, automargin=True, tickfont=dict(color="#2C2C2C", size=14)),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=PLOT_FONT,
        margin=dict(l=16, r=88, t=64, b=40),
        height=max(320, 88 * len(plotted) + 140),
        showlegend=False,
    )
    return figure


def slow_reply_figure(summary: pd.DataFrame) -> go.Figure:
    plotted = summary.sort_values("읽씹", ascending=True)
    peak = int(plotted["읽씹"].max()) if len(plotted) else 1
    colors = [PEAK_COLOR if int(value) == peak and peak > 0 else REST_COLOR for value in plotted["읽씹"]]
    figure = go.Figure(
        go.Bar(
            x=plotted["읽씹"],
            y=plotted["참여자"],
            orientation="h",
            marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
            text=[f"{int(value):,}회" for value in plotted["읽씹"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:,}회<extra></extra>",
        )
    )
    figure.update_traces(textfont=dict(color="#2C2C2C", size=13))
    figure.update_layout(
        title=dict(text="1시간 이상 늦은 답장", x=0, xanchor="left", font=dict(size=18, color="#2C2C2C")),
        xaxis=dict(
            title="횟수",
            range=[0, max(peak * 1.4, 1)],
            showgrid=True,
            gridcolor="#F0E8DA",
            zeroline=False,
            tickformat=",",
            tickfont=dict(color="#6B645B"),
            title_font=dict(color="#6B645B"),
        ),
        yaxis=dict(title="", showgrid=False, automargin=True, tickfont=dict(color="#2C2C2C", size=14)),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=PLOT_FONT,
        margin=dict(l=16, r=64, t=64, b=40),
        height=max(280, 88 * len(plotted) + 120),
        showlegend=False,
    )
    return figure


def render_reply_speed(chat: pd.DataFrame) -> None:
    st.subheader("답장 속도")
    st.caption(
        "같은 사람이 이어서 보낸 메시지는 빼고, 상대의 마지막 메시지 이후 걸린 시간만 답장으로 셉니다. "
        "6시간 이상 비어 있으면 새 대화로 보고 답장 시간에서 제외합니다."
    )

    participants = sorted(name for name in chat["User"].astype(str).str.strip().unique() if name)
    events = reply_events(chat)
    if events.empty or not participants:
        st.info("답장으로 셀 수 있는 주고받기가 없습니다.")
        return

    summary = reply_summary(events, participants)
    me = st.selectbox("나", participants, help="이 사람의 평균 답장 시간을 위에 표시합니다.")
    mine = summary.loc[summary["참여자"] == me].iloc[0]
    average_col, count_col, slow_col = st.columns(3)
    average_col.metric(
        "내 평균 답장 시간",
        format_duration(mine["평균초"]) if pd.notna(mine["평균초"]) else "없음",
    )
    count_col.metric("내 답장 횟수", f"{int(mine['답장횟수']):,}회")
    slow_col.metric("내 읽씹", f"{int(mine['읽씹']):,}회")

    speed_chart, speed_table = st.columns([1.25, 1])
    with speed_chart:
        st.plotly_chart(reply_speed_figure(summary), width="stretch")
    with speed_table:
        table = summary.copy()
        table["평균 답장"] = table["평균초"].map(lambda value: format_duration(value) if pd.notna(value) else "없음")
        table = table.rename(columns={"답장횟수": "답장 횟수"})
        st.dataframe(
            table[["참여자", "평균 답장", "답장 횟수", "읽씹"]],
            width="stretch",
            hide_index=True,
        )

    st.subheader("가장 빠른 답장 TOP 5")
    st.caption("내보내기 시각은 분 단위라, 같은 분에 보낸 답장은 1분 이내로 표시합니다.")
    fastest = events.sort_values(["초", "시각"], kind="mergesort").head(5).copy()
    fastest.insert(0, "순위", range(1, len(fastest) + 1))
    fastest["걸린 시간"] = fastest["초"].map(format_reply_seconds)
    fastest["시각"] = fastest["시각"].dt.strftime("%Y-%m-%d %H:%M")
    fastest["메시지"] = fastest["메시지"].map(shorten_message)
    st.dataframe(
        fastest[["순위", "답장한 사람", "걸린 시간", "시각", "메시지"]],
        width="stretch",
        hide_index=True,
    )

    st.subheader("읽씹")
    st.caption("1시간 이상, 6시간 미만으로 답한 횟수입니다. 6시간 이상 비어 새로 시작된 대화는 포함하지 않습니다.")
    st.plotly_chart(slow_reply_figure(summary), width="stretch")


def load_chat(filename: str, raw: bytes) -> pd.DataFrame:
    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        return load_csv(raw)
    if lower_name.endswith(".txt"):
        return parse_kakao_txt(decode_bytes(raw))
    raise ValueError("csv 또는 카카오톡 대화 txt 파일만 업로드할 수 있습니다.")


def chat_period_label(chat: pd.DataFrame) -> str:
    dated = messages_with_datetime(chat)
    if dated.empty:
        return "기간을 읽을 수 없습니다."
    start = dated["Date"].min()
    end = dated["Date"].max()
    return f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d}"


def render_sidebar() -> tuple[pd.DataFrame | None, str | None]:
    with st.sidebar:
        st.title("카톡 대화 분석기")
        st.caption("카카오톡 대화를 올려 메시지, 시간, 단어, 답장 속도를 확인합니다.")
        uploaded = st.file_uploader(
            "CSV 파일 업로드",
            type=["csv", "txt"],
            help="Date, User, Message 컬럼이 있는 CSV 또는 카카오톡 대화 txt",
        )
        notice = "왼쪽에서 대화 파일을 업로드하면 분석 결과가 표시됩니다."
        chat = None
        if uploaded is not None:
            notice = None
            try:
                loaded = load_chat(uploaded.name, uploaded.getvalue())
            except ValueError as error:
                st.error(str(error))
            else:
                if loaded.empty:
                    st.warning("파일에서 메시지를 찾지 못했습니다.")
                else:
                    chat = loaded
                    participants = sorted(
                        name for name in chat["User"].astype(str).unique() if str(name).strip()
                    )
                    st.subheader("기본 정보")
                    st.metric("메시지 수", f"{len(chat):,}개")
                    st.markdown("**참여자**")
                    if participants:
                        st.markdown("\n".join(f"- {name}" for name in participants))
                    else:
                        st.caption("참여자 이름을 찾지 못했습니다.")
                    st.markdown("**기간**")
                    st.markdown(chat_period_label(chat))

        st.divider()
        st.subheader("사용 방법")
        st.markdown(
            "1. CSV 또는 카카오톡에서 내보낸 txt를 업로드합니다.\n"
            "2. CSV에는 Date, User, Message 컬럼이 있어야 합니다.\n"
            "3. 본문 탭에서 기본 통계, 시간, 단어, 답장 속도를 확인합니다."
        )
    return chat, notice


def main() -> None:
    st.set_page_config(
        page_title="카톡 대화 분석기",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    chat, notice = render_sidebar()
    if chat is None:
        if notice:
            st.info(notice)
        return

    participants = sorted(name for name in chat["User"].unique() if str(name).strip())
    if not participants:
        st.warning("참여자 이름을 찾지 못했습니다.")
        return

    basic_tab, time_tab, word_tab, reply_tab = st.tabs(
        ["📊 기본 통계", "🕒 시간 분석", "☁️ 단어 분석", "⚡ 답장 속도"]
    )
    with basic_tab:
        render_overview(chat)
    with time_tab:
        render_time_patterns(chat)
    with word_tab:
        render_word_usage(chat)
    with reply_tab:
        render_reply_speed(chat)


if __name__ == "__main__":
    main()
