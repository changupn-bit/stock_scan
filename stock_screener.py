"""
종목 조건 검색기 v4.3
- TradingView 스캐너 + 네이버 일봉/주봉 7이평 거래량 급증
- 120 / 240 장기이평선 배열 (정배열 / 역배열) 선택 옵션 추가
"""
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ───────────────────────── 기본 설정 ─────────────────────────
MA_WINDOWS = [5, 10, 20, 60, 120]
LOOKBACK = 120
BARS = 600
TIMEOUT = 5
MARGIN = 3.0

HERE = Path(__file__).parent
CACHE_DIR = HERE / "cache"
TV_URL = "https://scanner.tradingview.com/korea/scan"
NAVER_URL = "https://fchart.stock.naver.com/sise.nhn"
HEADERS = {"User-Agent": "Mozilla/5.0"}
ITEM_RE = re.compile(r'data="([^"]+)"')

TV_COLS = [
    "name", "description", "close", "market_cap_basic", "typespecs", "industry",
    "return_on_equity", "book_value_per_share_fq",
    "average_volume_10d_calc", "average_volume_30d_calc", "High.6M",
    "SMA3", "SMA5", "SMA10", "SMA20", "SMA60", "SMA120", "SMA240",
]

# ───────────────────────── 7대 테마 ─────────────────────────
THEME_NAMES = {
    "우주": {"한화에어로스페이스", "한화시스템", "한국항공우주", "인텔리안테크", "쎄트렉아이", "AP위성",
            "컨텍", "제노코", "켄코아에어로스페이스", "이노스페이스", "루미르", "LK삼양", "비츠로테크",
            "파이버프로", "덕산하이메탈"},
    "양자": {"엑스게이트", "우리로", "드림시큐리티", "케이씨에스", "코위버", "아이윈플러스", "피피아이",
            "시큐센", "우리넷", "옵티시스", "아이씨티케이", "쏠리드", "라닉스", "에치에프알", "텔레필드",
            "한울소재과학"},
    "로봇": {"두산로보틱스", "레인보우로보틱스", "로보티즈", "뉴로메카", "유일로보틱스", "티로보틱스",
            "휴림로봇", "에스비비테크", "로보스타", "현대무벡스", "클로봇", "씨메스", "하이젠알앤엠",
            "에스피지", "이랜시스", "삼현", "로보로보", "유진로봇", "라온테크", "TPC", "대동기어"},
    "조선": {"HD한국조선해양", "HD현대중공업", "삼성중공업", "한화오션", "HD현대미포", "HJ중공업",
            "STX엔진", "세진중공업", "성광벤드", "태광", "동성화인텍", "한국카본", "하이록코리아",
            "케이에스피", "오리엔탈정공", "일승", "대양전기공업", "인화정공"},
    "건설": {"현대건설", "DL이앤씨", "GS건설", "대우건설", "HDC현대산업개발", "삼성E&A", "금호건설",
            "동부건설", "계룡건설", "코오롱글로벌", "서희건설", "한신공영", "아이에스동서", "신세계건설",
            "HL D&I", "KCC건설"},
}
THEME_INDUSTRIES = {
    "반도체": {"Semiconductors"},
    "방산": {"Aerospace & Defense"},
    "건설": {"Engineering & Construction", "Homebuilding"},
}
THEME_ORDER = ["반도체", "방산", "우주", "양자", "로봇", "조선", "건설"]

INDUSTRY_KO = {
    "Semiconductors": "반도체", "Electronic Production Equipment": "반도체장비",
    "Electronic Components": "전자부품", "Electronic Equipment/Instruments": "전자장비",
    "Electronics/Appliances": "가전", "Electrical Products": "전기장비",
    "Computer Peripherals": "컴퓨터주변기기", "Computer Processing Hardware": "컴퓨터HW",
    "Computer Communications": "네트워크장비", "Telecommunications Equipment": "통신장비",
    "Information Technology Services": "IT서비스", "Packaged Software": "소프트웨어",
    "Internet Software/Services": "인터넷", "Data Processing Services": "데이터서비스",
    "Major Telecommunications": "통신", "Wireless Telecommunications": "무선통신",
    "Specialty Telecommunications": "통신서비스", "Electronics Distributors": "전자유통",
    "Aerospace & Defense": "방산·항공", "Engineering & Construction": "건설",
    "Homebuilding": "주택건설", "Construction Materials": "건자재", "Building Products": "건축자재",
    "Industrial Machinery": "산업기계", "Trucks/Construction/Farm Machinery": "건설·농기계",
    "Industrial Conglomerates": "복합기업", "Metal Fabrication": "금속가공",
    "Miscellaneous Manufacturing": "기타제조", "Tools & Hardware": "공구",
    "Steel": "철강", "Other Metals/Minerals": "비철금속", "Aluminum": "알루미늄",
    "Precious Metals": "귀금속", "Chemicals: Major Diversified": "화학",
    "Chemicals: Specialty": "특수화학", "Chemicals: Agricultural": "농화학",
    "Industrial Specialties": "산업소재", "Containers/Packaging": "포장재",
    "Pulp & Paper": "제지", "Forest Products": "목재", "Textiles": "섬유",
    "Pharmaceuticals: Major": "제약", "Pharmaceuticals: Other": "제약",
    "Pharmaceuticals: Generic": "제네릭", "Biotechnology": "바이오",
    "Medical Specialties": "의료기기", "Medical/Nursing Services": "의료서비스",
    "Hospital/Nursing Management": "병원", "Medical Distributors": "의약품유통",
    "Services to the Health Industry": "헬스케어서비스",
    "Major Banks": "은행", "Regional Banks": "지방은행", "Savings Banks": "저축은행",
    "Financial Conglomerates": "금융지주", "Investment Banks/Brokers": "증권",
    "Investment Managers": "자산운용", "Investment Trusts/Mutual Funds": "투자회사",
    "Life/Health Insurance": "생명보험", "Property/Casualty Insurance": "손해보험",
    "Multi-Line Insurance": "보험", "Insurance Brokers/Services": "보험중개",
    "Specialty Insurance": "보험", "Finance/Rental/Leasing": "여신·리스",
    "Real Estate Development": "부동산개발", "Real Estate Investment Trusts": "리츠",
    "Motor Vehicles": "자동차", "Auto Parts: OEM": "자동차부품",
    "Automotive Aftermarket": "자동차부품", "Marine Shipping": "해운", "Airlines": "항공",
    "Air Freight/Couriers": "물류", "Trucking": "육상운송", "Railroads": "철도",
    "Other Transportation": "운송", "Oil Refining/Marketing": "정유", "Integrated Oil": "석유",
    "Oil & Gas Production": "석유·가스", "Oilfield Services/Equipment": "유전장비",
    "Coal": "석탄", "Electric Utilities": "전력", "Gas Distributors": "가스",
    "Alternative Power Generation": "신재생에너지", "Environmental Services": "환경",
    "Food: Major Diversified": "식품", "Food: Specialty/Candy": "식품",
    "Food: Meat/Fish/Dairy": "식품", "Agricultural Commodities/Milling": "농산물",
    "Beverages: Alcoholic": "주류", "Beverages: Non-Alcoholic": "음료", "Tobacco": "담배",
    "Household/Personal Care": "화장품·생활용품", "Apparel/Footwear": "의류",
    "Apparel/Footwear Retail": "의류유통", "Department Stores": "백화점",
    "Discount Stores": "할인점", "Specialty Stores": "전문소매", "Internet Retail": "온라인쇼핑",
    "Food Retail": "식품유통", "Wholesale Distributors": "도매", "Home Furnishings": "가구",
    "Recreational Products": "레저용품", "Consumer Sundries": "생활잡화",
    "Movies/Entertainment": "엔터", "Broadcasting": "방송", "Cable/Satellite TV": "케이블TV",
    "Advertising/Marketing Services": "광고", "Publishing: Books/Magazines": "출판",
    "Publishing: Newspapers": "신문", "Casinos/Gaming": "카지노·게임",
    "Hotels/Resorts/Cruise lines": "호텔·레저", "Restaurants": "외식",
    "Other Consumer Services": "소비자서비스", "Miscellaneous Commercial Services": "상업서비스",
    "Personnel Services": "인력서비스", "Commercial Printing/Forms": "인쇄",
    "Office Equipment/Supplies": "사무용품", "Miscellaneous": "기타",
}

def clean_name(v) -> str:
    n = str(v or "").strip()
    for x in ("보통주", "(주)", " 주식회사"):
        n = n.replace(x, "")
    return n.strip()

def theme_list(name: str, industry: str, extra: str = "") -> list:
    r = [t for t in THEME_ORDER
         if industry in THEME_INDUSTRIES.get(t, ()) or name in THEME_NAMES.get(t, ())]
    extra = "" if pd.isna(extra) else str(extra)
    for t in extra.split(","):
        t = t.strip()
        if t and t not in r:
            r.append(t)
    return r

# ───────────────────────── UI 설정 ─────────────────────────
st.set_page_config(page_title="종목 조건 검색기", layout="wide")
st.title("📊 종목 조건 검색기")

with st.form("search_form"):
    c = st.columns(7)
    min_cap = c[0].number_input("시총 최소(억원)", value=200, step=100)
    max_cap = c[1].number_input("시총 미만(억원)", value=5_000_000, step=1000)
    min_roe = c[2].number_input("ROE 이상(%)", value=0.0, step=1.0)
    max_dispar = c[3].number_input("이평 이격 미만(%)", value=20.0, step=1.0)
    min_drop = c[4].number_input("고점 하락 이상(%)", value=10.0, step=1.0)
    min_bps = c[5].number_input("BPS 이상(원)", value=0, step=1000)
    ma3_cond = c[6].selectbox("종가와 3일선", ["조건 없음", "위", "아래"])

    c2 = st.columns([3, 2, 2, 2])
    sel_themes = c2[0].multiselect("테마 (비우면 전 종목)", THEME_ORDER, default=[])
    min_vol = c2[1].number_input("최소 거래량(주)", value=100_000, step=10_000,
                                 help="일봉: 20일 평균거래량 / 주봉: 최근 완성 주봉 거래량")
    mode = c2[2].radio("검색 모드", ["⚡ 빠른", "🎯 정밀"], horizontal=True,
                       help="빠른: TradingView 지표만 사용\n정밀: 네이버 일봉/주봉 데이터로 재계산")
    exclude_misc = c2[3].checkbox("우선주·스팩 제외", value=True)

    # ── 거래량 및 장기이평선 배열 상세 설정 ──
    st.markdown("##### 📈 거래량 및 이평선 상세 설정")
    c_vol = st.columns([2, 2, 2, 2, 2])
    timeframe = c_vol[0].selectbox("분석 기준 봉", ["일봉", "주봉"], index=0,
                                  help="주봉 선택 시 네이버 일봉을 주봉으로 리샘플링하여 계산합니다.")
    vol_ratio = c_vol[1].number_input("거래량 / 7이평 배수", value=1.0, min_value=0.0, step=0.1,
                                     help="1.0은 조건 없음. 1.2 입력 시 7이평의 120% 이상, 2.0은 200% 이상.")
    long_ma_align = c_vol[2].selectbox("120 / 240 배열", ["조건 없음", "정배열 (120 > 240)", "역배열 (120 < 240)"],
                                       help="120일(주)선과 240일(주)선의 정배열/역배열 여부를 검사합니다.")
    ma_type = c_vol[3].radio("이평 종류", ["SMA", "WMA", "EMA"], index=1, horizontal=True)
    ma_windows = c_vol[4].multiselect("압축 계산 이평선", [3, 5, 10, 20, 60, 120, 240],
                                      default=[5, 10, 20, 60, 120, 240])

    with st.expander("정밀 모드 옵션"):
        o = st.columns(2)
        use_cache = o[0].checkbox("오늘 받은 데이터 재사용 (캐시)", value=True)
        workers = o[1].slider("동시 조회 수", 2, 12, 6)

    submitted = st.form_submit_button("🔍 검색 실행", type="primary")

ma_windows = sorted(ma_windows) or [5, 10, 20, 60, 120]

# ───────────────────────── TradingView ─────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_tv() -> pd.DataFrame:
    body = {
        "markets": ["korea"],
        "symbols": {"query": {"types": []}, "tickers": []},
        "options": {"lang": "ko"},
        "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
        "columns": TV_COLS,
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
        "range": [0, 6000],
    }
    r = requests.post(TV_URL, json=body, headers=HEADERS, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame([row["d"] for row in r.json().get("data", [])], columns=TV_COLS)

    num = [c for c in TV_COLS if c not in ("name", "description", "typespecs", "industry")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df["코드"] = df["name"].astype(str).str.zfill(6)
    df["종목명"] = df["description"].map(clean_name)
    df["업종"] = df["industry"].fillna("").astype(str)
    df["업종명"] = df["업종"].map(lambda x: INDUSTRY_KO.get(x, x))
    df["시가총액_억"] = df["market_cap_basic"] / 1e8
    df["ROE"] = df["return_on_equity"]
    df["BPS"] = df["book_value_per_share_fq"]
    df["우선주"] = (df["typespecs"].apply(lambda t: isinstance(t, list) and "preferred" in t)
                  | df["description"].astype(str).str.contains("우선주"))

    sma = df[[f"SMA{w}" for w in MA_WINDOWS]]
    df["tv_이격"] = (sma.max(axis=1) - sma.min(axis=1)) / sma.min(axis=1) * 100
    df["tv_고점대비"] = (df["close"] - df["High.6M"]) / df["High.6M"] * 100
    pos = sma.gt(0).all(axis=1) & (df["close"] > 0) & (df["High.6M"] > 0) & (df["SMA3"] > 0)
    sane = (sma.max(axis=1) <= df["High.6M"] * 1.01) & (df["close"] <= df["High.6M"] * 1.01)
    df["tv_정상"] = pos & sane
    return df

@st.cache_data(show_spinner=False)
def load_theme_csv(path: str = "themes.csv") -> pd.DataFrame:
    try:
        t = pd.read_csv(HERE / path, dtype={"코드": str})
        t["코드"] = t["코드"].str.zfill(6)
        return (t.groupby("코드")["테마"]
                 .apply(lambda s: ", ".join(s.dropna().astype(str).unique()))
                 .reset_index().rename(columns={"테마": "csv테마"}))
    except FileNotFoundError:
        return pd.DataFrame(columns=["코드", "csv테마"])

def base_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["시가총액_억"] >= min_cap) & (df["시가총액_억"] < max_cap)]
    df = df[(df["ROE"] >= min_roe) & (df["BPS"] >= min_bps)]
    if exclude_misc:
        df = df[~df["우선주"] & (df["코드"].str[-1] == "0")]
        df = df[~df["종목명"].str.contains("스팩")]
    if sel_themes:
        df = df[df["테마목록"].apply(lambda ts: any(t in ts for t in sel_themes))]
    return df

# ───────────────────────── 네이버 일봉 / 주봉 분석 ─────────────────────────
def fetch_ohlcv(code: str, use_cache: bool):
    cache_file = CACHE_DIR / dt.date.today().strftime("%Y%m%d") / f"{code}_{BARS}.pkl"
    if use_cache and cache_file.exists():
        return pd.read_pickle(cache_file)
    params = {"symbol": code, "timeframe": "day", "count": BARS, "requestType": 0}
    for _ in range(2):
        try:
            r = requests.get(NAVER_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            rows = [x.split("|") for x in ITEM_RE.findall(r.text)]
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").apply(pd.to_numeric, errors="coerce").sort_index()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(cache_file)
            return df
        except Exception:
            continue
    return None

def moving_avg(series: pd.Series, n: int, kind: str) -> float:
    if len(series) < n:
        return np.nan
    if kind == "EMA":
        return float(series.astype(float).ewm(span=n, adjust=False).mean().iloc[-1])
    x = series.tail(n).to_numpy(dtype=float)
    if kind == "WMA":
        w = np.arange(1, n + 1, dtype=float)
        return float(np.dot(x, w) / w.sum())
    return float(x.mean())

def analyze(code: str, use_cache: bool, windows: list, kind: str, tf: str, min_vol_ratio: float, align_req: str):
    try:
        df = fetch_ohlcv(code, use_cache)
        if df is None:
            return None, True
        df = df[(df["Close"] > 0) & (df["High"] > 0)]

        # 주봉 모드일 경우 리샘플링
        if tf == "주봉":
            df = df.resample("W-FRI").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()

        needed_bars = max(LOOKBACK, max(windows), 240, 8)
        if len(df) < needed_bars:
            return None, False

        close = df["Close"]
        vol = df["Volume"]
        price = float(close.iloc[-1])
        curr_vol = float(vol.iloc[-1])

        # 120, 240 장기이평선 계산 및 배열 판정
        ma120 = moving_avg(close, 120, kind)
        ma240 = moving_avg(close, 240, kind)
        if np.isnan(ma120) or np.isnan(ma240):
            return None, False

        align_status = "정배열" if ma120 > ma240 else ("역배열" if ma120 < ma240 else "동일")
        if align_req == "정배열 (120 > 240)" and align_status != "정배열":
            return None, False
        if align_req == "역배열 (120 < 240)" and align_status != "역배열":
            return None, False

        # 거래량 7이평 (단순 평균) 및 배수 계산
        vol_ma7 = float(vol.tail(7).mean())
        vol_multiple = (curr_vol / vol_ma7) if vol_ma7 > 0 else 0.0

        if min_vol_ratio > 0 and vol_multiple < min_vol_ratio:
            return None, False

        ma3 = moving_avg(close, 3, kind)
        mas = [moving_avg(close, w, kind) for w in windows]
        high = float(df["High"].tail(LOOKBACK).max())

        if any(np.isnan(mas)) or min(mas) <= 0 or high <= 0:
            return None, False

        return {
            "코드": code,
            "현재가": price,
            "3일선": ma3,
            "120선": ma120,
            "240선": ma240,
            "120/240배열": align_status,
            "이평 이격": (max(mas) - min(mas)) / min(mas) * 100,
            "120일 고점": high,
            "고점 대비": (price - high) / high * 100,
            "당기 거래량": curr_vol,
            "거래량 7이평": vol_ma7,
            "거래량배수": vol_multiple,
            "평균거래량": float(vol.tail(20).mean()),
        }, False
    except Exception:
        return None, True

# ───────────────────────── 검색 실행 ─────────────────────────
if submitted:
    try:
        with st.spinner("TradingView에서 종목 목록을 불러오는 중..."):
            tv = load_tv().merge(load_theme_csv(), on="코드", how="left")
        tv["테마목록"] = [theme_list(n, i, e) for n, i, e in zip(tv["종목명"], tv["업종"], tv["csv테마"])]
        base = base_filter(tv)
        info = ["코드", "종목명", "업종명", "테마목록", "시가총액_억", "ROE", "BPS"]

        # 7이평 거래량 필터, 장기이평 배열, 주봉 설정, WMA/EMA 선택 시 정밀 모드 자동 전환
        requires_precision = (timeframe == "주봉") or (vol_ratio > 1.0) or (long_ma_align != "조건 없음") \
                             or (ma_type != "SMA") or (set(ma_windows) != set(MA_WINDOWS))
        forced = mode.startswith("⚡") and requires_precision

        if mode.startswith("⚡") and not requires_precision:
            b = base[base["tv_정상"]]
            res = pd.DataFrame({
                "코드": b["코드"], "현재가": b["close"], "3일선": b["SMA3"],
                "120선": b["SMA120"], "240선": b["SMA240"],
                "120/240배열": np.where(b["SMA120"] > b["SMA240"], "정배열", "역배열"),
                "이평 이격": b["tv_이격"], "120일 고점": b["High.6M"],
                "고점 대비": b["tv_고점대비"], "당기 거래량": b["average_volume_10d_calc"],
                "거래량 7이평": b["average_volume_10d_calc"], "거래량배수": 1.0,
                "평균거래량": b["average_volume_30d_calc"],
            })
            scanned, failed, anomalies = len(base), 0, int((~base["tv_정상"]).sum())
        else:
            codes = base["코드"].tolist()
            rows, failed = [], 0
            bar = st.progress(0.0, text=f"{timeframe} 정밀 분석 중... 0/{len(codes)}")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(analyze, c, use_cache, ma_windows, ma_type, timeframe, vol_ratio, long_ma_align) for c in codes]
                for i, fut in enumerate(as_completed(futures), 1):
                    r, fail = fut.result()
                    failed += fail
                    if r is not None:
                        rows.append(r)
                    if i % 10 == 0 or i == len(codes):
                        bar.progress(i / max(len(codes), 1), text=f"{timeframe} 정밀 분석 중... {i}/{len(codes)} (실패 {failed})")
            bar.empty()
            res = pd.DataFrame(rows)
            scanned, anomalies = len(codes), 0

        if not res.empty:
            res["3일선 위치"] = np.where(res["현재가"] >= res["3일선"], "위", "아래")
            cond = ((res["이평 이격"] < max_dispar)
                    & (-res["고점 대비"] >= min_drop)
                    & (res["평균거래량"] >= min_vol))
            if ma3_cond != "조건 없음":
                cond &= res["3일선 위치"] == ma3_cond
            res = res[cond]

        if not res.empty:
            res = res.merge(base[info], on="코드")
            res["테마"] = [", ".join(t) if t else i for t, i in zip(res["테마목록"], res["업종명"])]
            res["차트"] = "https://kr.tradingview.com/chart/?symbol=KRX%3A" + res["코드"]
            res = res.sort_values(["거래량배수", "고점 대비"], ascending=[False, True]).reset_index(drop=True)

            for col in ["현재가", "3일선", "120선", "240선", "시가총액_억", "BPS", "120일 고점", "당기 거래량", "거래량 7이평"]:
                res[col] = res[col].round(0).astype("Int64")
            res.insert(0, "순위", range(1, len(res) + 1))
            res = res[["순위", "테마", "종목명", "코드", "현재가", "3일선", "3일선 위치",
                       "120/240배열", "120선", "240선",
                       "거래량배수", "당기 거래량", "거래량 7이평",
                       "시가총액_억", "ROE", "BPS", "이평 이격", "120일 고점", "고점 대비", "차트"]]

        if forced:
            mode = f"🎯 정밀 (조건에 맞추어 자동 전환: {timeframe} / 배열: {long_ma_align})"
        st.session_state.update(result=res, total=len(tv), base=len(base), scanned=scanned,
                                failed=failed, anomalies=anomalies, mode=mode,
                                at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    except Exception as e:
        st.error(f"오류가 발생했습니다: {type(e).__name__}: {e}")

# ───────────────────────── 결과 표시 ─────────────────────────
res = st.session_state.get("result")
if res is not None:
    ss = st.session_state
    if ss.get("failed"):
        st.info(f"데이터 조회 실패 {ss['failed']}건 제외됨.")
    if res.empty:
        st.warning("조건에 일치하는 종목이 없습니다. 장기이평 배열이나 거래량 배수 기준을 조정해 보세요.")
    else:
        l, r = st.columns([8, 2])
        l.write(f"{ss['mode']} · 전체 {ss['total']:,} → 기본조건 {ss['base']:,} → **최종 {len(res)}건** ({ss['at']})")
        r.download_button(
            "💾 CSV 저장",
            res.drop(columns=["차트"]).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"stock_search_{dt.date.today():%Y%m%d}.csv",
            mime="text/csv",
        )
        cfg = st.column_config
        table_kw = dict(hide_index=True, height=600, column_config={
            "현재가": cfg.NumberColumn("현재가(원)", format="localized"),
            "3일선": cfg.NumberColumn("3일선(원)", format="localized"),
            "120선": cfg.NumberColumn("120선(원)", format="localized"),
            "240선": cfg.NumberColumn("240선(원)", format="localized"),
            "거래량배수": cfg.NumberColumn("7이평 대비(배)", format="%.2f배"),
            "당기 거래량": cfg.NumberColumn("현재 거래량(주)", format="localized"),
            "거래량 7이평": cfg.NumberColumn("거래량 7이평(주)", format="localized"),
            "시가총액_억": cfg.NumberColumn("시가총액(억)", format="localized"),
            "ROE": cfg.NumberColumn(format="%.1f%%"),
            "BPS": cfg.NumberColumn("BPS(원)", format="localized"),
            "이평 이격": cfg.NumberColumn(format="%.2f%%"),
            "120일 고점": cfg.NumberColumn("120일 고점(원)", format="localized"),
            "고점 대비": cfg.NumberColumn(format="%.1f%%"),
            "차트": cfg.LinkColumn("차트", display_text="📈 열기"),
        })
        try:
            st.dataframe(res, width="stretch", **table_kw)
        except TypeError:
            st.dataframe(res, use_container_width=True, **table_kw)