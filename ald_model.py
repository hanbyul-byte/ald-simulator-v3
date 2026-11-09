"""ALD HfOx 웨이퍼 두께 모델 · 피팅 · Zernike 유틸리티 (Streamlit과 독립적으로 동작)."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R = 8.314
# 9-point 좌표 [cm], Gas inlet = 원점, 흐름 방향 = +y
PTS = np.array([[0, 10.5], [-1.5, 9], [1.5, 9], [-3, 7.5], [0, 7.5], [3, 7.5], [-1.5, 6], [1.5, 6], [0, 4.5]])
PT_NAMES = ["P1 원거리", "P2 좌상", "P3 우상", "P4 좌", "P5 중심", "P6 우", "P7 좌하", "P8 우하", "P9 근거리"]
WC = np.array([0.0, 7.5])   # 웨이퍼 중심
WR = 4.5                    # 웨이퍼 반경 [cm]


# ---------------------------------------------------------------- 모델 정의
# 각 파라미터: (기본값, 하한, 상한, 단위, 이름, 설명)
FRONT_SPEC = {
    "H0":  (16.3827, 5.0, 30.0, "nm", "100 °C 포화 두께 H₀", "논문 사이클 수 기준 총 두께"),
    "b":   (0.00317399, 0.0, 0.01, "1/K", "포화 두께 감소 계수 b", "H_sat(T) = H₀·exp(−b(T−100))"),
    "Lam": (7.64757, 0.5, 80.0, "cm", "front 길이 Λ (전구체 55 °C)", "공급 분자 수 / 흡착 자리"),
    "a0":  (0.795441, 1e-3, 50.0, "1/cm", "흡착 속도 a (100 °C)", "front의 날카로움. 작을수록 exp 고갈에 가까움"),
    "Ea":  (15.6545, -100.0, 200.0, "kJ/mol", "흡착 활성화 에너지 Eₐ", "a(T)의 Arrhenius 온도 의존"),
    "dH":  (31.7117, 0.0, 300.0, "kJ/mol", "증발 엔탈피 ΔH", "전구체 증기압의 Clausius–Clapeyron 기울기"),
    "xs":  (-0.253549, -5.0, 5.0, "cm", "jet 중심 x_s", "경험항: 흐름의 좌우 치우침"),
    "w":   (4.62358, 0.5, 30.0, "cm", "jet 폭 w", "경험항: 유선별 공급량의 가우시안 폭"),
    "f0":  (0.0451213, 0.0, 0.9, "", "바닥 성장 비율 f₀", "경험항: front 밖에서의 성장"),
}
OLD_SPEC = {
    "H0":  (15.1155, 5.0, 30.0, "nm", "100 °C 포화 두께 H₀", ""),
    "b":   (0.00218012, 0.0, 0.01, "1/K", "포화 두께 감소 계수 b", ""),
    "k0":  (64.3773, 1e-3, 1e4, "", "흡착 노출 계수 k (100 °C)", ""),
    "Ea":  (28.1623, -100.0, 200.0, "kJ/mol", "활성화 에너지 Eₐ", ""),
    "dH":  (290.941, 0.0, 300.0, "kJ/mol", "증발 엔탈피 ΔH", ""),
    "L":   (1.5601, 0.5, 100.0, "cm", "고갈 길이 L_dep", "P(d) = P₀·exp(−d/L_dep)"),
}


def _hsat(T, H0, b):
    return H0 * np.exp(-b * (T - 100.0))


def _pv(Tp, dH):            # 증기압 비, 55 °C 기준
    return np.exp(-dH * 1e3 / R * (1.0 / (Tp + 273.15) - 1.0 / 328.15))


def _arr(T, E):             # Arrhenius 비, 100 °C 기준
    return np.exp(-E * 1e3 / R * (1.0 / (T + 273.15) - 1.0 / 373.15))


def front_model(p, X, Y, T, Tp, dose=1.0):
    """포화 front + 흐름 jet. 반환: (두께 nm, 포화도 θ, 포화 두께 h)."""
    h = _hsat(T, p["H0"], p["b"])
    a = p["a0"] * _arr(T, p["Ea"])
    Lf = p["Lam"] * _pv(Tp, p["dH"]) * (p["H0"] / h) * dose * np.exp(-(X - p["xs"]) ** 2 / (2 * p["w"] ** 2))
    u0 = np.clip(a * Lf, 1e-9, 60.0)
    E = np.expm1(u0) * np.exp(-a * Y)
    th = E / (1.0 + E)
    return h * (p["f0"] + (1 - p["f0"]) * th), th, h


def old_model(p, X, Y, T, Tp, dose=1.0):
    """기존 모델: Inlet 거리 d만의 함수, P(d) = P₀·exp(−d/L)."""
    h = _hsat(T, p["H0"], p["b"])
    k = p["k0"] * _arr(T, p["Ea"])
    d = np.hypot(X, Y)
    th = 1 - np.exp(-k * _pv(Tp, p["dH"]) * dose * np.exp(-d / p["L"]) * (p["H0"] / h))
    return h * th, th, h


MODELS = {
    "front": {"name": "포화 front + 흐름 jet", "fn": front_model, "spec": FRONT_SPEC},
    "old": {"name": "기존 모델 (exp 고갈)", "fn": old_model, "spec": OLD_SPEC},
}


def defaults(key):
    return {k: v[0] for k, v in MODELS[key]["spec"].items()}


def front_line(p, T, Tp, dose, xs):
    """front 모델에서 θ = 0.5가 되는 y 위치 (유선 x별)."""
    h = _hsat(T, p["H0"], p["b"])
    a = p["a0"] * _arr(T, p["Ea"])
    Lf = p["Lam"] * _pv(Tp, p["dH"]) * (p["H0"] / h) * dose * np.exp(-(xs - p["xs"]) ** 2 / (2 * p["w"] ** 2))
    u0 = np.clip(a * Lf, 1e-9, 60.0)
    with np.errstate(divide="ignore"):
        return np.log(np.expm1(u0)) / a


# ---------------------------------------------------------------- 논문 데이터
# 판독값. 9점 순서 = PTS 순서. 출처: 사용자가 제공한 논문 Fig. 1b (전구체 55 °C 행), Fig. 1a (중심점)
FIG1B_55 = {
    100: [3.2, 3.1, 2.8, 5.2, 8.4, 5.3, 15.2, 12.2, 15.5],
    125: [3.3, 3.5, 2.7, 5.2, 7.7, 4.2, 14.3, 9.8, 14.3],
    150: [3.4, 3.3, 2.8, 7.5, 13.1, 6.5, 13.6, 13.5, 13.8],
    175: [4.4, 9.2, 6.8, 10.4, 12.3, 5.8, 12.4, 12.3, 12.3],
    200: [8.0, 10.6, 11.0, 12.0, 12.4, 11.1, 12.4, 12.2, 12.4],
    225: [10.8, 11.4, 11.3, 11.1, 11.2, 10.8, 11.3, 11.2, 11.2],
    250: [9.8, 11.7, 11.5, 10.7, 11.1, 10.6, 11.0, 10.9, 11.1],
}
FIG1A_P75 = {100: 15.3, 125: 14.7, 150: 13.3, 175: 12.9, 200: 11.8, 225: 11.4, 250: 10.5}
FIG1A_T100 = {55: 7.0, 60: 14.0, 65: 14.6, 70: 15.5, 75: 15.3}

DATA_COLS = ["source", "Tp", "T", "x", "y", "t_ox"]


def paper_data():
    rows = []
    for T, v in FIG1B_55.items():
        for (x, y), t in zip(PTS, v):
            rows.append(("Fig1b", 55, T, x, y, t))
    for T, t in FIG1A_P75.items():
        rows.append(("Fig1a-left", 75, T, 0.0, 7.5, t))
    for Tp, t in FIG1A_T100.items():
        rows.append(("Fig1a-right", Tp, 100, 0.0, 7.5, t))
    return pd.DataFrame(rows, columns=DATA_COLS)


def clean_data(df):
    df = df.copy()
    for c in DATA_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["source"] = df["source"].fillna("user").astype(str)
    return df.dropna(subset=DATA_COLS[1:]).reset_index(drop=True)


def predict(key, p, df, dose=1.0):
    fn = MODELS[key]["fn"]
    return fn(p, df["x"].to_numpy(float), df["y"].to_numpy(float),
              df["T"].to_numpy(float), df["Tp"].to_numpy(float), dose)[0]


# ---------------------------------------------------------------- 피팅 · 검증
def fit(key, df, p0=None, mask=None):
    spec = MODELS[key]["spec"]
    names = list(spec)
    p0 = p0 or defaults(key)
    x0 = np.array([np.clip(p0[n], spec[n][1], spec[n][2]) for n in names])
    lb = np.array([spec[n][1] for n in names])
    ub = np.array([spec[n][2] for n in names])
    sub = df if mask is None else df[mask]
    obs = sub["t_ox"].to_numpy(float)

    def res(x):
        return predict(key, dict(zip(names, x)), sub) - obs

    r = least_squares(res, x0, bounds=(lb, ub))
    p = dict(zip(names, r.x))
    return p, float(np.sqrt(np.mean(r.fun ** 2)))


def rmse(key, p, df):
    if len(df) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((predict(key, p, df) - df["t_ox"].to_numpy(float)) ** 2)))


def loto_cv(key, df, p0=None):
    """증착 온도 하나를 통째로 빼고 피팅 → 빠진 온도 예측 RMSE."""
    out = []
    for T in sorted(df["T"].unique()):
        te = (df["T"] == T).to_numpy()
        if te.all() or (~te).sum() < len(MODELS[key]["spec"]) + 2:
            continue
        p, _ = fit(key, df, p0, mask=~te)
        out.append({"held_out_T": T, "n_points": int(te.sum()), "rmse": rmse(key, p, df[te])})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Zernike
def zernike_modes(order):
    return [(n, m) for n in range(order + 1) for m in range(-n, n + 1, 2)]


def _radial(n, m, rho):
    m = abs(m)
    out = np.zeros_like(rho, dtype=float)
    from math import factorial
    for k in range((n - m) // 2 + 1):
        c = (-1) ** k * factorial(n - k) / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
        out += c * rho ** (n - 2 * k)
    return out


def zernike(n, m, rho, phi):
    Rnm = _radial(n, m, rho)
    return Rnm * (np.cos(m * phi) if m >= 0 else np.sin(-m * phi))


def to_polar(X, Y):
    dx, dy = np.asarray(X) - WC[0], np.asarray(Y) - WC[1]
    return np.hypot(dx, dy) / WR, np.arctan2(dy, dx)


def zernike_fit(X, Y, vals, order):
    rho, phi = to_polar(X, Y)
    modes = zernike_modes(order)
    A = np.stack([zernike(n, m, rho, phi) for n, m in modes], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.asarray(vals, float), rcond=None)
    return modes, coef


def zernike_eval(modes, coef, X, Y):
    rho, phi = to_polar(X, Y)
    return sum(c * zernike(n, m, rho, phi) for (n, m), c in zip(modes, coef))


# ---------------------------------------------------------------- 격자 · 데이터셋
def wafer_grid(res):
    g = np.linspace(-WR, WR, res)
    X, Y = np.meshgrid(g + WC[0], g + WC[1])
    mask = np.hypot(X - WC[0], Y - WC[1]) <= WR
    return X, Y, mask


def nine_point_stats(v):
    v = np.asarray(v, float)
    mean = v.mean()
    return {"mean": mean, "nu_sigma": v.std(ddof=1) / mean * 100, "nu_range": (v.max() - v.min()) / (2 * mean) * 100}


def make_dataset(key, p, n, res, T_rng, Tp_rng, dose_rng, seed=0):
    """라틴 하이퍼큐브로 공정 조건을 뽑아 2D 두께 맵 데이터셋 생성 (3단계 학습용)."""
    rng = np.random.default_rng(seed)
    u = (np.stack([rng.permutation(n) for _ in range(3)], 1) + rng.random((n, 3))) / n
    T = T_rng[0] + u[:, 0] * (T_rng[1] - T_rng[0])
    Tp = Tp_rng[0] + u[:, 1] * (Tp_rng[1] - Tp_rng[0])
    dose = np.exp(np.log(dose_rng[0]) + u[:, 2] * (np.log(dose_rng[1]) - np.log(dose_rng[0])))
    X, Y, mask = wafer_grid(res)
    fn = MODELS[key]["fn"]
    maps = np.stack([np.where(mask, fn(p, X, Y, t, tp, d)[0], 0.0) for t, tp, d in zip(T, Tp, dose)]).astype(np.float32)
    cond = np.stack([T, Tp, dose], 1).astype(np.float32)
    return {"cond": cond, "cond_names": np.array(["T_dep_C", "T_prec_C", "dose_mult"]),
            "maps": maps, "mask": mask, "x_cm": X[0].astype(np.float32), "y_cm": Y[:, 0].astype(np.float32)}


# ---------------------------------------------------------------- 역설계 (목표 사양 → 공정 조건 추천)
# 모델 계산이 빠르므로 후보 조건을 촘촘한 격자로 전부 평가한다 (전수 탐색이라 격자 해상도 안에서는 최적이 보장됨).
OBJECTIVES = {"nu": "불균일도 최소", "usage": "전구체 사용량 최소", "temp": "증착 온도 최소"}


def eval_points(region):
    """평가 지점: 9-point(측정 근거) 또는 웨이퍼 전체 격자(가장자리는 외삽)."""
    if region == "9pt":
        return PTS.astype(float)
    g = np.linspace(-WR, WR, 13)
    X, Y = np.meshgrid(g + WC[0], g + WC[1])
    m = np.hypot(X - WC[0], Y - WC[1]) <= WR + 1e-9
    return np.stack([X[m], Y[m]], 1)


def candidate_grid(region, T_rng, Tp_rng, dose_rng, dose_free):
    fine = region == "9pt"
    T = np.arange(T_rng[0], T_rng[1] + 1e-9, 1.0 if fine else 2.0)
    Tp = np.arange(Tp_rng[0], Tp_rng[1] + 1e-9, 0.5 if fine else 1.0)
    if dose_free:
        n = 21 if fine else 13
        D = np.exp(np.linspace(np.log(dose_rng[0]), np.log(dose_rng[1]), n))
    else:
        D = np.array([1.0])
    TT, PP, DD = np.meshgrid(T, Tp, D, indexing="ij")
    return TT.ravel(), PP.ravel(), DD.ravel(), (T, Tp, D)


def evaluate_candidates(key, p, T, Tp, dose, pts, chunk=20000):
    fn = MODELS[key]["fn"]
    out = {k: np.empty(len(T)) for k in ("mean", "nu", "nur", "thmin", "hsat")}
    X, Y = pts[:, 0][None, :], pts[:, 1][None, :]
    for s in range(0, len(T), chunk):
        sl = slice(s, s + chunk)
        z, th, h = fn(p, X, Y, T[sl, None], Tp[sl, None], dose[sl, None])
        mean = z.mean(1)
        out["mean"][sl] = mean
        out["nu"][sl] = z.std(1, ddof=1) / mean * 100
        out["nur"][sl] = (z.max(1) - z.min(1)) / (2 * mean) * 100
        out["thmin"][sl] = th.min(1)
        out["hsat"][sl] = np.broadcast_to(h, z.shape)[:, 0]
    return out


def measured_conditions(df, min_points=9):
    g = df.groupby(["Tp", "T"]).size()
    return np.array([k for k, n in g.items() if n >= min_points], float).reshape(-1, 2)


def data_support(Tp, T, dose, cyc, meas):
    """'측정 맵 인근' = dose·사이클이 논문 조건이고 9점 맵이 측정된 조건과 가까움. 그 외는 외삽."""
    ok = (np.abs(dose - 1) < 1e-6) & (np.abs(cyc - 1) < 1e-6)
    if len(meas) == 0:
        return np.zeros_like(T, bool)
    near = ((np.abs(Tp[:, None] - meas[None, :, 0]) <= 2.5) & (np.abs(T[:, None] - meas[None, :, 1]) <= 12.5)).any(1)
    return ok & near


def measured_nu_floor(df):
    """측정된 9점 맵 중 가장 균일한 맵의 불균일도 1σ [%] → 모델이 다루지 않는 장비 고유 편차의 추정치."""
    best = None
    for (tp, t), g in df.groupby(["Tp", "T"]):
        if len(g) >= 9:
            v = g["t_ox"].to_numpy(float)
            nu = v.std(ddof=1) / v.mean() * 100
            if best is None or nu < best[0]:
                best = (nu, tp, t)
    return best   # (불균일도 %, 전구체 °C, 증착 °C) 또는 None


def pareto_mask(cost_x, cost_y):
    """두 값 모두 작을수록 좋은 Pareto 최적 집합."""
    order = np.lexsort((cost_y, cost_x))
    keep = np.zeros(len(cost_x), bool)
    best = np.inf
    for i in order:
        if cost_y[i] < best - 1e-4:     # 1e-4 % 미만 차이는 계산 잡음
            keep[i] = True
            best = cost_y[i]
    return keep


def inverse_design(key, p, target, tol, nu_max, theta_min, objective="nu", region="9pt",
                   T_rng=(100, 250), Tp_rng=(50, 80), dose_rng=(0.3, 3.0), dose_free=False,
                   cyc_free=False, cyc_rng=(0.3, 3.0), meas=None, n_top=5, nu_floor=0.0):
    T, Tp, D, axes = candidate_grid(region, T_rng, Tp_rng, dose_rng, dose_free)
    pts = eval_points(region)
    ev = evaluate_candidates(key, p, T, Tp, D, pts)
    # 모델 밖 편차(히터 온도 분포 등)를 제곱합으로 더한다
    ev["nu_model"] = ev["nu"]
    ev["nu"] = np.sqrt(ev["nu"] ** 2 + nu_floor ** 2)
    # 사이클 수: ALD 선형 성장 가정 → 두께 ∝ 사이클 (불균일도·포화도는 사이클과 무관)
    if cyc_free:
        cyc = np.clip(target / ev["mean"], cyc_rng[0], cyc_rng[1])
    else:
        cyc = np.ones_like(T)
    mean = ev["mean"] * cyc
    usage = D * _pv(Tp, p["dH"]) * cyc          # 논문 55 °C · dose 1 · 사이클 1 = 1
    c_thk = np.abs(mean - target) <= tol
    c_nu = ev["nu"] <= nu_max
    c_th = ev["thmin"] >= theta_min
    feas = c_thk & c_nu & c_th
    df = pd.DataFrame({"T": T, "Tp": Tp, "dose": D, "cyc": cyc, "mean": mean, "nu": ev["nu"], "nu_model": ev["nu_model"], "nur": ev["nur"],
                       "thmin": ev["thmin"], "usage": usage, "ok_thk": c_thk, "ok_nu": c_nu, "ok_th": c_th, "feasible": feas})
    df["support"] = data_support(Tp, T, D, cyc, meas if meas is not None else np.zeros((0, 2)))
    # 순위: 모델 오차(수 %)보다 훨씬 작은 차이는 같은 값으로 본다 → 불균일도 0.1 %, 사용량 1 % 단위로 반올림
    nu_r = np.round(df["nu"], 1)
    use_r = np.round(np.log(df["usage"]) / np.log(1.01))
    extrap = np.abs(np.log(df["dose"])) + np.abs(np.log(df["cyc"])) + (~df["support"]) * 0.5   # 논문 조건에 가까울수록 우선
    keys = {"nu": [nu_r, use_r, df["T"], extrap], "usage": [use_r, nu_r, df["T"], extrap],
            "temp": [df["T"], nu_r, use_r, extrap]}[objective]
    if feas.any():
        idx = np.flatnonzero(feas)
        status = "ok"
    else:   # 만족하는 조건이 없으면 제약 위반이 가장 작은 절충안
        viol = (np.maximum(0, np.abs(mean - target) - tol) / max(tol, 1e-9)
                + np.maximum(0, ev["nu"] - nu_max) / max(nu_max, 1e-9)
                + np.maximum(0, theta_min - ev["thmin"]) / max(theta_min, 1e-2))
        idx = np.arange(len(df))
        keys = [np.round(viol, 3)] + keys
        status = "none"
    order = idx[np.lexsort([np.asarray(k)[idx] for k in reversed(keys)])]
    # 서로 다른 추천만 고른다: 모델에서 전구체 온도와 dose는 '공급량'으로 합쳐지므로 (온도, 공급량)으로 구분
    picks = []
    for i in order:
        r = df.iloc[i]
        if all(abs(r["T"] - q["T"]) >= 15 or abs(np.log(r["usage"] / q["usage"])) >= 0.15 for q in picks):
            picks.append(r)
        if len(picks) >= n_top:
            break
    top = pd.DataFrame(picks) if picks else df.iloc[:0]
    thk_ok = df[c_thk & c_th]
    pm = pareto_mask(thk_ok["usage"].to_numpy(), thk_ok["nu"].to_numpy()) if len(thk_ok) else np.zeros(0, bool)
    return {"status": status, "top": top.reset_index(drop=True), "all": df, "pareto": thk_ok[pm].sort_values("usage"),
            "axes": axes, "n_points": len(pts), "n_cand": len(df), "n_feas": int(feas.sum())}
