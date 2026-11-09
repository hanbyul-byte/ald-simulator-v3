"""HfOx ALD 웨이퍼 두께 시뮬레이터 (Streamlit).

실행:  pip install -r requirements.txt
       streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ald_model as M

st.set_page_config(page_title="ALD 시뮬레이터", page_icon="🧪", layout="wide")

# ------------------------------------------------------------------ 상태 초기화
if "data_base" not in st.session_state:
    st.session_state.data_base = M.paper_data()
    st.session_state.data_ver = 0
for key in M.MODELS:
    for name, sp in M.MODELS[key]["spec"].items():
        st.session_state.setdefault(f"p_{key}_{name}", float(sp[0]))
st.session_state.setdefault("Tp", 55.0)
st.session_state.setdefault("T", 150)
st.session_state.setdefault("dose", 1.0)
st.session_state.setdefault("cyc", 1.0)
st.session_state.setdefault("fit_msg", "")


def params(key):
    return {n: float(st.session_state[f"p_{key}_{n}"]) for n in M.MODELS[key]["spec"]}


def set_params(key, p):
    spec = M.MODELS[key]["spec"]
    for n, v in p.items():
        st.session_state[f"p_{key}_{n}"] = float(np.clip(v, spec[n][1], spec[n][2]))


def fmt_for(v):
    return "%.5f" if abs(v) < 0.1 else "%.3f"


# ------------------------------------------------------------------ 탭 (데이터 탭을 먼저 그려 편집값을 다른 탭에 반영)
st.title("HfOx ALD 웨이퍼 두께 시뮬레이터")
st.caption("TDMAH 전구체 · Cross-flow 반응기 · 9-point 측정 · 파라미터는 논문 그림 판독 데이터로 피팅한 값 (사전 이론값 아님)")

tab_map, tab_inv, tab_curve, tab_fit, tab_zern, tab_data = st.tabs(
    ["웨이퍼 맵", "역설계", "공정 곡선", "피팅 · 검증", "Zernike 분해", "데이터"])

with tab_data:
    st.subheader("측정 데이터")
    st.write("논문 판독값이 들어 있습니다. 표에서 직접 수정·추가하거나 CSV를 올릴 수 있습니다. "
             "열: `source, Tp(전구체 °C), T(증착 °C), x(cm), y(cm), t_ox(nm)` · 좌표는 Gas inlet 원점, 흐름 방향 +y.")
    up = st.file_uploader("CSV 추가", type="csv")
    if up is not None and st.session_state.get("last_upload") != up.file_id:
        try:
            new = M.clean_data(pd.read_csv(up)[M.DATA_COLS])
            st.session_state.data_base = pd.concat([st.session_state.data_base, new], ignore_index=True)
            st.session_state.data_ver += 1
            st.session_state.last_upload = up.file_id
            st.success(f"{len(new)}개 행을 추가했습니다.")
        except Exception as e:  # noqa: BLE001
            st.error(f"CSV를 읽지 못했습니다: {e}")
    edited = st.data_editor(st.session_state.data_base, num_rows="dynamic",
                            key=f"editor_{st.session_state.data_ver}", height=320)
    c1, c2 = st.columns(2)
    if c1.button("논문 데이터로 초기화"):
        st.session_state.data_base = M.paper_data()
        st.session_state.data_ver += 1
        st.rerun()
    DATA = M.clean_data(edited)
    c2.download_button("데이터 CSV 내려받기", DATA.to_csv(index=False).encode("utf-8-sig"),
                       "ald_data.csv", "text/csv")
    st.caption(f"유효 데이터 {len(DATA)}점 · 조건 {DATA[['Tp', 'T']].drop_duplicates().shape[0]}개")

# ------------------------------------------------------------------ 사이드바
CONDS = (DATA.groupby(["Tp", "T"]).size().reset_index(name="n")
         .query("n >= 9").sort_values(["Tp", "T"]))


def jump_condition():
    sel = st.session_state.jump
    if sel and sel != "—":
        tp, t = sel.split(" / ")
        st.session_state.Tp = float(tp.split()[1])
        st.session_state.T = int(float(t.split()[1]))


with st.sidebar:
    st.header("공정 조건")
    model_key = st.radio("모델", list(M.MODELS), format_func=lambda k: M.MODELS[k]["name"])
    opts = ["—"] + [f"전구체 {int(r.Tp)} °C / 증착 {int(r.T)} °C" for r in CONDS.itertuples()]
    st.selectbox("측정 데이터가 있는 조건으로 이동", opts, key="jump", on_change=jump_condition)
    Tp = st.slider("전구체(캐니스터) 온도 [°C]", 50.0, 80.0, step=0.5, key="Tp")
    T = st.slider("증착 온도 [°C]", 100, 250, step=1, key="T")
    dose = st.number_input("Dose 배율 (논문 조건 = 1)", 0.1, 10.0, step=0.1, format="%.3f", key="dose")
    cyc = st.number_input("사이클 배율 (논문 = 1)", 0.2, 5.0, step=0.05, format="%.3f", key="cyc",
                          help="ALD 선형 성장 가정: 두께 ∝ 사이클 수. 1이 아니면 실측과의 비교는 참고용입니다.")

    st.header("모델 파라미터")
    with st.expander(f"{M.MODELS[model_key]['name']} · {len(M.MODELS[model_key]['spec'])}개", expanded=False):
        for n, (v0, lo, hi, unit, label, hlp) in M.MODELS[model_key]["spec"].items():
            st.number_input(f"{label} [{unit}]" if unit else label, min_value=float(lo), max_value=float(hi),
                            key=f"p_{model_key}_{n}", format=fmt_for(v0), help=hlp or None)
        if st.button("사전 피팅값으로 되돌리기"):
            set_params(model_key, M.defaults(model_key))
            st.rerun()

    st.header("표시")
    paper_scale = st.checkbox("논문 색 범위 고정 (3–18 nm)", value=True)
    show_front = st.checkbox("포화 front 선 표시", value=True, disabled=model_key != "front")

P = params(model_key)
FN0 = M.MODELS[model_key]["fn"]


def FN(p, X, Y, T_, Tp_, d):
    """사이클 배율을 반영한 모델 (두께·포화 두께만 배율, 포화도는 그대로)."""
    z, th, h = FN0(p, X, Y, T_, Tp_, d)
    return z * cyc, th, h * cyc


# ------------------------------------------------------------------ 공통 계산
px, py = M.PTS[:, 0], M.PTS[:, 1]
pred9, th9, hsat = FN(P, px, py, T, Tp, dose)
obs_df = DATA[(DATA.Tp == Tp) & (DATA["T"] == T)]
obs9 = np.full(9, np.nan)
for i, (x, y) in enumerate(M.PTS):
    hit = obs_df[np.isclose(obs_df.x, x) & np.isclose(obs_df.y, y)]
    if len(hit):
        obs9[i] = hit.t_ox.mean()
has_obs = ~np.isnan(obs9)
stats = M.nine_point_stats(pred9)
CMAP = [[0, "#d7191c"], [0.25, "#fdae61"], [0.5, "#ffffbf"], [0.7, "#abdda4"], [0.85, "#2b83ba"], [1, "#253494"]]

# ------------------------------------------------------------------ 웨이퍼 맵
with tab_map:
    k = st.columns(6)
    k[0].metric("평균 두께 (9점)", f"{stats['mean']:.2f} nm")
    k[1].metric("불균일도 1σ (모델)", f"{stats['nu_sigma']:.1f} %")
    k[2].metric("(max−min)/2·mean", f"±{stats['nu_range']:.1f} %")
    k[3].metric("포화 두께 H_sat(T)", f"{float(hsat):.2f} nm")
    k[4].metric("최소 포화도 θ", f"{float(np.min(th9)):.2f}")
    if has_obs.any():
        r = float(np.sqrt(np.nanmean((pred9 - obs9) ** 2)))
        k[5].metric(f"실측 대비 RMSE ({has_obs.sum()}점)", f"{r:.2f} nm")
    else:
        k[5].metric("실측 대비 RMSE", "데이터 없음")

    left, right = st.columns([1.35, 1])
    with left:
        X, Y, mask = M.wafer_grid(161)
        Z, TH, _ = FN(P, X, Y, T, Tp, dose)
        Z = np.where(mask, Z, np.nan)
        zr = dict(zmin=3, zmax=18) if paper_scale else {}
        fig = go.Figure(go.Heatmap(x=X[0], y=Y[:, 0], z=Z, colorscale=CMAP, colorbar=dict(title="nm"),
                                   hovertemplate="x %{x:.2f} cm<br>y %{y:.2f} cm<br>두께 %{z:.2f} nm<extra></extra>",
                                   **zr))
        fig.add_shape(type="circle", x0=-M.WR, y0=M.WC[1] - M.WR, x1=M.WR, y1=M.WC[1] + M.WR,
                      line=dict(color="#333", width=1.5))
        if show_front and model_key == "front":
            xs = np.linspace(-M.WR, M.WR, 121)
            yf = M.front_line(P, T, Tp, dose, xs)
            inside = np.isfinite(yf) & (np.hypot(xs - M.WC[0], yf - M.WC[1]) <= M.WR)
            if inside.any():
                fig.add_trace(go.Scatter(x=xs[inside], y=yf[inside], mode="lines", name="front (θ=0.5)",
                                         line=dict(color="white", width=2, dash="dash"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=px, y=py, mode="markers", showlegend=False,
                                 marker=dict(size=9, color="white", line=dict(color="black", width=1.5)),
                                 customdata=np.stack([pred9, obs9], 1), text=M.PT_NAMES,
                                 hovertemplate="%{text}<br>예측 %{customdata[0]:.2f} nm<br>실측 %{customdata[1]:.2f} nm<extra></extra>"))
        for i in range(9):
            txt = f"<b>{pred9[i]:.1f}</b>" + (f"<br><span style='color:#555'>({obs9[i]:.1f})</span>" if has_obs[i] else "")
            fig.add_annotation(x=px[i], y=py[i], text=txt, showarrow=False, yshift=22 if has_obs[i] else 15,
                               bgcolor="rgba(255,255,255,0.85)", borderpad=2, font=dict(size=11, color="#111"))
        fig.add_shape(type="rect", x0=-0.6, y0=-0.1, x1=0.6, y1=0.35, fillcolor="#222", line_width=0)
        fig.add_annotation(x=0, y=2.4, ax=0, ay=0.5, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowwidth=2.5, arrowcolor="#777", text="")
        fig.add_annotation(x=1.1, y=0.1, text="Gas inlet", showarrow=False, xanchor="left", font=dict(size=12))
        fig.update_layout(height=640, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)",
                          title=f"전구체 {Tp} °C · 증착 {T} °C · dose ×{dose:g}" + (f" · 사이클 ×{cyc:g}" if cyc != 1 else "")
                                + "  (굵은 숫자 = 예측, 괄호 = 실측)",
                          xaxis=dict(range=[-5.3, 5.3], showgrid=False, zeroline=False, title="x [cm]"),
                          yaxis=dict(range=[-0.6, 12.4], showgrid=False, zeroline=False, title="y [cm] (Inlet 기준)",
                                     scaleanchor="x", scaleratio=1),
                          legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig)
    with right:
        st.subheader("9-point 비교")
        tbl = pd.DataFrame({"지점": M.PT_NAMES, "x": px, "y": py, "예측 [nm]": pred9, "포화도 θ": th9,
                            "실측 [nm]": obs9, "잔차 [nm]": pred9 - obs9})
        st.dataframe(tbl.style.format({"x": "{:.1f}", "y": "{:.1f}", "예측 [nm]": "{:.2f}", "포화도 θ": "{:.2f}",
                                       "실측 [nm]": "{:.2f}", "잔차 [nm]": "{:+.2f}"}, na_rep="—"),
                     hide_index=True)
        if not has_obs.any():
            st.info("이 조건에는 측정 데이터가 없습니다. 사이드바에서 데이터가 있는 조건으로 이동할 수 있습니다.")
        st.markdown(
            "**읽는 법** · 저온·저증기압(전구체 55 °C)에서는 흡착 자리가 많아 한 펄스로 웨이퍼 전체를 채우지 못하고, "
            "포화 front가 Inlet 쪽에 머뭅니다. 온도를 올리면 자리 수가 줄어 front가 하류로 밀려납니다.")

# ------------------------------------------------------------------ 역설계 (목표 사양 → 공정 조건 추천)


def apply_cond(row):
    st.session_state.T = int(round(row["T"]))
    st.session_state.Tp = float(row["Tp"])
    st.session_state.dose = float(round(row["dose"], 3))
    st.session_state.cyc = float(round(row["cyc"], 3))
    st.session_state.jump = "—"


with tab_inv:
    st.subheader("목표 사양 → 공정 조건 추천")
    st.write("원하는 두께·균일도를 넣으면, 모델로 후보 조건을 촘촘한 격자로 **전부** 계산해 조건을 만족하는 것 중 목표에 가장 좋은 "
             "조건을 추천합니다. 모델 계산이 빨라 전수 탐색이 가능하므로, 격자 간격(증착 1 °C, 전구체 0.5 °C) 안에서는 최적이 보장됩니다.")
    floor_info = M.measured_nu_floor(DATA)
    with st.form("inv_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("**목표 사양**")
            target = st.number_input("목표 평균 두께 [nm]", 0.5, 50.0, 12.0, step=0.5)
            tol = st.number_input("허용 오차 ± [nm]", 0.01, 10.0, 0.5, step=0.1)
            nu_max = st.number_input("최대 불균일도 1σ [%]", 0.0, 100.0, 3.0, step=0.5)
            th_min = st.number_input("최소 포화도 θ (1 = 완전 포화)", 0.0, 1.0, 0.9, step=0.05)
        with f2:
            st.markdown("**평가 방식**")
            obj = st.selectbox("최적화 목표", list(M.OBJECTIVES), format_func=M.OBJECTIVES.get)
            region = st.selectbox("평가 영역", ["9pt", "wafer"],
                                  format_func={"9pt": "9-point (측정 위치)", "wafer": "웨이퍼 전체 (가장자리는 외삽)"}.get)
            floor = st.number_input("모델 밖 편차 [%]", 0.0, 50.0, float(round(floor_info[0], 2)) if floor_info else 0.0, step=0.1)
            if floor_info:
                st.caption(f"측정 맵 중 가장 균일한 전구체 {floor_info[1]:g} °C / 증착 {floor_info[2]:g} °C 맵의 불균일도. "
                           "모델에 히터 온도 분포 같은 공간 요인이 없어서, 이 값을 빼면 포화 영역의 예측 불균일도가 0 %가 됩니다.")
        with f3:
            st.markdown("**탐색 범위**")
            T_rng = st.slider("증착 온도 [°C]", 100, 250, (100, 250))
            Tp_rng = st.slider("전구체 온도 [°C]", 50.0, 80.0, (50.0, 80.0), step=0.5)
            dose_free = st.checkbox("dose 조정 허용")
            D_rng = st.slider("dose 범위", 0.1, 10.0, (0.3, 3.0), step=0.1)
            cyc_free = st.checkbox("사이클 수 조정 허용 (두께는 사이클로 맞춤)")
            C_rng = st.slider("사이클 배율 범위", 0.2, 5.0, (0.3, 3.0), step=0.1)
        submitted = st.form_submit_button("추천 조건 찾기", type="primary")
    if submitted:
        if dose_free and D_rng[0] >= D_rng[1]:
            st.error("dose 범위의 최솟값이 최댓값보다 작아야 합니다.")
        elif cyc_free and C_rng[0] >= C_rng[1]:
            st.error("사이클 범위의 최솟값이 최댓값보다 작아야 합니다.")
        else:
            opts = dict(target=target, tol=tol, nu_max=nu_max, theta_min=th_min, objective=obj, region=region,
                        T_rng=T_rng, Tp_rng=Tp_rng, dose_rng=D_rng, dose_free=dose_free, cyc_free=cyc_free, cyc_rng=C_rng,
                        nu_floor=floor)
            res = M.inverse_design(model_key, P, meas=M.measured_conditions(DATA), **opts)
            st.session_state.inv = {"res": res, "opts": opts, "model": model_key, "params": dict(P)}

    INV = st.session_state.get("inv")
    if INV and (INV["model"] != model_key or INV["params"] != P):
        st.info("모델이나 파라미터가 바뀌었습니다. '추천 조건 찾기'를 다시 눌러 주세요.")
        INV = None
    if INV:
        R, o = INV["res"], INV["opts"]
        if R["status"] == "ok":
            st.success(f"후보 {R['n_cand']:,}개 × 평가 지점 {R['n_points']}개 계산 · {R['n_feas']:,}개가 조건 만족")
        else:
            tips = []
            if not o["cyc_free"]:
                tips.append('두께가 문제라면 "사이클 수 조정 허용"을 켜 보세요.')
            if o["nu_max"] < o["nu_floor"]:
                tips.append(f"최대 불균일도({o['nu_max']} %)가 모델 밖 편차({o['nu_floor']} %)보다 작아 만족할 수 없습니다.")
            st.error(f"조건을 모두 만족하는 후보가 없습니다 ({R['n_cand']:,}개 계산). 아래는 제약 위반이 가장 작은 절충안입니다. "
                     + " ".join(tips))

        top = R["top"]
        st.markdown("**추천 조건**")
        show = pd.DataFrame({
            "순위": np.arange(1, len(top) + 1), "증착 °C": top["T"], "전구체 °C": top["Tp"], "dose": top["dose"], "사이클": top["cyc"],
            "평균 두께 [nm]": top["mean"], "불균일도 1σ [%]": top["nu"], "(모델만)": top["nu_model"], "최소 θ": top["thmin"],
            "전구체 사용량": top["usage"], "근거": np.where(top["support"], "측정 맵 인근", "외삽"),
            "조건 만족": np.where(top["feasible"], "✓", "✕")})
        st.dataframe(show.style.format({"증착 °C": "{:.0f}", "전구체 °C": "{:.1f}", "dose": "×{:.3g}", "사이클": "×{:.3g}",
                                        "평균 두께 [nm]": "{:.2f}", "불균일도 1σ [%]": "{:.2f}", "(모델만)": "{:.2f}",
                                        "최소 θ": "{:.2f}", "전구체 사용량": "×{:.2f}"}), hide_index=True)
        bcols = st.columns(len(top) or 1)
        for i, (bc, (_, row)) in enumerate(zip(bcols, top.iterrows())):
            bc.button(f"{i + 1}순위 적용", key=f"apply_{i}", on_click=apply_cond, args=(row.to_dict(),))
        st.caption("불균일도 = √(모델 예측² + 모델 밖 편차²). 전구체 사용량 = dose × 증기압비(전구체 온도) × 사이클, "
                   "논문 조건(전구체 55 °C, dose 1, 사이클 1) = 1. '적용'을 누른 뒤 웨이퍼 맵 탭에서 그 조건을 확인할 수 있습니다.")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**균일도 vs 전구체 사용량**")
            allv = R["all"]
            thk = allv[allv.ok_thk & allv.ok_th]
            cloud = thk.iloc[:: max(1, len(thk) // 3000)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cloud.usage, y=cloud.nu, mode="markers", name="후보",
                                     marker=dict(color="#c3c8cf", size=4), hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=R["pareto"].usage, y=R["pareto"].nu, mode="lines+markers", name="Pareto 최적",
                                     line=dict(color="#0d7f74", width=2), marker=dict(size=5)))
            fig.add_trace(go.Scatter(x=top.usage, y=top.nu, mode="markers+text", name="추천", text=[str(i + 1) for i in range(len(top))],
                                     textposition="top center", marker=dict(color="#eb6834", size=11)))
            fig.add_hline(y=o["nu_max"], line_color="#c23b3a", line_dash="dash", annotation_text="불균일도 상한")
            ymax = max(o["nu_max"] * 1.6, float(top.nu.max()) if len(top) else 0) + 0.5
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22),
                              xaxis=dict(type="log", title="전구체 사용량 (논문 조건 = 1, 로그 축)"),
                              yaxis=dict(title="불균일도 1σ [%]", range=[0, ymax]))
            st.plotly_chart(fig)
            st.caption("두께·포화도 조건을 만족하는 후보 전체(회색)와 Pareto 최적선. 왼쪽 아래로 갈수록 좋습니다.")
        with g2:
            Ts, Tps, Ds = R["axes"]
            best_dose = float(top["dose"].iloc[0]) if len(top) else 1.0
            di = int(np.argmin(np.abs(np.log(Ds / best_dose))))
            cube = lambda col: allv[col].to_numpy().reshape(len(Ts), len(Tps), len(Ds))[:, :, di].T
            nu_g, ok_g = cube("nu"), cube("feasible").astype(bool)
            st.markdown(f"**가능 영역 · dose ×{Ds[di]:.3g}**" + (" · 사이클은 두께 목표에 맞춤" if o["cyc_free"] else ""))
            fig = go.Figure()
            fig.add_trace(go.Heatmap(x=Ts, y=Tps, z=nu_g, zmin=0, zmax=max(o["nu_max"] * 2, o["nu_floor"] + 1),
                                     colorscale=[[0, "#253494"], [0.15, "#2b83ba"], [0.3, "#abdda4"], [0.5, "#ffffbf"],
                                                 [0.75, "#fdae61"], [1, "#d7191c"]],
                                     colorbar=dict(title="1σ %"), hovertemplate="증착 %{x} °C · 전구체 %{y} °C<br>불균일도 %{z:.2f} %<extra></extra>"))
            fig.add_trace(go.Heatmap(x=Ts, y=Tps, z=np.where(ok_g, np.nan, 1.0), showscale=False, hoverinfo="skip",
                                     colorscale=[[0, "rgba(120,124,130,0.68)"], [1, "rgba(120,124,130,0.68)"]]))
            meas = M.measured_conditions(DATA)
            if len(meas):
                fig.add_trace(go.Scatter(x=meas[:, 1], y=meas[:, 0], mode="markers", name="9점 측정 조건",
                                         marker=dict(color="black", size=7)))
            on = np.abs(np.log(top["dose"].to_numpy() / Ds[di])) < 1e-6 if len(top) else []
            fig.add_trace(go.Scatter(x=top["T"][on], y=top["Tp"][on], mode="markers+text", name="추천",
                                     text=[str(i + 1) for i in np.flatnonzero(on)], textfont=dict(color="#111", size=11),
                                     marker=dict(color="white", size=17, line=dict(color="#eb6834", width=2))))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22),
                              xaxis_title="증착 온도 [°C]", yaxis_title="전구체 온도 [°C]")
            st.plotly_chart(fig)
            st.caption("색 = 불균일도 1σ, 회색 = 조건 불만족, 검은 점 = 9점 맵이 측정된 조건, 숫자 = 추천 순위")

        n_sup = int(top["support"].sum()) if len(top) else 0
        meas_tp = ", ".join(f"{v:g}" for v in sorted(set(M.measured_conditions(DATA)[:, 0])))
        st.markdown("**해석할 때 주의할 점**\n" + "\n".join("- " + t for t in [
            "추천은 **모델 예측**입니다. 증착 온도별 교차검증에서 모델의 예측 오차는 약 1.4 nm였으므로, 예측 차이가 이보다 작은 추천끼리는 실제로 구분되지 않을 수 있습니다.",
            f"9점 맵이 측정된 조건은 현재 데이터에서 전구체 {meas_tp} °C뿐입니다. 추천 {len(top)}개 중 {n_sup}개만 '측정 맵 인근'이고 나머지는 외삽이므로, 실제 공정에 쓰기 전에 확인 실험이 필요합니다.",
            "모델에서 전구체 온도와 dose는 '공급량'(증기압 × dose) 하나로만 작용합니다. 그래서 둘은 서로 대체할 수 있고, 증기압비는 피팅된 ΔH에 기반해 불확실성이 큽니다.",
            ("사이클 수 조정은 ALD 선형 성장(두께 ∝ 사이클)을 가정합니다. 초기 핵생성 지연은 반영하지 않으며, 논문의 실제 사이클 수를 모르므로 배율로만 표시합니다."
             if o["cyc_free"] else
             "사이클 수는 논문 조건으로 고정했습니다. 이때 평균 두께는 포화 두께 H_sat(T), 즉 주로 증착 온도로 정해집니다. 실제 공정에서는 두께를 사이클 수로 맞추므로 '사이클 수 조정 허용'을 켜면 균일도 조건만으로 온도를 고를 수 있습니다."),
            ("평가 영역이 9-point라 웨이퍼 가장자리(중심에서 3 cm 바깥)는 반영되지 않습니다." if o["region"] == "9pt"
             else "웨이퍼 전체 평가는 측정점이 없는 가장자리까지 외삽한 결과입니다."),
        ]))

# ------------------------------------------------------------------ 공정 곡선
with tab_curve:
    sel_pts = {"P9 근거리": 8, "P5 중심": 4, "P1 원거리": 0}
    colors = {"P9 근거리": "#0d7f74", "P5 중심": "#3b6fd1", "P1 원거리": "#eb6834"}

    def curve(xvar, grid):
        f = go.Figure()
        for nm, i in sel_pts.items():
            x, y = M.PTS[i]
            if xvar == "T":
                v = FN(P, x, y, grid, Tp, dose)[0]
                d = DATA[(DATA.Tp == Tp) & np.isclose(DATA.x, x) & np.isclose(DATA.y, y)]
                dx = d["T"]
            else:
                v = FN(P, x, y, T, grid, dose)[0]
                d = DATA[(DATA["T"] == T) & np.isclose(DATA.x, x) & np.isclose(DATA.y, y)]
                dx = d["Tp"]
            f.add_trace(go.Scatter(x=grid, y=v, mode="lines", name=f"{nm} 예측", line=dict(color=colors[nm], width=2.5)))
            if len(d):
                f.add_trace(go.Scatter(x=dx, y=d.t_ox, mode="markers", name=f"{nm} 실측",
                                       marker=dict(color=colors[nm], size=9, symbol="circle-open", line=dict(width=2))))
        if xvar == "T":
            f.add_trace(go.Scatter(x=grid, y=FN(P, 0, 7.5, grid, Tp, dose)[2], mode="lines", name="H_sat(T)",
                                   line=dict(color="#888", dash="dash")))
            f.add_vline(x=T, line_color="#999", line_dash="dot")
            f.update_xaxes(title="증착 온도 [°C]")
        else:
            f.add_vline(x=Tp, line_color="#999", line_dash="dot")
            f.update_xaxes(title="전구체 온도 [°C]")
        f.update_yaxes(title="두께 [nm]", rangemode="tozero")
        f.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
        return f

    a, b = st.columns(2)
    with a:
        st.markdown(f"**ALD 온도 창** · 전구체 {Tp} °C")
        st.plotly_chart(curve("T", np.linspace(100, 250, 151)))
    with b:
        st.markdown(f"**전구체 공급 곡선** · 증착 {T} °C")
        st.plotly_chart(curve("Tp", np.linspace(50, 80, 121)))
    st.caption("실선 = 모델, 빈 원 = 측정 데이터(현재 조건과 같은 전구체/증착 온도, 같은 지점일 때만 표시).")

# ------------------------------------------------------------------ 피팅 · 검증


def do_fit(key):
    try:
        p, r = M.fit(key, DATA, params(key))
        set_params(key, p)
        st.session_state.fit_msg = f"{M.MODELS[key]['name']}: {len(DATA)}점에 피팅 완료 · 학습 RMSE {r:.3f} nm"
    except Exception as e:  # noqa: BLE001
        st.session_state.fit_msg = f"피팅 실패: {e}"


@st.cache_data(show_spinner="온도별 교차검증 중…")
def cached_cv(key, df):
    return M.loto_cv(key, df)


with tab_fit:
    st.markdown(
        "식의 **형태**는 물리(Langmuir 흡착, 유선 방향 질량수지, Arrhenius, Clausius–Clapeyron)에서 왔지만, "
        "파라미터 **값**은 모두 측정 데이터에 맞춘 것입니다. 사전 이론값과의 비교는 아직 들어 있지 않습니다.")
    c1, c2 = st.columns([1, 2])
    c1.button(f"현재 데이터로 '{M.MODELS[model_key]['name']}' 다시 피팅", on_click=do_fit, args=(model_key,),
              type="primary")
    if st.session_state.fit_msg:
        c2.success(st.session_state.fit_msg)

    spec = M.MODELS[model_key]["spec"]
    ptab = pd.DataFrame([{"파라미터": v[4], "현재 값": P[n], "단위": v[3], "사전 피팅값": v[0],
                          "성격": "경험항" if n in ("xs", "w", "f0") else "물리 형태 + 피팅",
                          "허용 범위": f"{v[1]} ~ {v[2]}"} for n, v in spec.items()])
    st.dataframe(ptab.style.format({"현재 값": "{:.4g}", "사전 피팅값": "{:.4g}"}), hide_index=True)

    st.subheader("모델 비교")
    rows = []
    for key in M.MODELS:
        rows.append({"모델": M.MODELS[key]["name"], "파라미터 수": len(M.MODELS[key]["spec"]),
                     "학습 RMSE [nm]": M.rmse(key, params(key), DATA)})
    run_cv = st.checkbox("증착 온도별 Leave-one-out 교차검증 실행 (수 초)", value=False)
    if run_cv:
        cvs = {key: cached_cv(key, DATA) for key in M.MODELS}
        for row, key in zip(rows, M.MODELS):
            row["검증 RMSE 평균 [nm]"] = cvs[key]["rmse"].mean() if len(cvs[key]) else np.nan
    st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True)
    if run_cv:
        wide = pd.concat({M.MODELS[k]["name"]: cvs[k].set_index("held_out_T")["rmse"] for k in M.MODELS}, axis=1)
        wide.index.name = "빼낸 증착 온도 [°C]"
        st.caption("각 증착 온도의 데이터를 통째로 빼고 나머지로 피팅한 뒤, 빠진 온도를 예측한 RMSE [nm]")
        st.dataframe(wide.style.format("{:.2f}"))

    st.subheader("예측 vs 실측")
    pr = M.predict(model_key, P, DATA)
    fig = go.Figure(go.Scatter(x=DATA.t_ox, y=pr, mode="markers",
                               marker=dict(color=DATA["T"], colorscale="Viridis", size=9,
                                           colorbar=dict(title="증착 °C")),
                               customdata=DATA[["Tp", "T", "x", "y"]].to_numpy(),
                               hovertemplate="전구체 %{customdata[0]} °C · 증착 %{customdata[1]} °C<br>"
                                             "(%{customdata[2]}, %{customdata[3]}) cm<br>실측 %{x:.2f} · 예측 %{y:.2f}<extra></extra>"))
    lim = [0, max(18, float(DATA.t_ox.max()) + 1)]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#999", dash="dash"), showlegend=False))
    fig.update_layout(height=460, xaxis_title="실측 [nm]", yaxis_title="예측 [nm]", margin=dict(l=10, r=10, t=10, b=10),
                      showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig)

# ------------------------------------------------------------------ Zernike
with tab_zern:
    st.markdown("**A. 모델 맵의 Zernike 분해** · 3단계에서 Z-Reg가 예측할 저차 구조가 어떤 모드로 이루어지는지 확인합니다.")
    order = st.slider("최대 차수", 1, 8, 6)
    Xg, Yg, mg = M.wafer_grid(121)
    Zg = FN(P, Xg, Yg, T, Tp, dose)[0]
    modes, coef = M.zernike_fit(Xg[mg], Yg[mg], Zg[mg], order)
    recon = M.zernike_eval(modes, coef, Xg[mg], Yg[mg])
    err = float(np.sqrt(np.mean((recon - Zg[mg]) ** 2)))
    z1, z2 = st.columns([2, 1])
    with z1:
        labels = [f"Z({n},{m})" for n, m in modes]
        f = go.Figure(go.Bar(x=labels, y=coef, marker_color=["#0d7f74" if m == 0 else "#eb6834" for n, m in modes]))
        f.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="계수 [nm]")
        st.plotly_chart(f)
    with z2:
        st.metric(f"{len(modes)}개 모드 재구성 오차 (RMSE)", f"{err:.3f} nm")
        errs = []
        for o in range(1, 9):
            md, cf = M.zernike_fit(Xg[mg], Yg[mg], Zg[mg], o)
            errs.append(float(np.sqrt(np.mean((M.zernike_eval(md, cf, Xg[mg], Yg[mg]) - Zg[mg]) ** 2))))
        f2 = go.Figure(go.Scatter(x=list(range(1, 9)), y=errs, mode="lines+markers"))
        f2.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="차수", yaxis_title="RMSE [nm]")
        st.plotly_chart(f2)
        st.caption("초록 = 축대칭(m=0), 주황 = 방위각 성분")

    st.divider()
    st.markdown("**B. 실측 잔차의 저차 Zernike 분해** · 9점으로는 2차(6개 모드)까지만 안정적으로 추정됩니다.")
    if has_obs.sum() >= 6:
        resid = (obs9 - pred9)[has_obs]
        md, cf = M.zernike_fit(px[has_obs], py[has_obs], resid, 2)
        zc1, zc2 = st.columns([1, 1.2])
        with zc1:
            st.dataframe(pd.DataFrame({"모드": [f"Z({n},{m})" for n, m in md], "계수 [nm]": cf})
                         .style.format({"계수 [nm]": "{:+.3f}"}), hide_index=True)
            st.caption(f"남는 오차 (6모드로 설명 못 한 부분): "
                       f"{np.sqrt(np.mean((M.zernike_eval(md, cf, px[has_obs], py[has_obs]) - resid) ** 2)):.3f} nm")
        with zc2:
            Rg = np.where(mg, M.zernike_eval(md, cf, Xg, Yg), np.nan)
            lim = float(np.nanmax(np.abs(Rg))) or 1.0
            f3 = go.Figure(go.Heatmap(x=Xg[0], y=Yg[:, 0], z=Rg, colorscale="RdBu", zmid=0, zmin=-lim, zmax=lim,
                                      colorbar=dict(title="실측−예측")))
            f3.add_trace(go.Scatter(x=px, y=py, mode="markers", marker=dict(color="black", size=6), showlegend=False))
            f3.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                             yaxis=dict(scaleanchor="x", scaleratio=1))
            st.plotly_chart(f3)
        st.caption("3단계 아이디어: 공정 조건 → 이 6개 잔차 계수를 예측하는 작은 회귀 모델(Z-Reg)을 학습해 물리 모델에 더합니다. "
                   "측정점이 없는 가장자리(반경 3 cm 바깥)는 외삽입니다.")
    else:
        st.info("현재 조건에 측정점이 6개 이상 없어 잔차 분해를 할 수 없습니다.")

# ------------------------------------------------------------------ 데이터셋 생성 (데이터 탭 하단)
with tab_data:
    st.divider()
    st.subheader("3단계 학습용 합성 데이터셋 생성")
    st.write("현재 모델·파라미터로 공정 조건을 라틴 하이퍼큐브 샘플링해 2D 두께 맵을 만듭니다 (`.npz`: cond, maps, mask, x_cm, y_cm).")
    g1, g2, g3 = st.columns(3)
    n = g1.number_input("샘플 수", 10, 2000, 200, step=10)
    res = g1.select_slider("해상도", [32, 64, 128, 256], value=128)
    T_rng = g2.slider("증착 온도 범위 [°C]", 100, 250, (100, 250))
    Tp_rng = g2.slider("전구체 온도 범위 [°C]", 50, 80, (55, 75))
    d_rng = g3.select_slider("Dose 배율 범위", [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0], value=(0.5, 2.0))
    seed = g3.number_input("seed", 0, 10_000, 0)
    if st.button("데이터셋 생성"):
        if d_rng[0] == d_rng[1]:
            d_rng = (d_rng[0], d_rng[0] * 1.0001)
        ds = M.make_dataset(model_key, P, int(n), int(res), T_rng, Tp_rng, d_rng, int(seed))
        buf = io.BytesIO()
        np.savez_compressed(buf, **ds, model=np.array(model_key),
                            params=np.array(list(P.values())), param_names=np.array(list(P)))
        st.session_state.ds_bytes = buf.getvalue()
        st.session_state.ds_info = f"{int(n)}개 × {res}×{res} · {len(buf.getvalue()) / 1e6:.1f} MB"
    if st.session_state.get("ds_bytes"):
        st.download_button(f"데이터셋 내려받기 ({st.session_state.ds_info})", st.session_state.ds_bytes,
                           "ald_synthetic_dataset.npz", "application/octet-stream")

# ------------------------------------------------------------------ 모델 식
with st.expander("모델 식과 가정"):
    st.markdown("**포화 front + 흐름 jet 모델** (유선 x를 따라 Inlet에서 y만큼 떨어진 지점)")
    st.latex(r"H_{sat}(T)=H_0\,e^{-b(T-100)},\qquad a(T)=a_0\,e^{-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{373}\right)}")
    st.latex(r"L_f(x)=\Lambda\,\frac{P_v(T_p)}{P_v(55^\circ\mathrm{C})}\,\frac{H_0}{H_{sat}(T)}\,\mathrm{dose}\;"
             r"e^{-\frac{(x-x_s)^2}{2w^2}}")
    st.latex(r"\frac{dG}{dy}=-n_s\left(1-e^{-aG/n_s}\right)\;\Rightarrow\;"
             r"\theta=\frac{(e^{aL_f}-1)e^{-ay}}{1+(e^{aL_f}-1)e^{-ay}}")
    st.latex(r"t_{ox}=H_{sat}(T)\,\left[f_0+(1-f_0)\,\theta\right]")
    st.markdown(
        "- θ는 front 위치 `y_f ≈ L_f`를 중심으로 한 로지스틱 함수이고, `a`가 작은 극한에서는 기존 `exp(−d/L_dep)` 모델로 돌아갑니다.\n"
        "- `x_s`, `w`, `f₀`는 데이터 모양을 보고 넣은 경험항입니다.\n"
        "- 모든 파라미터는 논문 그림 판독 데이터로 피팅한 값이며, 공정 조건으로부터 계산한 사전 이론값이 아닙니다.\n"
        "- 데이터 출처: 사용자가 제공한 논문 Fig. 1a, 1b(전구체 55 °C 행)의 판독값. 판독 오차가 있을 수 있습니다.")
