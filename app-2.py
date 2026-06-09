import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from PIL import Image
import io

st.set_page_config(
    page_title="RGB 빛 탐색기",
    page_icon="🎨",
    layout="centered"
)

# ─── Pantone 색상 데이터베이스 ──────────────────────────────────────────────
PANTONE_COLORS = {
    "Pantone White — 흰색":                   (255, 255, 255),
    "Pantone 9001 C — 아이보리":               (243, 239, 225),
    "Pantone 9060 C — 크림":                  (240, 236, 213),
    "Pantone 705 C — 연한 분홍빛 흰색":        (247, 211, 215),
    "Pantone 012 C — 옐로":                   (255, 211, 0),
    "Pantone 100 C — 연한 노랑":              (243, 234, 108),
    "Pantone 102 C — 레몬 옐로":              (253, 218, 0),
    "Pantone 109 C — 밝은 노랑":              (255, 195, 0),
    "Pantone 116 C — 미모사 옐로":            (255, 194, 0),
    "Pantone 123 C — 옥수수 노랑":            (255, 199, 44),
    "Pantone 127 C — 스트로우":               (232, 218, 116),
    "Pantone 1235 C — 머스터드":              (255, 178, 0),
    "Pantone 021 C — 오렌지":                 (254, 80, 0),
    "Pantone 130 C — 앰버":                   (232, 156, 0),
    "Pantone 151 C — 브라이트 오렌지":        (255, 130, 0),
    "Pantone 1505 C — 다크 오렌지":           (255, 102, 0),
    "Pantone 157 C — 멜론":                   (235, 122, 70),
    "Pantone 1575 C — 피치":                  (255, 140, 90),
    "Pantone 7416 C — 테라코타":              (206, 93, 68),
    "Pantone 7550 C — 버터스카치":            (220, 165, 60),
    "Pantone 032 C — 웜 레드":                (239, 51, 64),
    "Pantone 179 C — 토마토 레드":            (220, 55, 32),
    "Pantone 186 C — 레드":                   (200, 16, 46),
    "Pantone 485 C — 빨강":                   (218, 37, 29),
    "Pantone 1795 C — 미디엄 레드":           (213, 43, 30),
    "Pantone 1805 C — 다크 레드":             (188, 32, 30),
    "Pantone 1815 C — 버건디":                (148, 42, 34),
    "Pantone 7621 C — 딥 레드":               (162, 34, 35),
    "Pantone 7423 C — 라즈베리":              (175, 30, 85),
    "Pantone 196 C — 블러시 핑크":            (245, 183, 190),
    "Pantone 205 C — 로즈":                   (228, 110, 130),
    "Pantone 211 C — 플라밍고":               (246, 139, 166),
    "Pantone 213 C — 핑크":                   (214, 64, 128),
    "Pantone 218 C — 딥 핑크":                (213, 0, 115),
    "Pantone 232 C — 핫 핑크":               (240, 78, 152),
    "Pantone 239 C — 로즈 핑크":              (235, 80, 140),
    "Pantone 812 C — 네온 핑크":              (255, 10, 130),
    "Pantone 2365 C — 라이트 핑크":           (253, 188, 200),
    "Pantone Rhodamine Red C — 마젠타":       (225, 0, 128),
    "Pantone 2415 C — 오키드":                (200, 60, 165),
    "Pantone 2562 C — 라벤더":               (200, 140, 210),
    "Pantone 2583 C — 모브":                  (185, 130, 180),
    "Pantone 2587 C — 라이트 퍼플":           (181, 108, 196),
    "Pantone 265 C — 미디엄 퍼플":            (152, 77, 189),
    "Pantone 267 C — 바이올렛":               (88, 29, 145),
    "Pantone 2685 C — 다크 퍼플":             (66, 19, 115),
    "Pantone 2627 C — 다크 바이올렛":         (61, 20, 100),
    "Pantone 2736 C — 블루 바이올렛":         (85, 89, 191),
    "Pantone 2746 C — 로열 퍼플":             (60, 47, 168),
    "Pantone 7442 C — 위스테리아":            (178, 143, 200),
    "Pantone 279 C — 콘플라워 블루":          (76, 147, 207),
    "Pantone 285 C — 비비드 블루":            (0, 115, 207),
    "Pantone 286 C — 로열 블루":              (0, 62, 168),
    "Pantone 293 C — 블루":                   (0, 56, 168),
    "Pantone 300 C — 다크 블루":              (0, 83, 160),
    "Pantone 072 C — 인디고":                 (18, 18, 178),
    "Pantone 2748 C — 코발트":                (0, 38, 175),
    "Pantone 2766 C — 다크 네이비":           (14, 28, 107),
    "Pantone 2955 C — 네이비":                (0, 48, 120),
    "Pantone 7456 C — 페리윙클":              (135, 141, 193),
    "Pantone 7686 C — 카뎃 블루":             (65, 130, 178),
    "Pantone 7700 C — 스틸 블루":             (50, 120, 155),
    "Pantone 306 C — 시안":                   (0, 191, 243),
    "Pantone 2985 C — 스카이 블루":           (0, 183, 227),
    "Pantone 2995 C — 브라이트 시안":         (0, 166, 214),
    "Pantone 312 C — 틸 블루":               (0, 163, 182),
    "Pantone 315 C — 피콕":                   (0, 119, 147),
    "Pantone 320 C — 틸":                     (0, 148, 159),
    "Pantone 3262 C — 터쿼이즈":             (0, 193, 176),
    "Pantone 3272 C — 브라이트 터쿼이즈":    (0, 186, 180),
    "Pantone 3295 C — 제이드":               (0, 155, 143),
    "Pantone 339 C — 민트":                   (0, 175, 130),
    "Pantone 3516 C — 시폼":                  (100, 206, 180),
    "Pantone 347 C — 포레스트 그린":          (0, 153, 80),
    "Pantone 354 C — 그린":                   (0, 175, 66),
    "Pantone 355 C — 켈리 그린":              (0, 167, 75),
    "Pantone 360 C — 브라이트 그린":          (82, 177, 81),
    "Pantone 362 C — 리프 그린":              (86, 160, 71),
    "Pantone 368 C — 라임 그린":              (120, 190, 60),
    "Pantone 376 C — 옐로-그린":              (148, 193, 31),
    "Pantone 382 C — 샤르트뢰즈":            (186, 212, 0),
    "Pantone 390 C — 옐로-라임":              (192, 213, 8),
    "Pantone 802 C — 네온 그린":              (0, 238, 100),
    "Pantone 3985 C — 라임":                  (192, 215, 25),
    "Pantone 4485 C — 올리브 그린":           (103, 107, 48),
    "Pantone 4495 C — 아미 그린":             (120, 120, 60),
    "Pantone 7489 C — 세이지":                (114, 158, 107),
    "Pantone 7494 C — 셀라돈":               (160, 187, 164),
    "Pantone 7739 C — 펀":                    (64, 130, 90),
    "Pantone 5773 C — 유칼립투스":            (130, 160, 130),
    "Pantone 462 C — 탄":                     (165, 130, 85),
    "Pantone 469 C — 커피":                   (120, 80, 40),
    "Pantone 476 C — 다크 브라운":            (88, 55, 39),
    "Pantone 478 C — 브라운":                 (113, 71, 51),
    "Pantone 483 C — 러스트 브라운":          (154, 74, 57),
    "Pantone 729 C — 카라멜":                 (188, 152, 95),
    "Pantone 1685 C — 시에나":               (175, 100, 60),
    "Pantone 7504 C — 카키":                  (195, 169, 132),
    "Pantone 7527 C — 밀":                    (212, 197, 175),
    "Pantone 7531 C — 토프":                  (175, 163, 148),
    "Pantone Cool Gray 1 C — 매우 연한 회색": (230, 228, 226),
    "Pantone Cool Gray 2 C — 연한 회색":      (220, 217, 214),
    "Pantone Cool Gray 3 C — 소프트 그레이":  (210, 207, 202),
    "Pantone Cool Gray 4 C — 라이트 미드":    (199, 195, 188),
    "Pantone Cool Gray 5 C — 미드 그레이":    (188, 184, 177),
    "Pantone Cool Gray 6 C — 미디엄":         (175, 170, 163),
    "Pantone Cool Gray 7 C — 그레이":         (163, 158, 151),
    "Pantone Cool Gray 8 C — 미디엄 다크":    (148, 142, 135),
    "Pantone Cool Gray 9 C — 다크 그레이":    (130, 124, 117),
    "Pantone Cool Gray 10 C — 더 다크":       (109, 103, 98),
    "Pantone Cool Gray 11 C — 매우 어두운":   (83, 78, 72),
    "Pantone Black C — 블랙":                 (30, 30, 30),
    "Pantone Black 6 C — 리치 블랙":          (10, 10, 20),
    "Pantone 419 C — 니어 블랙":              (35, 31, 32),
    "Pantone 877 C — 실버":                   (166, 169, 170),
    "Pantone 7406 C — 골드 옐로":             (240, 189, 0),
    "Pantone 191 C — 살몬":                   (240, 118, 128),
}


# ─── 공통 유틸리티 ──────────────────────────────────────────────────────────
def find_closest_pantone(r, g, b):
    min_dist = float('inf')
    result = {"name": "Unknown", "rgb": (r, g, b), "distance": 0.0}
    for name, (pr, pg, pb) in PANTONE_COLORS.items():
        dist = ((r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            result = {"name": name, "rgb": (pr, pg, pb), "distance": round(min_dist, 1)}
    return result


def brightness(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b


def split_pantone_name(full_name):
    parts = full_name.split("—")
    code    = parts[0].strip()
    name_kr = parts[1].strip() if len(parts) > 1 else ""
    return code, name_kr


# ─── 탭1: 빛 합성 ──────────────────────────────────────────────────────────
def create_light_venn(r, g, b, size=540):
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    cx, cy = size // 2, size // 2
    radius = int(size * 0.285)
    offset = int(radius * 0.68)

    centers = [
        (cx,                          cy - offset),
        (cx - int(offset * 0.866),    cy + offset // 2),
        (cx + int(offset * 0.866),    cy + offset // 2),
    ]
    channel_vals = [r / 255.0, g / 255.0, b / 255.0]
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    sigma = size * 0.013

    for i, (cx_, cy_) in enumerate(centers):
        mask = ((x_grid - cx_) ** 2 + (y_grid - cy_) ** 2 <= radius ** 2).astype(np.float32)
        soft = gaussian_filter(mask, sigma=sigma)
        canvas[:, :, i] += soft * channel_vals[i]

    canvas = np.clip(canvas, 0, 1)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.imshow(canvas, aspect='equal', interpolation='bilinear', extent=[0, size, size, 0])
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)

    lbl_dist = radius + int(size * 0.05)
    labels_info = [
        ('R', cx,                                          cy - offset - lbl_dist,                       '#FF5555'),
        ('G', cx - int(offset * 0.866) - int(lbl_dist * 0.78), cy + offset // 2 + int(lbl_dist * 0.55), '#55FF55'),
        ('B', cx + int(offset * 0.866) + int(lbl_dist * 0.78), cy + offset // 2 + int(lbl_dist * 0.55), '#5599FF'),
    ]
    for label, lx, ly, color in labels_info:
        ax.text(lx, ly, label, color=color, fontsize=24, fontweight='bold',
                ha='center', va='center', fontfamily='monospace')

    ax.axis('off')
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#0d0d0d', dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf


def render_tab_mix():
    venn_area = st.container()
    info_area = st.container()

    st.markdown("---")
    st.markdown("### 🎚️ RGB 값 조절")
    r_val = st.slider("🔴  Red   (R)", 0, 255, 200)
    g_val = st.slider("🟢  Green (G)", 0, 255, 60)
    b_val = st.slider("🔵  Blue  (B)", 0, 255, 210)
    st.caption("💡 R+G=노랑 / R+B=마젠타 / G+B=시안 / R+G+B=흰색 (가산혼합)")

    with venn_area:
        st.image(create_light_venn(r_val, g_val, b_val), use_container_width=True)

    with info_area:
        hex_col = f"#{r_val:02X}{g_val:02X}{b_val:02X}"
        pantone = find_closest_pantone(r_val, g_val, b_val)
        pr, pg, pb = pantone["rgb"]
        p_hex   = f"#{pr:02X}{pg:02X}{pb:02X}"
        fg      = "black" if brightness(r_val, g_val, b_val) > 140 else "white"
        p_fg    = "black" if brightness(pr, pg, pb) > 140 else "white"
        p_code, p_name_kr = split_pantone_name(pantone["name"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="color-card" style="background:{hex_col}; color:{fg};">
                <div class="card-label">현재 색상</div>
                <div class="card-name">{hex_col.upper()}</div>
                <div class="card-code">R {r_val} · G {g_val} · B {b_val}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="color-card" style="background:{p_hex}; color:{p_fg};">
                <div class="card-label">🎯 가장 유사한 팬톤</div>
                <div class="card-name">{p_name_kr}</div>
                <div class="card-code">{p_code}</div>
                <div class="card-code">R {pr} · G {pg} · B {pb} · 거리 {pantone['distance']}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


# ─── 탭2: 빛 분석 ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_dominant_colors(img_bytes, n_colors=6):
    """K-Means 클러스터링으로 대표 색상 추출"""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 분석 속도를 위해 최대 400px로 리사이즈
    max_dim = 400
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    pixels = np.array(img).reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)

    # 비율 계산
    counts = np.bincount(labels, minlength=n_colors)
    ratios = counts / counts.sum()

    # 비율 내림차순 정렬
    order  = np.argsort(-ratios)
    return [(tuple(centers[i]), float(ratios[i])) for i in order]


def make_palette_image(colors_ratios, width=600, height=70):
    """가로 비율 팔레트 이미지 생성"""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for (r, g, b), ratio in colors_ratios:
        seg_w = max(1, int(round(ratio * width)))
        x_end = min(x + seg_w, width)
        arr[:, x:x_end, :] = [r, g, b]
        x = x_end
    if x < width:
        arr[:, x:width, :] = arr[:, x - 1:x, :]
    return Image.fromarray(arr)


def render_tab_analyze():
    st.markdown("#### 📂 이미지 업로드")
    st.caption("JPG, PNG, WEBP 등 이미지를 올리면 주요 색상과 팬톤 코드를 분석합니다.")

    uploaded = st.file_uploader(
        "이미지를 드래그하거나 클릭해서 업로드하세요",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded is None:
        st.info("⬆️ 이미지를 업로드하면 색상 분석이 시작됩니다.")
        return

    img_bytes = uploaded.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    n_colors = st.slider("추출할 대표 색상 수", 3, 8, 6)

    col_img, col_gap = st.columns([2, 1])
    with col_img:
        st.image(img, caption="업로드된 이미지", use_container_width=True)

    with st.spinner("색상 분석 중..."):
        dominant = extract_dominant_colors(img_bytes, n_colors)

    # 팔레트 바
    st.markdown("#### 🎨 대표 색상 팔레트")
    palette_img = make_palette_image(dominant)
    st.image(palette_img, use_container_width=True)

    # 각 색상 카드
    st.markdown("#### 🎯 팬톤 색상 매칭 결과")

    cols = st.columns(min(n_colors, 3))
    for idx, ((r, g, b), ratio) in enumerate(dominant):
        pantone  = find_closest_pantone(r, g, b)
        pr, pg, pb = pantone["rgb"]
        hex_col  = f"#{r:02X}{g:02X}{b:02X}"
        p_hex    = f"#{pr:02X}{pg:02X}{pb:02X}"
        fg       = "black" if brightness(r, g, b)    > 140 else "white"
        p_fg     = "black" if brightness(pr, pg, pb) > 140 else "white"
        p_code, p_name_kr = split_pantone_name(pantone["name"])

        with cols[idx % 3]:
            st.markdown(f"""
            <div class="analysis-card">
                <div class="ac-top" style="background:{hex_col}; color:{fg};">
                    <div class="ac-pct">{ratio*100:.1f}%</div>
                    <div class="ac-hex">{hex_col.upper()}</div>
                    <div class="ac-rgb">R {r} · G {g} · B {b}</div>
                </div>
                <div class="ac-bottom" style="background:{p_hex}; color:{p_fg};">
                    <div class="ac-pantone-label">🎯 팬톤 근사색</div>
                    <div class="ac-pantone-name">{p_name_kr}</div>
                    <div class="ac-pantone-code">{p_code}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # 3열마다 새 줄
        if (idx + 1) % 3 == 0 and (idx + 1) < n_colors:
            cols = st.columns(min(n_colors - idx - 1, 3))

    st.markdown("<br>", unsafe_allow_html=True)


# ─── 공통 CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.big-title  { text-align:center; font-size:2.1rem; font-weight:800; margin-bottom:0.1rem; }
.subtitle   { text-align:center; color:#888; font-size:0.95rem; margin-bottom:1.2rem; }

/* 합성 탭 카드 */
.color-card {
    border-radius: 14px; padding: 18px 14px; text-align: center;
    min-height: 115px; display:flex; flex-direction:column;
    justify-content:center; gap:4px; border:1px solid #333;
}
.card-label { font-size:0.78rem; opacity:0.72; margin-bottom:3px; }
.card-name  { font-size:1.0rem; font-weight:700; margin-bottom:2px; }
.card-code  { font-family:monospace; font-size:0.82rem; opacity:0.85; }

/* 분석 탭 카드 */
.analysis-card {
    border-radius: 14px; overflow:hidden;
    border:1px solid #333; margin-bottom:14px;
}
.ac-top {
    padding: 14px 12px 12px;
    display:flex; flex-direction:column; gap:3px;
}
.ac-pct   { font-size:1.3rem; font-weight:800; }
.ac-hex   { font-family:monospace; font-size:0.9rem; font-weight:600; }
.ac-rgb   { font-family:monospace; font-size:0.75rem; opacity:0.82; }
.ac-bottom {
    padding: 10px 12px 12px;
    display:flex; flex-direction:column; gap:3px;
    border-top: 1px solid rgba(128,128,128,0.25);
}
.ac-pantone-label { font-size:0.72rem; opacity:0.72; }
.ac-pantone-name  { font-size:0.92rem; font-weight:700; }
.ac-pantone-code  { font-family:monospace; font-size:0.76rem; opacity:0.82; }
</style>
""", unsafe_allow_html=True)

# ─── 메인 레이아웃 ──────────────────────────────────────────────────────────
st.markdown('<div class="big-title">🎨 RGB 빛 탐색기</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">빛을 합성하고, 이미지의 색을 분석해보세요</div>', unsafe_allow_html=True)
st.markdown("")

tab1, tab2 = st.tabs(["💡 빛 합성해보기", "🔍 빛 분석해보기"])

with tab1:
    render_tab_mix()

with tab2:
    render_tab_analyze()
