import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st

# Optional deps (do not hard-fail demo if missing)
try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
    CV2_OK = True
except Exception:
    cv2 = None
    CV2_OK = False


# =========================
# Baseline constraints (single source of truth)
# =========================
REPO_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = REPO_ROOT / "assets"
MARKER_PDF_PATH = ASSETS_DIR / "FormFoundry_A4_MarkerSheet_v1_1.pdf"
DEMO_GALLERY_DIR = ASSETS_DIR / "demo_gallery"


# =========================
# Utilities
# =========================
def _make_placeholder(text: str, w: int = 1100, h: int = 650):
    """Return an image array placeholder. Never raises."""
    if np is None:
        return None
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    if CV2_OK:
        y = 60
        for line in text.split("\n"):
            cv2.putText(img, line[:80], (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
            y += 45
    return img


def _read_image(path: Path):
    """Read image from path; fallback to placeholder if missing/broken."""
    if not path.exists():
        return _make_placeholder(f"Missing file:\n{path}")
    if CV2_OK:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return _make_placeholder(f"Unreadable image:\n{path.name}")
        # OpenCV BGR -> RGB for Streamlit
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _make_placeholder(f"cv2 not available.\nCannot read: {path.name}")


def _read_upload(uploaded_file):
    """Read Streamlit uploaded image to RGB."""
    if uploaded_file is None:
        return None
    data = uploaded_file.getvalue()
    if (not CV2_OK) or (np is None):
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _list_demo_packs() -> Dict[str, Path]:
    packs = {}
    if DEMO_GALLERY_DIR.exists():
        for p in sorted(DEMO_GALLERY_DIR.iterdir()):
            if p.is_dir() and (p / "preset.json").exists():
                packs[p.name] = p
    return packs


def _load_preset(pack_dir: Path) -> dict:
    try:
        return json.loads((pack_dir / "preset.json").read_text(encoding="utf-8"))
    except Exception:
        return {
            "preset_id": pack_dir.name,
            "title": "Broken preset.json",
            "middleware": {},
            "artifacts": {},
            "notes": {"demo_mode": True, "disclaimer": "preset.json could not be read."},
        }


def _render_kv(title: str, d: dict):
    st.markdown(f"### {title}")
    st.json(d, expanded=True)


def _download_json_button(label: str, obj: dict, filename: str):
    st.download_button(
        label=label,
        data=json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=filename,
        mime="application/json",
    )


# =========================
# UI
# =========================
st.set_page_config(page_title="FormFoundry AI — Stable Demo MVP", layout="wide")
st.title("FormFoundry AI — Stable Demo MVP")
st.caption("Stable Demo Gallery는 어떤 사진을 올려도 항상 end-to-end 결과를 재현(프리셋)합니다. Live(Experimental)는 별도 모드로 유지합니다.")

with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "실행 모드",
        ["Demo Gallery (Always Success)", "Live (Experimental)"],
        index=0,
    )

    st.header("Demo Gallery")
    packs = _list_demo_packs()
    if not packs:
        st.error("assets/demo_gallery 아래에 preset.json이 있는 데모팩이 없습니다. (1단계 placeholder 생성부터 실행하세요.)")

    pack_id = st.selectbox("Demo Pack", options=list(packs.keys()) if packs else ["demo_01"], index=0)
    use_upload_as_input = st.checkbox("업로드한 사진을 '입력 이미지'로 표시 (결과는 프리셋 유지)", value=True)

    st.header("User Context (Demo stability)")
    user_height_cm = st.number_input("키(cm) — 설명/표기용", min_value=120.0, max_value=220.0, value=183.0, step=0.5)

    st.header("Marker Sheet")
    if MARKER_PDF_PATH.exists():
        st.download_button(
            "A4 마커 시트 PDF 다운로드",
            data=MARKER_PDF_PATH.read_bytes(),
            file_name=MARKER_PDF_PATH.name,
            mime="application/pdf",
        )
    else:
        st.warning("Marker PDF not found: assets/FormFoundry_A4_MarkerSheet_v1_1.pdf")


uploaded = st.file_uploader("Upload any full-body photo (for demo input display)", type=["jpg", "jpeg", "png"])


# =========================
# Demo Gallery
# =========================
if mode.startswith("Demo"):
    colL, colR = st.columns([1.1, 1.0], gap="large")

    pack_dir = packs.get(pack_id, DEMO_GALLERY_DIR / pack_id)
    preset = _load_preset(pack_dir)

    artifacts = preset.get("artifacts", {})
    note = preset.get("notes", {})
    # robust note handling (note may be dict or string)
    try:
        disclaimer = note.get("disclaimer", "Demo Gallery: preset outputs for stability.")
    except Exception:
        disclaimer = str(note) if note is not None else "Demo Gallery: preset outputs for stability."

    st.info(disclaimer)

    # Input image (either preset input.jpg if exists, or upload)
    input_img = None
    if use_upload_as_input:
        input_img = _read_upload(uploaded)
        if input_img is None:
            # fallback to preset input file
            input_img = _read_image(pack_dir / artifacts.get("input", "input.jpg"))
    else:
        input_img = _read_image(pack_dir / artifacts.get("input", "input.jpg"))

    with colL:
        st.markdown("## Layer 1 — Vision (Input)")
        st.markdown(f"**Demo Pack:** {preset.get('preset_id', pack_id)}  |  **Profile:** {preset.get('title', '-')}")
        if input_img is not None:
            st.image(input_img, caption="Input image (display only in Demo mode)", width='stretch')
        else:
            st.warning("Input image could not be displayed (missing cv2/numpy).")

        st.markdown("### Step 01 — PnP & Homography (Demo)")
        ph = _make_placeholder("PnP/Homography\n(visualization placeholder)\nDemo mode: preset outputs")
        if ph is not None:
            st.image(ph, caption="Reference-plane calibration visualization (demo)", width='stretch')
        else:
            st.caption("PnP/Homography visualization placeholder (no image backend).")

        st.markdown("### Step 02 — Pose (Demo)")
        ph2 = _make_placeholder("MediaPipe Pose\n(used for IK in full pipeline)\nDemo mode: preset outputs")
        if ph2 is not None:
            st.image(ph2, caption="Pose landmarks visualization (demo)", width='stretch')

        st.markdown("### Step 03 — Segmentation (Demo)")
        mask_img = _read_image(pack_dir / artifacts.get("mask", "step03_mask.png"))
        if mask_img is not None:
            st.image(mask_img, caption="Person/garment mask (demo artifact)", width='stretch')
        else:
            st.caption("Segmentation mask placeholder.")

    with colR:
        st.markdown("## Layer 2 — Middleware (Preset Output)")
        middleware = preset.get("middleware", {})
        body = middleware.get("body_specs_cm", {})

        # Compact output block (judge-friendly)
        st.markdown("### Body Specs (Output)")
        def _v(x):
            return "-" if x is None else f"{float(x):.1f} cm"

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Shoulder Width", _v(body.get("shoulder_width_cm")))
            st.metric("Chest Circumference", _v(body.get("chest_circ_cm")))
        with c2:
            st.metric("Waist Circumference", _v(body.get("waist_circ_cm")))
            st.metric("Hip Circumference", _v(body.get("hip_circ_cm")))

        st.markdown("### Clothing & Physics Signals")
        st.write({
            "garment_class": middleware.get("garment_class", "-"),
            "material": middleware.get("material", "-"),
            "material_props": middleware.get("material_props", {}),
            "volume_offset": middleware.get("volume_offset", {}),
            "warm_start": middleware.get("warm_start", {}),
            "user_height_cm (display)": float(user_height_cm),
        })

        st.markdown("### Export JSON")
        _download_json_button("Download preset middleware output JSON", preset, f"{preset.get('preset_id','demo')}_preset.json")

        st.markdown("---")
        st.markdown("## Layer 3 — 3D Engine (Demo Artifact)")
        drape_img = _read_image(pack_dir / artifacts.get("drape", "step06_drape.png"))
        if drape_img is not None:
            st.image(drape_img, caption="Draped garment render / 3D preview (demo artifact)", width='stretch')
        else:
            st.caption("3D preview placeholder.")

        st.markdown("## Layer 4 — GenAI Synthesis (Demo Artifacts)")
        edge_img = _read_image(pack_dir / artifacts.get("edge", "step06_edge.png"))
        depth_img = _read_image(pack_dir / artifacts.get("depth", "step06_depth.png"))
        final_img = _read_image(pack_dir / artifacts.get("final", "step06_final.png"))

        g1, g2 = st.columns(2)
        with g1:
            if edge_img is not None:
                st.image(edge_img, caption="ControlNet guide: Edge", width='stretch')
        with g2:
            if depth_img is not None:
                st.image(depth_img, caption="ControlNet guide: Depth", width='stretch')

        if final_img is not None:
            st.image(final_img, caption="Final photoreal output (demo artifact)", width='stretch')

        with st.expander("Full preset.json (for auditing)", expanded=False):
            st.json(preset, expanded=True)

    st.stop()


# =========================
# Live (Experimental) mode
# =========================
st.warning(
    "Live (Experimental) 모드는 현재 안정화 대상이 아닙니다. "
    "심사/시연은 Demo Gallery(Always Success)를 사용하세요."
)

st.markdown("### Live 모드 상태")
st.write({
    "cv2_available": CV2_OK,
    "numpy_available": (np is not None),
    "marker_pdf_exists": MARKER_PDF_PATH.exists(),
    "demo_gallery_packs_found": list(_list_demo_packs().keys()),
})

st.info("원하면 다음 단계로 Live 파이프라인(ArUco/Pose/YOLO)을 Demo 안정화와 분리해서 테스트 하네스부터 다시 잡아줄 수 있습니다.")
