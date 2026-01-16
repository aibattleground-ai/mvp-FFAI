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



def _normalize_preset(preset: dict, pack_dir: Path) -> dict:
    """
    Make Demo Gallery robust across preset schema variants.
    - Accepts either v0 (signals/body keys without _cm suffix, no artifacts mapping)
    - Produces a normalized dict that the UI expects.
    """
    if not isinstance(preset, dict):
        preset = {}
    p = dict(preset)

    # id / title
    pid = p.get("preset_id") or p.get("id") or pack_dir.name
    p["preset_id"] = pid

    # notes (note -> notes)
    if "notes" not in p and "note" in p:
        p["notes"] = p.get("note")
    if not isinstance(p.get("notes"), dict):
        p["notes"] = {}

    # profile label (avoid "Demo Pack: ... | Profile: Demo Pack: ...")
    profile = p.get("profile") if isinstance(p.get("profile"), dict) else {}
    body_type = profile.get("body_type") or profile.get("profile") or ""
    garment = profile.get("garment_class") or ""
    label = " / ".join([x for x in [str(body_type).strip(), str(garment).strip()] if x]) or str(pid)

    title = p.get("title")
    if (not isinstance(title, str)) or (not title.strip()) or ("Demo Pack:" in title):
        p["title"] = label

    # middleware normalization
    mw = p.get("middleware") if isinstance(p.get("middleware"), dict) else {}
    mw = dict(mw)

    signals = mw.get("signals") if isinstance(mw.get("signals"), dict) else {}

    body = mw.get("body_specs_cm") if isinstance(mw.get("body_specs_cm"), dict) else {}
    body = dict(body)

    # copy aliases both ways
    alias_pairs = {
        "shoulder_width_cm": "shoulder_width",
        "chest_circ_cm": "chest_circ",
        "waist_circ_cm": "waist_circ",
        "hip_circ_cm": "hip_circ",
    }
    for dst, src in alias_pairs.items():
        if dst not in body and src in body:
            body[dst] = body.get(src)
        if src not in body and dst in body:
            body[src] = body.get(dst)

    mw["body_specs_cm"] = body

    # flatten common signal keys
    for k in ["garment_class", "material", "material_props", "volume_offset", "warm_start"]:
        v = mw.get(k)
        if v in (None, "", {}, []):
            if k in signals:
                mw[k] = signals.get(k)

    # ensure dicts
    for k in ["material_props", "volume_offset", "warm_start"]:
        if not isinstance(mw.get(k), dict):
            mw[k] = {}

    p["middleware"] = mw

    # artifacts normalization (support either explicit mapping or implicit filenames)
    artifacts = p.get("artifacts") if isinstance(p.get("artifacts"), dict) else {}
    artifacts = dict(artifacts)

    def pick_existing(names):
        for n in names:
            if isinstance(n, str) and (pack_dir / n).exists():
                return n
        return None

    def set_if_missing(key, candidates):
        if key in artifacts and isinstance(artifacts.get(key), str) and artifacts.get(key):
            return
        v = pick_existing(candidates)
        if v:
            artifacts[key] = v

    # from vision paths (store as filename within pack_dir)
    vision = p.get("vision") if isinstance(p.get("vision"), dict) else {}
    v_calib = vision.get("pnp_homography_image")
    v_pose = vision.get("pose_landmarks_image")
    v_mask = vision.get("segmentation_mask_image")
    if isinstance(v_calib, str):
        artifacts.setdefault("calib", Path(v_calib).name)
    if isinstance(v_pose, str):
        artifacts.setdefault("pose", Path(v_pose).name)
    if isinstance(v_mask, str):
        artifacts.setdefault("mask", Path(v_mask).name)

    # fallbacks by common filenames in your repo
    set_if_missing("input", ["input.jpg", "input.png"])
    set_if_missing("calib", ["step01_calib.jpg", "step01_calib.png", "pnp_homography.png"])
    set_if_missing("pose", ["pose.png", "step02_pose.jpg", "step02_pose.png"])
    set_if_missing("mask", ["mask.png", "step03_mask.png"])
    set_if_missing("mask_overlay", ["step03_mask_overlay.jpg", "step03_mask_overlay.png"])
    set_if_missing("drape", ["layer3_drape.png", "layer3_drape.jpg", "draped.png", "draped.jpg"])
    set_if_missing("edge", ["layer4_edge.png", "layer4_edge.jpg", "edge.png", "edge.jpg"])
    set_if_missing("depth", ["layer4_depth.png", "layer4_depth.jpg", "depth.png", "depth.jpg"])
    set_if_missing("final", ["layer4_final.jpg", "layer4_final.png", "final.png", "final.jpg"])

    # remove invalid artifact entries (avoid Path / None)
    for k in list(artifacts.keys()):
        if not isinstance(artifacts.get(k), str) or not artifacts.get(k):
            del artifacts[k]

    p["artifacts"] = artifacts
    return p


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
    preset = _normalize_preset(_load_preset(pack_dir), pack_dir)
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
