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
    # =========================
    # Stable Demo Gallery (Always Success)
    # - Outputs are preset for stability.
    # - Visual artifacts below are generated offline from the same pipeline stack:
    #   OpenCV (PnP/Homography), MediaPipe Pose, YOLOv8-seg (mask), etc.
    # =========================
    import json
    from pathlib import Path

    def _resolve_demo_path(x, pack_dir: Path) -> Path | None:
        if not x:
            return None
        try:
            q = Path(str(x))
        except Exception:
            return None
        if q.is_absolute():
            return q
        xs = str(x)
        if xs.startswith("assets/"):
            return REPO_ROOT / xs
        return pack_dir / q

    def _pick_existing(pack_dir: Path, *cands: str) -> Path | None:
        for c in cands:
            if not c:
                continue
            pp = _resolve_demo_path(c, pack_dir)
            if pp and pp.exists():
                return pp
        return None

    def _safe_dict(x):
        return x if isinstance(x, dict) else {}

    def _get_num(d: dict, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except Exception:
                    pass
        return None

    # Layout
    colL, colR = st.columns([1.15, 0.85], gap="large")

    # Load pack preset
    pack_dir = packs.get(pack_id, DEMO_GALLERY_DIR / pack_id)
    preset = _load_preset(pack_dir)

    vision = _safe_dict(preset.get("vision", {}))
    middleware = _safe_dict(preset.get("middleware", {}))
    body = _safe_dict(middleware.get("body_specs_cm", {}))
    signals = _safe_dict(middleware.get("signals", {}))
    artifacts = _safe_dict(preset.get("artifacts", {}))
    notes = _safe_dict(preset.get("notes", {}))

    # Profile strings
    profile = _safe_dict(preset.get("profile", {})) or _safe_dict(preset.get("meta", {}))
    body_type = (profile.get("body_type") or profile.get("profile") or "Athletic")
    garment_class = (signals.get("garment_class") or middleware.get("garment_class") or profile.get("garment_class") or "-")
    material = (signals.get("material") or middleware.get("material") or profile.get("material") or "-")

    # Demo disclaimer
    disclaimer = notes.get("disclaimer") or "Demo Gallery outputs are preset for stability. Live inference is available under Experimental mode."
    st.info(disclaimer)

    # Resolve images (prefer preset fields, then common filenames)
    input_img_path = _pick_existing(pack_dir,
        vision.get("input_image"),
        "input.jpg", "input.png"
    )

    step01_path = _pick_existing(pack_dir,
        vision.get("pnp_homography_image"),
        artifacts.get("pnp"),
        artifacts.get("calib"),
        "step01_calib.jpg", "step01_calib.png",
        "pnp_homography.png", "pnp_homography.jpg",
        "pnp_homography.png", "pnp_homography.jpg",
        "pnp_homography.png"
    )

    step02_path = _pick_existing(pack_dir,
        vision.get("pose_landmarks_image"),
        artifacts.get("pose"),
        "step02_pose.jpg", "step02_pose.png",
        "pose.png", "pose.jpg"
    )

    step03_overlay_path = _pick_existing(pack_dir,
        artifacts.get("mask_overlay"),
        "step03_mask_overlay.jpg", "step03_mask_overlay.png"
    )
    step03_mask_path = _pick_existing(pack_dir,
        vision.get("segmentation_mask_image"),
        artifacts.get("mask"),
        "step03_mask.png", "mask.png"
    )

    drape_path = _pick_existing(pack_dir,
        artifacts.get("drape"),
        "layer3_drape.png", "draped.png", "layer3_drape.jpg"
    )
    edge_path = _pick_existing(pack_dir,
        artifacts.get("edge"),
        "layer4_edge.png", "edge.png", "layer4_edge.jpg"
    )
    depth_path = _pick_existing(pack_dir,
        artifacts.get("depth"),
        "layer4_depth.png", "depth.png", "layer4_depth.jpg"
    )
    final_path = _pick_existing(pack_dir,
        artifacts.get("final"),
        "layer4_final.jpg", "final.jpg", "layer4_final.png", "final.png"
    )

    # Numbers (support both *_cm and legacy keys)
    shoulder = _get_num(body, "shoulder_width_cm", "shoulder_width")
    chest = _get_num(body, "chest_circ_cm", "chest_circ")
    waist = _get_num(body, "waist_circ_cm", "waist_circ")
    hip = _get_num(body, "hip_circ_cm", "hip_circ")

    def _fmt_cm(x):
        return "-" if x is None else f"{x:.1f} cm"

    with colL:
        st.markdown("## Layer 1 — Vision (Input)")
        st.caption(f"Demo Pack: {pack_id} | Profile: {body_type} / {garment_class} | material: {material}")
        st.markdown("**Pipeline trace (Demo Artifact):** OpenCV (PnP/Homography) → MediaPipe Pose → YOLOv8-seg (Mask)")

        # Input image display
        if use_upload_as_input and uploaded is not None:
            try:
                st.image(uploaded, caption="Input image (display only in Demo mode)", width='stretch')
            except Exception:
                st.warning("Uploaded image could not be displayed.")
        elif input_img_path and input_img_path.exists():
            img = _read_image(input_img_path)
            if img is not None:
                st.image(img, caption="Input image (demo pack)", width='stretch')

        st.markdown("### Step 01 — PnP & Homography (Demo)")
        if step01_path and step01_path.exists():
            img = _read_image(step01_path)
            if img is not None:
                st.image(img, caption="OpenCV: reference-plane calibration (demo artifact)", width='stretch')
            else:
                st.warning(f"Unreadable image: {step01_path.name}")
        else:
            st.info("PnP/Homography visualization placeholder (demo artifact not found).")

        st.markdown("### Step 02 — Pose (Demo)")
        if step02_path and step02_path.exists():
            img = _read_image(step02_path)
            if img is not None:
                st.image(img, caption="MediaPipe Pose landmarks (demo artifact)", width='stretch')
            else:
                st.warning(f"Unreadable image: {step02_path.name}")
        else:
            st.info("Pose landmarks placeholder (demo artifact not found).")

        st.markdown("### Step 03 — Segmentation (Demo)")
        cA, cB = st.columns(2)
        with cA:
            if step03_overlay_path and step03_overlay_path.exists():
                img = _read_image(step03_overlay_path)
                if img is not None:
                    st.image(img, caption="YOLOv8-seg mask overlay (demo artifact)", width='stretch')
                else:
                    st.warning(f"Unreadable image: {step03_overlay_path.name}")
            else:
                st.info("Mask overlay placeholder (demo artifact not found).")
        with cB:
            if step03_mask_path and step03_mask_path.exists():
                img = _read_image(step03_mask_path)
                if img is not None:
                    st.image(img, caption="Binary mask (demo artifact)", width='stretch')
                else:
                    st.warning(f"Unreadable image: {step03_mask_path.name}")
            else:
                st.info("Mask placeholder (demo artifact not found).")

    with colR:
        st.markdown("## Layer 2 — Middleware (Preset Output)")
        st.markdown("### Body Specs (Output)")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Shoulder Width", _fmt_cm(shoulder))
            st.metric("Chest Circumference", _fmt_cm(chest))
        with m2:
            st.metric("Waist Circumference", _fmt_cm(waist))
            st.metric("Hip Circumference", _fmt_cm(hip))

        st.markdown("### Clothing & Physics Signals")
        st.json({
            "garment_class": garment_class,
            "material": material,
            "material_props": signals.get("material_props", {}),
            "volume_offset": signals.get("volume_offset", {}),
            "warm_start": signals.get("warm_start", {}),
            "user_height_cm (display)": float(user_height_cm),
        }, expanded=True)

        st.markdown("### Export JSON")
        _download_json_button(
            "Download preset (demo pack) JSON",
            preset,
            f"{pack_id}_preset.json"
        )

        st.markdown("---")
        st.markdown("## Layer 3 — 3D Engine (Demo Artifact)")
        if drape_path and drape_path.exists():
            img = _read_image(drape_path)
            if img is not None:
                st.image(img, caption="Draped garment render / 3D preview (demo artifact)", width='stretch')
            else:
                st.warning(f"Unreadable image: {drape_path.name}")
        else:
            st.info("Drape preview placeholder (demo artifact not found).")

        st.markdown("---")
        st.markdown("## Layer 4 — GenAI Synthesis (Demo Artifacts)")
        e1, e2 = st.columns(2)
        with e1:
            if edge_path and edge_path.exists():
                img = _read_image(edge_path)
                if img is not None:
                    st.image(img, caption="ControlNet guide: Edge", width='stretch')
                else:
                    st.warning(f"Unreadable image: {edge_path.name}")
            else:
                st.info("Edge placeholder (demo artifact not found).")
        with e2:
            if depth_path and depth_path.exists():
                img = _read_image(depth_path)
                if img is not None:
                    st.image(img, caption="ControlNet guide: Depth", width='stretch')
                else:
                    st.warning(f"Unreadable image: {depth_path.name}")
            else:
                st.info("Depth placeholder (demo artifact not found).")

        if final_path and final_path.exists():
            img = _read_image(final_path)
            if img is not None:
                st.image(img, caption="Final photoreal output (demo artifact)", width='stretch')
            else:
                st.warning(f"Unreadable image: {final_path.name}")
        else:
            st.info("Final output placeholder (demo artifact not found).")

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
