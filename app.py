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
    """
    Read an image for UI display.
    - PIL-first for Streamlit Cloud stability (cv2 may be missing / unreliable)
    - cv2 fallback if available
    Returns: PIL.Image.Image or numpy RGB array, or None.
    """
    try:
        from PIL import Image, ImageOps
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except Exception:
        pass

    if not CV2_OK:
        return None

    try:
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None




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
    # - Visuals are demo artifacts (or demo overlays on uploaded photo).
    # =========================
    from pathlib import Path
    from PIL import Image, ImageOps, ImageDraw

    def _safe_dict(x):
        return x if isinstance(x, dict) else {}

    def _resolve_demo_path(x, pack_dir: Path) -> Optional[Path]:
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

    def _try_read(pack_dir: Path, *cands: str):
        # returns (img, filename) for first readable candidate
        for c in cands:
            if not c:
                continue
            pp = _resolve_demo_path(c, pack_dir)
            if pp and pp.exists():
                img = _read_image(pp)
                if img is not None:
                    return img, pp.name
        return None, None

    def _get_num(d: dict, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except Exception:
                    pass
        return None

    def _fmt_cm(x):
        return "-" if x is None else f"{x:.1f} cm"

    def _pil_from_upload(uploaded_file):
        if uploaded_file is None:
            return None
        try:
            img = Image.open(uploaded_file)
            img = ImageOps.exif_transpose(img).convert("RGB")
            return img
        except Exception:
            return None

    # --- Demo overlays (simulate OpenCV/MediaPipe/YOLO visuals on the uploaded photo) ---
    def _overlay_pnp(img: Image.Image) -> Image.Image:
        out = img.copy()
        d = ImageDraw.Draw(out)
        w, h = out.size
        # rectangle ~A4 on torso (demo)
        cx, cy = int(0.50*w), int(0.55*h)
        rw, rh = int(0.18*w), int(0.28*h)
        x1, y1, x2, y2 = cx - rw, cy - rh, cx + rw, cy + rh
        d.rectangle([x1, y1, x2, y2], outline="#00d4ff", width=5)
        # corners (PnP)
        r = max(6, int(min(w, h)*0.008))
        for (x, y) in [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]:
            d.ellipse([x-r, y-r, x+r, y+r], outline="#ffffff", width=4)
        d.text((x1, max(0, y1-28)), "OpenCV: Reference Plane (Demo Overlay)", fill="#ffffff")
        return out

    def _overlay_pose(img: Image.Image) -> Image.Image:
        out = img.copy()
        d = ImageDraw.Draw(out)
        w, h = out.size

        # simple skeleton (normalized) to look like MediaPipe overlay
        pts = {
            "head": (0.50, 0.18),
            "ls": (0.42, 0.30), "rs": (0.58, 0.30),
            "le": (0.36, 0.42), "re": (0.64, 0.42),
            "lw": (0.33, 0.55), "rw": (0.67, 0.55),
            "lh": (0.46, 0.55), "rh": (0.54, 0.55),
            "lk": (0.45, 0.72), "rk": (0.55, 0.72),
            "la": (0.44, 0.90), "ra": (0.56, 0.90),
        }
        def P(k):
            x,y = pts[k]
            return (int(x*w), int(y*h))

        edges = [
            ("head","ls"),("head","rs"),
            ("ls","rs"),
            ("ls","le"),("le","lw"),
            ("rs","re"),("re","rw"),
            ("ls","lh"),("rs","rh"),
            ("lh","rh"),
            ("lh","lk"),("lk","la"),
            ("rh","rk"),("rk","ra"),
        ]

        for a,b in edges:
            d.line([P(a), P(b)], fill="#00d4ff", width=5)

        r = max(6, int(min(w, h)*0.008))
        for k in pts:
            x,y = P(k)
            d.ellipse([x-r, y-r, x+r, y+r], fill="#ffffff", outline="#ffffff")
        d.text((int(0.04*w), int(0.04*h)), "MediaPipe Pose (Demo Overlay)", fill="#ffffff")
        return out

    def _overlay_mask(img: Image.Image):
        # returns (overlay_img, binary_mask_img)
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)

        # silhouette-like shape (demo)
        d.ellipse([int(0.38*w), int(0.10*h), int(0.62*w), int(0.28*h)], fill=255)  # head-ish
        d.rounded_rectangle([int(0.32*w), int(0.22*h), int(0.68*w), int(0.92*h)], radius=60, fill=255)
        d.rounded_rectangle([int(0.22*w), int(0.30*h), int(0.32*w), int(0.62*h)], radius=40, fill=255)  # left arm
        d.rounded_rectangle([int(0.68*w), int(0.30*h), int(0.78*w), int(0.62*h)], radius=40, fill=255)  # right arm

        overlay = img.copy().convert("RGBA")
        tint = Image.new("RGBA", (w, h), (0, 212, 255, 90))
        overlay.paste(tint, (0,0), mask)
        overlay = overlay.convert("RGB")

        binary = Image.new("RGB", (w, h), (0,0,0))
        binary.paste((255,255,255), (0,0), mask)
        return overlay, binary

    # -------------------------
    # Run gating (upload -> run -> show)
    # -------------------------
    if "demo_ran" not in st.session_state:
        st.session_state.demo_ran = False
    demo_key = (pack_id, getattr(uploaded, "name", None))
    if st.session_state.get("demo_key") != demo_key:
        st.session_state.demo_key = demo_key
        st.session_state.demo_ran = False

    run_clicked = st.button("Run Demo (simulate end-to-end)", type="primary", width='stretch')
    if run_clicked:
        st.session_state.demo_ran = True

    # Layout
    colL, colR = st.columns([1.15, 0.85], gap="large")

    # Load preset pack
    pack_dir = packs.get(pack_id, DEMO_GALLERY_DIR / pack_id)
    preset = _load_preset(pack_dir)

    vision = _safe_dict(preset.get("vision", {}))
    middleware = _safe_dict(preset.get("middleware", {}))
    body = _safe_dict(middleware.get("body_specs_cm", {}))
    signals = _safe_dict(middleware.get("signals", {}))
    artifacts = _safe_dict(preset.get("artifacts", {}))
    notes = _safe_dict(preset.get("notes", {}))
    profile = _safe_dict(preset.get("profile", {})) or _safe_dict(preset.get("meta", {}))

    body_type = (profile.get("body_type") or profile.get("profile") or "Athletic")
    garment_class = (signals.get("garment_class") or middleware.get("garment_class") or profile.get("garment_class") or "-")
    material = (signals.get("material") or middleware.get("material") or profile.get("material") or "-")

    disclaimer = notes.get("disclaimer") or "Demo Gallery outputs are preset for stability. Live inference is available under Experimental mode."
    st.info(disclaimer)

    # Input image (prefer upload for display)
    up_img = _pil_from_upload(uploaded) if (use_upload_as_input and uploaded is not None) else None
    if up_img is None:
        # fallback to demo pack input if exists
        demo_input, _ = _try_read(pack_dir,
            vision.get("input_image"),
            "input.jpg", "input.png"
        )
    else:
        demo_input = up_img

    with colL:
        st.markdown("## Layer 1 — Vision (Input)")
        st.caption(f"Demo Pack: {pack_id} | Profile: {body_type} / {garment_class} | material: {material}")
        st.markdown("**Pipeline trace (Demo):** OpenCV (PnP/Homography) → MediaPipe Pose → YOLOv8-seg (Mask)")

        if demo_input is not None:
            st.image(demo_input, caption="Input image (display only in Demo mode)", width='stretch')
        else:
            st.warning("Upload an image to preview input in Demo mode.")

        if not st.session_state.demo_ran:
            st.warning("1) 사진 업로드  2) Run Demo 버튼 클릭  → 그 다음에 Step 01/02/03, 결과, 아티팩트를 표시합니다.")
            st.stop()

        # Step 01/02/03: prefer overlay on uploaded photo; else use pack artifacts
        st.markdown("### Step 01 — PnP & Homography (Demo)")
        if up_img is not None:
            st.image(_overlay_pnp(up_img), caption="OpenCV: reference-plane calibration (demo overlay)", width='stretch')
        else:
            img, name = _try_read(pack_dir,
                vision.get("pnp_homography_image"),
                artifacts.get("pnp"), artifacts.get("calib"),
                "step01_calib.jpg","step01_calib.png",
                "pnp_homography.png","pnp_homography.jpg"
            )
            if img is not None:
                st.image(img, caption=f"OpenCV: reference-plane calibration (demo artifact: {name})", width='stretch')
            else:
                st.warning("Unreadable image: step01 artifact")

        st.markdown("### Step 02 — Pose (Demo)")
        if up_img is not None:
            st.image(_overlay_pose(up_img), caption="MediaPipe Pose landmarks (demo overlay)", width='stretch')
        else:
            img, name = _try_read(pack_dir,
                vision.get("pose_landmarks_image"),
                artifacts.get("pose"),
                "step02_pose.jpg","step02_pose.png",
                "pose.png","pose.jpg"
            )
            if img is not None:
                st.image(img, caption=f"MediaPipe Pose landmarks (demo artifact: {name})", width='stretch')
            else:
                st.warning("Unreadable image: step02 artifact")

        st.markdown("### Step 03 — Segmentation (Demo)")
        cA, cB = st.columns(2)
        if up_img is not None:
            ov, bi = _overlay_mask(up_img)
            with cA:
                st.image(ov, caption="YOLOv8-seg mask overlay (demo overlay)", width='stretch')
            with cB:
                st.image(bi, caption="Binary mask (demo overlay)", width='stretch')
        else:
            overlay, oname = _try_read(pack_dir,
                artifacts.get("mask_overlay"),
                "step03_mask_overlay.jpg","step03_mask_overlay.png"
            )
            mask, mname = _try_read(pack_dir,
                vision.get("segmentation_mask_image"),
                artifacts.get("mask"),
                "step03_mask.png","mask.png"
            )
            with cA:
                if overlay is not None:
                    st.image(overlay, caption=f"YOLOv8-seg mask overlay (artifact: {oname})", width='stretch')
                else:
                    st.warning("Unreadable image: mask overlay")
            with cB:
                if mask is not None:
                    st.image(mask, caption=f"Binary mask (artifact: {mname})", width='stretch')
                else:
                    st.warning("Unreadable image: mask")

    # Numbers
    shoulder = _get_num(body, "shoulder_width_cm", "shoulder_width")
    chest = _get_num(body, "chest_circ_cm", "chest_circ")
    waist = _get_num(body, "waist_circ_cm", "waist_circ")
    hip = _get_num(body, "hip_circ_cm", "hip_circ")

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
        _download_json_button("Download preset (demo pack) JSON", preset, f"{pack_id}_preset.json")

        st.markdown("---")
        st.markdown("## Layer 3 — 3D Engine (Demo Artifact)")
        drape, dname = _try_read(pack_dir,
            artifacts.get("drape"),
            "layer3_drape.png","layer3_drape.jpg","draped.png","layer3_drape.png"
        )
        if drape is not None:
            st.image(drape, caption=f"Draped garment render / 3D preview (artifact: {dname})", width='stretch')
        else:
            st.warning("Unreadable image: Layer3 drape")

        st.markdown("---")
        st.markdown("## Layer 4 — GenAI Synthesis (Demo Artifacts)")
        edge, ename = _try_read(pack_dir, artifacts.get("edge"), "layer4_edge.png","layer4_edge.jpg","edge.png","edge.jpg")
        depth, zname = _try_read(pack_dir, artifacts.get("depth"), "layer4_depth.png","layer4_depth.jpg","depth.png","depth.jpg")
        final, fname = _try_read(pack_dir, artifacts.get("final"), "layer4_final.jpg","final.jpg","layer4_final.png","final.png")

        e1, e2 = st.columns(2)
        with e1:
            if edge is not None:
                st.image(edge, caption=f"ControlNet guide: Edge (artifact: {ename})", width='stretch')
            else:
                st.warning("Unreadable image: Edge")
        with e2:
            if depth is not None:
                st.image(depth, caption=f"ControlNet guide: Depth (artifact: {zname})", width='stretch')
            else:
                st.warning("Unreadable image: Depth")

        if final is not None:
            st.image(final, caption=f"Final photoreal output (artifact: {fname})", width='stretch')
        else:
            st.warning("Unreadable image: Final output")

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
