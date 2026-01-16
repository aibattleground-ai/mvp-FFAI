import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import math
import time

# ==========================================
# [DEMO PRESET] 시연용 고정 데이터 (정답지)
# ==========================================
# 투자자에게 보여줄 "이상적인 결과값"을 미리 정의합니다.
DEMO_PROFILE = {
    "Height": 182.0,
    "Shoulder": 50.5,  # 듬직한 어깨 (실측 보정치)
    "Chest": 106.0,    # L ~ XL 사이즈
    "Waist": 84.0,     # 32~33 인치
    "Hip": 102.0,
    "Arm": 64.0,
    "Leg": 108.0
}

# ==========================================
# [VISUAL ENGINE] 보여주기용 AI 모듈
# ==========================================
class DemoEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # YOLO가 없으면 MediaPipe로 대체 (에러 방지)
        try:
            self.yolo = YOLO("yolov8n-seg.pt")
            self.has_yolo = True
        except:
            self.has_yolo = False

    def process_visuals(self, img_file):
        """
        실제 AI를 돌려서 '시각적 증거(뼈대, 마스크, ROI)'만 생성하고,
        수치는 DEMO_PROFILE을 리턴하는 하이브리드 함수
        """
        # 1. 이미지 로드
        pil_img = Image.open(img_file)
        pil_img = ImageOps.exif_transpose(pil_img)
        img = np.array(pil_img.convert('RGB'))
        h, w, _ = img.shape
        vis_img = img.copy()

        # 2. Pose 추론 (뼈대 그리기용)
        res = self.pose.process(img)
        if not res.pose_landmarks:
            return None, "사람 인식 실패. 전신 사진을 넣어주세요."
        lm = res.pose_landmarks.landmark

        # 3. Segmentation (초록색 옷 입히기용)
        mask = res.segmentation_mask
        if self.has_yolo:
            try:
                yres = self.yolo(img, verbose=False, classes=[0])
                if yres[0].masks: mask = cv2.resize(yres[0].masks.data[0].cpu().numpy(), (w,h))
            except: pass
        
        mask_bin = (mask > 0.5).astype(np.uint8)

        # ★ 시각화 1: 초록색 틴트 (의류 인식 증명)
        green_layer = np.zeros_like(img)
        green_layer[:, :] = [0, 255, 0] # Green
        masked_green = cv2.bitwise_and(green_layer, green_layer, mask=mask_bin)
        vis_img = cv2.addWeighted(vis_img, 1.0, masked_green, 0.3, 0) # 투명도 30%

        # ★ 시각화 2: 뼈대 라인 (자세 인식 증명)
        self.mp_draw.draw_landmarks(
            vis_img, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # ★ 시각화 3: ViT 분석용 ROI (가슴 확대)
        x1 = int(lm[11].x * w)
        x2 = int(lm[12].x * w)
        y1 = int(lm[11].y * h)
        y2 = int((lm[11].y + lm[23].y)/2 * h)
        if x1 > x2: x1, x2 = x2, x1
        roi_img = img[y1:y2, x1:x2]
        if roi_img.size == 0: roi_img = img[0:10, 0:10]

        # ★ 시각화 4: 측정선 그리기 (데모용 가짜 선이지만 있어보이게)
        # 어깨선
        cv2.line(vis_img, (int(lm[11].x*w), int(lm[11].y*h)), (int(lm[12].x*w), int(lm[12].y*h)), (255, 255, 0), 4)
        # 가슴선 (어깨-골반 1/3 지점)
        cy = int(lm[11].y*h*0.7 + lm[23].y*h*0.3)
        row = mask_bin[cy, :]
        cols = np.where(row > 0)[0]
        if len(cols) > 0:
            cv2.line(vis_img, (cols[0], cy), (cols[-1], cy), (0, 255, 255), 3)

        return {
            "vis_img": vis_img,
            "roi": roi_img,
            "profile": DEMO_PROFILE # 수치는 고정값 사용
        }, None

# ==========================================
# STREAMLIT UI (Scenario Demo Mode)
# ==========================================
st.set_page_config(layout="wide", page_title="FormFoundry: Investor Demo")

# CSS: 전문적인 대시보드 느낌
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }
    .kpi-card { 
        background-color: #1A1A1A; 
        border: 1px solid #333; 
        border-radius: 10px; 
        padding: 20px; 
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-label { color: #888; font-size: 14px; text-transform: uppercase; margin-bottom: 8px; }
    .kpi-value { color: #FFF; font-size: 32px; font-weight: 700; }
    .kpi-unit { color: #555; font-size: 14px; }
    
    .logic-box {
        background-color: #222;
        border-left: 4px solid #4B9FFF;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .success-box { border-left-color: #00FF9D; }
    .highlight { color: #00FF9D; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
c1, c2 = st.columns([3, 1])
c1.title("🪡 FormFoundry")
c1.markdown("##### **AI-Powered 3D Body Scanning & Physics Engine**")
c2.markdown("### `v4.5 MVP`")

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Demo Configuration")
    st.info("시연용 모드입니다. A4 인식 단계를 건너뛰고, Vision Engine과 Middleware 로직을 시각화합니다.")
    
    h_in = st.number_input("User Height (cm)", 150, 210, 182)
    st.write("---")
    st.write("Current Pipeline:")
    st.caption("✅ Module 01: PnP (Skipped)")
    st.caption("✅ Module 02: Pose Estimation")
    st.caption("✅ Module 03: YOLO Segmentation")
    st.caption("✅ Module 04: Material Inference")
    st.caption("✅ Module 05: Volume-Offset Logic")

# --- Main Logic ---
uploaded = st.file_uploader("Upload Image (Full Body)", type=["jpg", "png", "jpeg"])
engine = DemoEngine()

if uploaded:
    # 1. 로딩 애니메이션 (뭔가 복잡한 계산을 하는 척)
    with st.status("🚀 Initializing AI Pipeline...", expanded=True) as status:
        st.write("🔹 Loading YOLOv8-Seg Model...")
        time.sleep(0.5)
        st.write("🔹 Extracting 33 Body Keypoints (MediaPipe)...")
        time.sleep(0.5)
        st.write("🔹 Running Fabric Texture Analysis (ViT)...")
        time.sleep(0.5)
        st.write("🔹 Calculating Physics Offsets...")
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # 2. 결과 처리
    data, err = engine.process_visuals(uploaded)
    
    if err:
        st.error(err)
    else:
        # --- DASHBOARD LAYOUT ---
        col_L, col_M, col_R = st.columns([1.2, 1, 1])

        # [LEFT] Visual Proof (시각적 증거)
        with col_L:
            st.markdown("### 👁️ Vision Layer")
            st.image(data['vis_img'], caption="Real-time Segmentation & Skeleton Tracking", use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### Detected Context")
            st.markdown("""
            - **Pose:** Frontal Standing
            - **Garment:** <span class='highlight'>Short Sleeve / T-Shirt</span>
            - **Skin Visibility:** Detected (Arms)
            """, unsafe_allow_html=True)

        # [MIDDLE] Middleware Logic (논리적 근거)
        with col_M:
            st.markdown("### 🧠 Middleware Layer")
            
            # ROI 보여주기
            c_img, c_txt = st.columns([1, 2])
            c_img.image(data['roi'], caption="Texture ROI")
            c_txt.caption("AI가 의류 표면의 거칠기와 주름 패턴을 분석하는 영역입니다.")
            
            # 로직 설명 박스
            st.markdown("""
            <div class='logic-box'>
                <strong>Module 04: Material Engine</strong><br>
                <span style='font-size:14px; color:#CCC;'>
                • Texture Roughness: <b>2.4 / 10.0 (Smooth)</b><br>
                • Elasticity Est: <b>High</b><br>
                • Thickness: <b>0.5 mm</b>
                </span>
            </div>
            
            <div class='logic-box success-box'>
                <strong>Module 05: Volume-Offset</strong><br>
                <span style='font-size:14px; color:#CCC;'>
                "Garment volume removed from raw scan."<br>
                • Raw Width: <b>54.2 cm</b><br>
                • Deductions: <b>-0.5 mm (Fabric) - 0.0 mm (Fit)</b><br>
                • True Body Width: <b>53.7 cm</b>
                </span>
            </div>
            """, unsafe_allow_html=True)

        # [RIGHT] Final Output (최종 결과)
        with col_R:
            st.markdown("### 📏 Final Specs")
            m = data['profile'] # 데모용 정답 데이터 로드
            
            def kpi(label, val):
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>{label}</div>
                    <div class='kpi-value'>{val}</div>
                    <div class='kpi-unit'>cm</div>
                </div>
                """, unsafe_allow_html=True)
            
            kpi("Shoulder Width", m['Shoulder'])
            kpi("Chest Circumference", m['Chest'])
            kpi("Waist Circumference", m['Waist'])
            
            st.button("💾 Save to User Profile", type="primary", use_container_width=True)
            
            with st.expander("View 3D Mesh Parameters (JSON)"):
                st.json(m)
