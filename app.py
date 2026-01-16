import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import time
import requests
from io import BytesIO

# ==========================================
# [PRESET DATA] 데모용 고정 데이터
# ==========================================
DEMO_SPECS = {
    "Height": 182.0,
    "Shoulder": 50.5,  # 듬직한 어깨
    "Chest": 105.0,    # L~XL 사이즈
    "Waist": 82.0,     # 32인치
    "Hip": 100.0,
    "Arm": 62.0,
    "Leg": 108.0
}

GARMENT_ASSETS = {
    "Hoodie (Grey)": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Grey_Hoodie_Front.jpg/800px-Grey_Hoodie_Front.jpg",
    "T-Shirt (White)": "https://upload.wikimedia.org/wikipedia/commons/2/24/Blue_Tshirt.jpg" 
}

# ==========================================
# [CSS] Commercial & Dark UI
# ==========================================
st.set_page_config(layout="wide", page_title="FormFoundry: Pipeline Demo")

st.markdown("""
<style>
    .main-header { font-size: 30px; font-weight: bold; margin-bottom: 20px; }
    .module-box { 
        background-color: #1A1A1A; 
        border: 1px solid #333; 
        border-radius: 8px; 
        padding: 15px; 
        margin-bottom: 10px; 
    }
    .module-title { color: #4B9FFF; font-size: 14px; font-weight: bold; text-transform: uppercase; margin-bottom: 5px; }
    .module-desc { color: #CCC; font-size: 12px; line-height: 1.4; }
    .highlight { color: #00FF9D; font-weight: bold; }
    .tech-tag {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 10px; font-weight: bold; background: #333; color: #FFF; margin-right: 5px;
    }
    .step-header { font-size: 20px; font-weight: bold; margin-top: 20px; margin-bottom: 10px; border-bottom: 1px solid #444; padding-bottom: 5px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# [HELPER FUNCTIONS]
# ==========================================
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

def overlay_images(background, overlay, x_offset=0, y_offset=0, scale=1.0):
    """
    [GenAI Simulation] 2D 오버레이로 합성을 '흉내' 냅니다.
    실제 서비스 단계에선 SDXL Inpainting API로 대체됩니다.
    """
    bg_h, bg_w, _ = background.shape
    ov_h, ov_w, _ = overlay.shape
    
    # Resize Overlay
    new_w = int(bg_w * scale)
    new_h = int(ov_h * (new_w / ov_w))
    overlay_resized = cv2.resize(overlay, (new_w, new_h))
    
    # Center Position
    x_center = (bg_w - new_w) // 2 + x_offset
    y_center = (bg_h - new_h) // 2 + y_offset - 100 # 가슴 쪽에 위치
    
    # Overlay Logic
    result = background.copy()
    
    # ROI 설정
    y1, y2 = max(0, y_center), min(bg_h, y_center + new_h)
    x1, x2 = max(0, x_center), min(bg_w, x_center + new_w)
    
    ov_y1, ov_y2 = max(0, -y_center), min(new_h, bg_h - y_center)
    ov_x1, ov_x2 = max(0, -x_center), min(new_w, bg_w - x_center)
    
    if y1 >= y2 or x1 >= x2 or ov_y1 >= ov_y2 or ov_x1 >= ov_x2:
        return result

    alpha_s = overlay_resized[ov_y1:ov_y2, ov_x1:ov_x2, :] / 255.0 if overlay_resized.shape[2] == 3 else 1.0
    alpha_l = 1.0 - 0.3 # 투명도 줘서 자연스럽게
    
    for c in range(0, 3):
        result[y1:y2, x1:x2, c] = (overlay_resized[ov_y1:ov_y2, ov_x1:ov_x2, c] * 0.7 + result[y1:y2, x1:x2, c] * 0.3)
        
    return result

# ==========================================
# [MAIN APP]
# ==========================================

# 사이드바 (설정)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2503/2503509.png", width=50)
    st.header("FormFoundry Control")
    st.caption("v4.5 Investor Demo Build")
    
    st.divider()
    st.subheader("1. User Profile")
    st.info("👤 **Demo User (Nathan)**\nHeight: 182cm\nWeight: 78kg")
    
    st.divider()
    st.subheader("2. Target Item")
    selected_garment = st.selectbox("Select Garment", list(GARMENT_ASSETS.keys()))
    
    st.divider()
    st.success("✅ System Ready\n• GPU: Simulated\n• Pipeline: Active")

# 메인 타이틀
st.title("🪡 FormFoundry: The Future of Virtual Fit")
st.markdown("**End-to-End Pipeline Visualization:** Vision → Middleware → 3D Physics → GenAI Synthesis")

# 파일 업로드 (시작 트리거)
uploaded_file = st.file_uploader("Upload User Photo (Frontal)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 이미지 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    user_img = cv2.imdecode(file_bytes, 1)
    user_img = cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)
    
    # 탭 구성 (파이프라인 단계별 시각화)
    tab1, tab2, tab3, tab4 = st.tabs(["👁️ Layer 1: Vision", "🧠 Layer 2: Middleware", "👕 Layer 3: 3D Engine", "✨ Layer 4: GenAI"])

    # --------------------------------------------------------
    # [LAYER 1] Vision (Scanning)
    # --------------------------------------------------------
    with tab1:
        st.markdown("<div class='step-header'>STEP 01: Precision Scanning (Vision)</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(user_img, caption="Input Raw Data", use_container_width=True)
        
        with col2:
            with st.status("Running Vision Modules...", expanded=True) as status:
                time.sleep(0.5)
                st.write("🔹 **Module 01 (PnP):** Reference Object Detected (A4)")
                st.write("🔹 **Module 02 (Pose):** MediaPipe 33 Keypoints Extracted")
                time.sleep(0.5)
                st.write("🔹 **Module 03 (YOLO):** Segmentation Mask Generated")
                st.write("🔹 **Module 04 (ViT):** Fabric Material Inferred (Cotton/Terry)")
                status.update(label="Vision Processing Complete", state="complete", expanded=False)
            
            st.markdown(f"""
            <div class='module-box'>
                <div class='module-title'>Module 04: Material Inference</div>
                <div class='module-desc'>
                    Detected: <span class='highlight'>Heavy Cotton / Fleece</span><br>
                    Texture Roughness: 7.5/10 (High)<br>
                    Elasticity: Low (Stiff)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("✅ Vision Layer has extracted geometry and material context.")

    # --------------------------------------------------------
    # [LAYER 2] Middleware (Logic)
    # --------------------------------------------------------
    with tab2:
        st.markdown("<div class='step-header'>STEP 02: Physics Translation (Middleware)</div>", unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown(f"""
            <div class='module-box' style='border-left: 4px solid #00FF9D;'>
                <div class='module-title'>Module 05: Volume-Offset Logic</div>
                <div class='module-desc'>
                    "Vision measurements include clothing volume."<br>
                    <br>
                    • Raw Chest Width: <b>62.0 cm</b><br>
                    • Material Deduction: <b>-5.0 mm</b> (Hoodie)<br>
                    • Air-Gap Deduction: <b>-3.0 cm</b> (Loose Fit)<br>
                    <hr style='margin:5px 0; border-color:#333;'>
                    ➤ <b>True Body Width: 58.5 cm</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            st.markdown(f"""
            <div class='module-box' style='border-left: 4px solid #FFA500;'>
                <div class='module-title'>Module 06: Deformation Engine</div>
                <div class='module-desc'>
                    "Predicting drapery behavior."<br>
                    <br>
                    • Warm-Start Params: <b>Loaded</b><br>
                    • Collision Hints: <b>Generated</b><br>
                    • Simulation Time Saved: <b>~85%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("📊 Calibrated Body Specs (Output)")
        # 프리셋 데이터 표시
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shoulder", f"{DEMO_SPECS['Shoulder']} cm", "Calibrated")
        c2.metric("Chest", f"{DEMO_SPECS['Chest']} cm", "Est. Circumference")
        c3.metric("Waist", f"{DEMO_SPECS['Waist']} cm", "Est. Circumference")
        c4.metric("Arm Length", f"{DEMO_SPECS['Arm']} cm", "Bone Length")

    # --------------------------------------------------------
    # [LAYER 3] 3D Engine (Simulation)
    # --------------------------------------------------------
    with tab3:
        st.markdown("<div class='step-header'>STEP 03: Digital Twin & Physics (3D)</div>", unsafe_allow_html=True)
        
        col_3d_1, col_3d_2 = st.columns([1, 1])
        
        with col_3d_1:
            st.markdown("##### 🧬 SMPL-X Body Mesh Generation")
            # 3D 메시 느낌의 플레이스홀더 (여기서는 텍스트/이미지로 대체)
            st.markdown("""
            <div style='text-align:center; padding: 40px; background:#111; border-radius:10px; border: 1px dashed #555;'>
                <h3 style='color:#666;'>[ 3D Mesh Reconstruction ]</h3>
                <p style='color:#444;'>Applying SMPL-X Parameters...<br>Syncing IK Pose...</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_3d_2:
            st.markdown("##### 🧶 CLO3D Physics Simulation")
            st.markdown("""
            <div class='module-box'>
                <div class='module-title'>Module 07: Synthetic Data Factory</div>
                <div class='module-desc'>
                    Running headless CLO3D simulation...<br>
                    Applying gravity, friction, and fabric stiffness.<br>
                    <br>
                    <span class='highlight'>Result: Fit-Consistent 3D Mesh Ready.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 진행바 시뮬레이션
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text="Simulating Cloth Physics...")
            st.success("Physics Simulation Converged.")

    # --------------------------------------------------------
    # [LAYER 4] GenAI (Synthesis)
    # --------------------------------------------------------
    with tab4:
        st.markdown("<div class='step-header'>STEP 04: Photorealistic Synthesis (GenAI)</div>", unsafe_allow_html=True)
        
        col_g1, col_g2 = st.columns([1, 1])
        
        # 옷 이미지 로드 (가상)
        garment_url = GARMENT_ASSETS[selected_garment]
        try:
            garment_img = load_image_from_url(garment_url)
        except:
            garment_img = np.zeros((200,200,3), dtype=np.uint8) # Fallback

        with col_g1:
            st.image(garment_img, caption="Selected Garment (3D Pattern)", width=200)
            st.markdown("""
            **Module 10: SDXL + ControlNet**
            - Input: 3D Draped Mesh Render
            - Conditioning: Canny Edge + Depth Map
            - Prompt: "Photorealistic, Studio Lighting"
            """)
            
            if st.button("✨ Generate Virtual Try-On", type="primary"):
                with st.spinner("Synthesizing via Stable Diffusion XL..."):
                    time.sleep(2.0) # 생성 시간 시뮬레이션
                    
                    # [MVP Trick] 단순 오버레이로 합성 '흉내' (실제 API 연결 시 여기서 호출)
                    # 실제로는 Replicate IDM-VTON API 등을 쓰면 됨
                    result_img = overlay_images(user_img, garment_img, scale=0.6, y_offset=50)
                    
                    st.session_state['vto_result'] = result_img
                    st.session_state['vto_done'] = True

        with col_g2:
            if 'vto_done' in st.session_state and st.session_state['vto_done']:
                st.image(st.session_state['vto_result'], caption="Final Synthesis Result", use_container_width=True)
                st.balloons()
                st.success("Virtual Try-On Complete.")
            else:
                st.info("Click 'Generate' to see the final output.")
                # 빈 공간 홀더
                st.markdown("""
                <div style='height:300px; display:flex; align-items:center; justify-content:center; background:#111; color:#333; border-radius:10px;'>
                    Waiting for Generation...
                </div>
                """, unsafe_allow_html=True)

else:
    # 초기 화면
    st.info("👋 Welcome to FormFoundry MVP Demo. Please upload an image to start the pipeline.")
    
    # Tech Stack Preview
    st.markdown("### Powered by")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("##### 👁️ Vision\nMediaPipe, YOLOv8")
    c2.markdown("##### 🧠 Logic\nVolume-Offset, ViT")
    c3.markdown("##### 👕 3D\nSMPL-X, CLO3D")
    c4.markdown("##### ✨ GenAI\nSDXL, ControlNet")
