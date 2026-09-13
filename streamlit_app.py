# import streamlit as st
# import joblib
# from preprocessing import clean_text

# # Tải mô hình
# model = joblib.load('models/best_model.pkl')
# vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# st.title("🔍 Phát hiện Tin giả")
# st.write("Nhập một bài báo để kiểm tra xem đó là tin thật hay tin giả.")

# text = st.text_area("Nội dung bài báo:", height=200)

# if st.button("Kiểm tra"):
#     if text:
#         cleaned = clean_text(text)
#         vectorized = vectorizer.transform([cleaned])
#         prediction = model.predict(vectorized)[0]
#         probability = model.predict_proba(vectorized)[0]
        
#         if prediction == 1:
#             st.error(f"❌ TIN GIẢ (Độ tin cậy: {probability[1]*100:.2f}%)")
#         else:
#             st.success(f"✅ TIN THẬT (Độ tin cậy: {probability[0]*100:.2f}%)")
import streamlit as st
import joblib
from preprocessing import clean_text
import time

# ============================================================================
# CẤU HÌNH TRANG
# ============================================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS TÙY CHỈNH - LÀM ĐẸP GIAO DIỆN
# ============================================================================

st.markdown("""
<style>
    /* Font chữ chính */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Ẩn menu mặc định của Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tiêu đề chính */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Phụ đề */
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Khung nhập liệu */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        font-size: 1rem;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Nút bấm */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Thẻ kết quả */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        margin-top: 2rem;
        animation: slideIn 0.5s ease;
    }
    
    .result-fake {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 6px solid #dc2626;
    }
    
    .result-real {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 6px solid #059669;
    }
    
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-fake .result-label { color: #991b1b; }
    .result-real .result-label { color: #065f46; }
    
    .result-confidence {
        font-size: 1.2rem;
        font-weight: 500;
        color: #4b5563;
    }
    
    /* Thanh tiến độ tin cậy */
    .confidence-bar {
        height: 12px;
        border-radius: 6px;
        background: #e5e7eb;
        margin-top: 1rem;
        overflow: hidden;
    }
    
    .confidence-fill-fake {
        height: 100%;
        background: linear-gradient(90deg, #dc2626, #ef4444);
        border-radius: 6px;
        animation: fillBar 1s ease;
    }
    
    .confidence-fill-real {
        height: 100%;
        background: linear-gradient(90deg, #059669, #10b981);
        border-radius: 6px;
        animation: fillBar 1s ease;
    }
    
    /* Animation */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fillBar {
        from { width: 0%; }
    }
    
    /* Sidebar */
    .sidebar-card {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .sidebar-title {
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-text {
        font-size: 0.9rem;
        color: #6b7280;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# TẢI MÔ HÌNH
# ============================================================================

@st.cache_resource
def load_model():
    """Tải mô hình và vectorizer (cache để không load lại mỗi lần)"""
    model = joblib.load('models/best_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()


# ============================================================================
# HÀM DỰ ĐOÁN
# ============================================================================

def predict(text):
    """Dự đoán tin thật/giả từ văn bản"""
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    label = "FAKE" if prediction == 1 else "REAL"
    confidence = probability[prediction] * 100
    return label, confidence


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 📖 Hướng dẫn sử dụng")
    
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">1️⃣ Nhập bài báo</div>
        <div class="sidebar-text">Dán nội dung bài báo (tiếng Anh) vào ô nhập liệu.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">2️⃣ Bấm "Kiểm tra"</div>
        <div class="sidebar-text">Mô hình sẽ phân tích và đưa ra dự đoán.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">3️⃣ Xem kết quả</div>
        <div class="sidebar-text">Kết quả hiển thị: FAKE (tin giả) hoặc REAL (tin thật) kèm độ tin cậy.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Về mô hình")
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">Thuật toán</div>
        <div class="sidebar-text">Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">Đặc trưng</div>
        <div class="sidebar-text">TF-IDF (5000 từ quan trọng nhất)</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">Dữ liệu huấn luyện</div>
        <div class="sidebar-text">ISOT Dataset (44,898 bài báo)</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Lưu ý")
    st.caption("Mô hình chỉ chính xác với bài báo tiếng Anh về chủ đề chính trị - xã hội (giống dữ liệu huấn luyện).")


# ============================================================================
# GIAO DIỆN CHÍNH
# ============================================================================

# Tiêu đề
st.markdown('<h1 class="main-title">🔍 Phát Hiện Tin Giả</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Phát hiện tin giả bằng Machine Learning - Logistic Regression + TF-IDF</p>', unsafe_allow_html=True)

# Layout 2 cột
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 📝 Nhập bài báo cần kiểm tra:")
    text = st.text_area(
        "",
        height=200,
        placeholder="Dán nội dung bài báo tiếng Anh vào đây...\n\nVí dụ: 'Breaking: Scientists discover new cure for cancer overnight!'",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("#### 📋 Mẫu thử")
    st.caption("Click để dùng mẫu:")
    
    if st.button("📰 Mẫu tin thật"):
        st.session_state['sample'] = """WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a bill to increase funding for border security. The bill now heads to the House of Representatives for a vote. President Trump has indicated he will sign the bill if it reaches his desk."""
    
    if st.button("⚠️ Mẫu tin giả"):
        st.session_state['sample'] = """BREAKING: Obama secretly born in Kenya, new documents reveal! The mainstream media won't tell you this. Share this before it gets deleted! Wake up America!"""
    
    if 'sample' in st.session_state:
        text = st.session_state['sample']

# Nút kiểm tra
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    check_button = st.button("🔍 KIỂM TRA NGAY", use_container_width=True)

# Xử lý khi bấm nút
if check_button:
    if not text or not text.strip():
        st.warning("⚠️ Vui lòng nhập nội dung bài báo trước khi kiểm tra!")
    else:
        with st.spinner("🔄 Đang phân tích..."):
            time.sleep(0.5)  # Tạo hiệu ứng loading
            label, confidence = predict(text)
        
        # Hiển thị kết quả
        st.markdown("---")
        st.markdown("### 🎯 Kết quả phân tích:")
        
        if label == "FAKE":
            st.markdown(f"""
            <div class="result-card result-fake">
                <div class="result-label">❌ TIN GIẢ</div>
                <div class="result-confidence">Độ tin cậy: <strong>{confidence:.2f}%</strong></div>
                <div class="confidence-bar">
                    <div class="confidence-fill-fake" style="width: {confidence}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-real">
                <div class="result-label">✅ TIN THẬT</div>
                <div class="result-confidence">Độ tin cậy: <strong>{confidence:.2f}%</strong></div>
                <div class="confidence-bar">
                    <div class="confidence-fill-real" style="width: {confidence}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Chi tiết
        with st.expander("📊 Xem chi tiết phân tích"):
            st.write(f"**Văn bản gốc:** {len(text)} ký tự")
            st.write(f"**Sau khi làm sạch:** {len(clean_text(text))} ký tự")
            st.write(f"**Dự đoán:** {label}")
            st.write(f"**Độ tin cậy:** {confidence:.2f}%")
            
            # Hiển thị xác suất 2 lớp
            cleaned = clean_text(text)
            vectorized = vectorizer.transform([cleaned])
            probs = model.predict_proba(vectorized)[0]
            
            st.markdown("**Xác suất từng lớp:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("REAL (Tin thật)", f"{probs[0]*100:.2f}%")
            with col_b:
                st.metric("FAKE (Tin giả)", f"{probs[1]*100:.2f}%")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #9ca3af; font-size: 0.9rem;">'
    'Đồ án môn Khoa học Dữ liệu | Fake News Detection | 2026'
    '</p>',
    unsafe_allow_html=True
)