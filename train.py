# ============================================================================
# IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ============================================================================

# pandas: Thư viện xử lý dữ liệu dạng bảng (đọc CSV, ghép bảng, thống kê)
import pandas as pd

# train_test_split: Hàm chia dữ liệu thành 2 phần: huấn luyện và kiểm tra
from sklearn.model_selection import train_test_split

# TfidfVectorizer: Công cụ biến văn bản thành vector số bằng phương pháp TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# LogisticRegression: Thuật toán học máy dùng để phân loại (FAKE/REAL)
from sklearn.linear_model import LogisticRegression

# Các hàm đánh giá mô hình:
# - classification_report: In báo cáo Precision/Recall/F1 cho từng lớp
# - accuracy_score: Tính độ chính xác tổng thể
# - confusion_matrix: Tạo ma trận nhầm lẫn
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# matplotlib.pyplot: Thư viện vẽ biểu đồ cơ bản
import matplotlib.pyplot as plt

# seaborn: Thư viện vẽ biểu đồ đẹp hơn, dùng để vẽ heatmap cho confusion matrix
import seaborn as sns

# joblib: Thư viện dùng để lưu và tải mô hình đã huấn luyện (dạng file .pkl)
import joblib

# os: Thư viện tương tác với hệ điều hành (tạo thư mục, kiểm tra file)
import os

# time: Thư viện đo thời gian chạy của từng bước
import time

# tqdm: Thư viện tạo thanh tiến độ (progress bar) khi xử lý dữ liệu lớn
from tqdm import tqdm

# clean_text: Hàm làm sạch văn bản mà chúng ta đã viết trong file preprocessing.py
from preprocessing import clean_text


# ============================================================================
# KÍCH HOẠT THANH TIẾN ĐỘ CHO PANDAS
# ============================================================================

# tqdm.pandas() cho phép dùng .progress_apply() thay vì .apply()
# để hiển thị thanh tiến độ khi xử lý từng dòng dữ liệu
tqdm.pandas()


# ============================================================================
# TẠO THƯ MỤC NẾU CHƯA CÓ
# ============================================================================

# os.makedirs(): Tạo thư mục nếu chưa tồn tại
# exist_ok=True: Không báo lỗi nếu thư mục đã tồn tại
os.makedirs('models', exist_ok=True)   # Thư mục lưu mô hình
os.makedirs('reports', exist_ok=True)  # Thư mục lưu báo cáo và biểu đồ


# ============================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU
# ============================================================================

print("=" * 60)
print("BƯỚC 1: ĐỌC DỮ LIỆU")
print("=" * 60)

# Bắt đầu đo thời gian
start_time = time.time()

# Đọc file True.csv (chứa tin thật) vào DataFrame
# low_memory=False: Buộc pandas đọc toàn bộ file một lần để xác định kiểu dữ liệu chính xác
true_df = pd.read_csv('data/True.csv', low_memory=False)

# Đọc file Fake.csv (chứa tin giả) vào DataFrame
fake_df = pd.read_csv('data/Fake.csv', low_memory=False)

# Thêm cột 'label' vào DataFrame tin thật, gán giá trị 0 cho tất cả các dòng
# 0 = REAL (tin thật)
true_df['label'] = 0

# Thêm cột 'label' vào DataFrame tin giả, gán giá trị 1 cho tất cả các dòng
# 1 = FAKE (tin giả)
fake_df['label'] = 1

# pd.concat(): Ghép 2 DataFrame lại với nhau theo chiều dọc (nối hàng)      
# ignore_index=True: Đánh lại số thứ tự hàng từ 0
df = pd.concat([true_df, fake_df], ignore_index=True)

# .sample(frac=1): Xáo trộn ngẫu nhiên toàn bộ dữ liệu (frac=1 nghĩa là lấy 100%)
# random_state=42: Cố định kết quả xáo trộn để chạy lại cho kết quả giống nhau
# .reset_index(drop=True): Đánh lại số thứ tự hàng sau khi xáo trộn
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# In thống kê về dữ liệu
print(f"✅ Tổng số bài báo: {len(df):,}")                        # Tổng số bài
print(f"   - REAL (tin thật): {len(df[df['label']==0]):,}")     # Số bài REAL
print(f"   - FAKE (tin giả):  {len(df[df['label']==1]):,}")     # Số bài FAKE
print(f"⏱️  Thời gian: {time.time() - start_time:.2f} giây\n")  # In thời gian


# ============================================================================
# BƯỚC 2: LÀM SẠCH VĂN BẢN
# ============================================================================

print("=" * 60)
print("BƯỚC 2: LÀM SẠCH VĂN BẢN (có thể mất 2-5 phút)")
print("=" * 60)

# Bắt đầu đo thời gian
start_time = time.time()

# Kết hợp cột 'title' (tiêu đề) và 'text' (nội dung) thành 1 cột 'content'
# Dấu " " ở giữa để phân tách tiêu đề và nội dung
df['content'] = df['title'] + " " + df['text']

# .progress_apply(): Áp dụng hàm clean_text cho từng dòng, có thanh tiến độ
# Hàm clean_text sẽ: chuyển chữ thường, bỏ URL, bỏ ký tự đặc biệt, bỏ stopwords, stemming
df['clean_content'] = df['content'].progress_apply(clean_text)

# In thông báo hoàn thành
print(f"\n✅ Đã làm sạch xong {len(df):,} bài báo")
print(f"⏱️  Thời gian: {time.time() - start_time:.2f} giây\n")

# Lưu dữ liệu đã làm sạch vào file CSV mới
# Chỉ lưu 2 cột: 'clean_content' và 'label'
# index=False: Không lưu cột số thứ tự
df[['clean_content', 'label']].to_csv('data/processed_news.csv', index=False)
print("✅ Đã lưu file data/processed_news.csv\n")


# ============================================================================
# BƯỚC 3: TF-IDF & CHIA DỮ LIỆU
# ============================================================================

print("=" * 60)
print("BƯỚC 3: TF-IDF & CHIA DỮ LIỆU")
print("=" * 60)

# Bắt đầu đo thời gian
start_time = time.time()

# X = dữ liệu đầu vào (văn bản đã làm sạch)
X = df['clean_content']

# y = nhãn (0 = REAL, 1 = FAKE)
y = df['label']

# train_test_split(): Chia dữ liệu thành 2 tập
# - test_size=0.2: 20% dùng để kiểm tra, 80% dùng để huấn luyện
# - random_state=42: Cố định kết quả chia để chạy lại giống nhau
# - stratify=y: Đảm bảo tỉ lệ REAL/FAKE trong 2 tập giống nhau
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Tập huấn luyện: {len(X_train):,} bài báo")
print(f"✅ Tập kiểm tra:   {len(X_test):,} bài báo")

# TfidfVectorizer(): Công cụ biến văn bản thành vector số
# - max_features=5000: Chỉ giữ lại 5000 từ quan trọng nhất (giảm nhiễu, tăng tốc)
# - ngram_range=(1, 2): Xét cả từ đơn (1 từ) và cụm 2 từ (bigram)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# fit_transform(): Học từ vựng từ tập train VÀ biến đổi luôn
X_train_tfidf = vectorizer.fit_transform(X_train)

# transform(): CHỈ biến đổi tập test, KHÔNG học từ vựng mới
# (Quan trọng: tập test phải dùng cùn   g từ vựng với tập train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✅ Số đặc trưng (từ): {X_train_tfidf.shape[1]:,}")
print(f"⏱️  Thời gian: {time.time() - start_time:.2f} giây\n")


# ============================================================================
# BƯỚC 4: HUẤN LUYỆN MÔ HÌNH
# ============================================================================

print("=" * 60)
print("BƯỚC 4: HUẤN LUYỆN MÔ HÌNH LOGISTIC REGRESSION")
print("=" * 60)

# Bắt đầu đo thời gian
start_time = time.time()

# LogisticRegression(): Khởi tạo mô hình
# - max_iter=1000: Số lần lặp tối đa để tìm ra trọng số tối ưu
# - n_jobs=-1: Dùng tất cả CPU có sẵn để chạy nhanh hơn
model = LogisticRegression(max_iter=1000, n_jobs=-1)

# .fit(): Huấn luyện mô hình trên tập train
# Mô hình sẽ tự tìm ra trọng số (w) và hằng số (b) tối ưu
model.fit(X_train_tfidf, y_train)

print(f"✅ Đã huấn luyện xong mô hình")
print(f"⏱️  Thời gian: {time.time() - start_time:.2f} giây\n")


# ============================================================================
# BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH
# ============================================================================

print("=" * 60)
print("BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH")
print("=" * 60)

# .predict(): Dự đoán nhãn cho tập test
y_pred = model.predict(X_test_tfidf)

# accuracy_score(): Tính độ chính xác = (số dự đoán đúng) / (tổng số)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 ĐỘ CHÍNH XÁC: {accuracy * 100:.2f}%\n")

# classification_report(): In báo cáo chi tiết Precision/Recall/F1 cho từng lớp
# target_names=['REAL', 'FAKE']: Đặt tên cho 2 lớp (0=REAL, 1=FAKE)
report = classification_report(y_test, y_pred, target_names=['REAL', 'FAKE'])
print("📊 BÁO CÁO PHÂN LOẠI CHI TIẾT:")
print(report)

# Lưu báo cáo vào file .txt
with open('reports/classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Độ chính xác: {accuracy * 100:.2f}%\n\n")
    f.write(report)
print("✅ Đã lưu báo cáo vào reports/classification_report.txt\n")


# ============================================================================
# BƯỚC 6: VẼ CONFUSION MATRIX (MA TRẬN NHẦM LẪN)
# ============================================================================

print("=" * 60)
print("BƯỚC 6: VẼ CONFUSION MATRIX")
print("=" * 60)

# plt.figure(): Tạo khung hình với kích thước 8x6 inch
plt.figure(figsize=(8, 6))

# confusion_matrix(): Tính ma trận nhầm lẫn
# Kết quả là ma trận 2x2: [[TN, FP], [FN, TP]]
cm = confusion_matrix(y_test, y_pred)

# sns.heatmap(): Vẽ ma trận nhầm lẫn dưới dạng biểu đồ nhiệt
# - annot=True: Hiện số trên mỗi ô
# - fmt='d': Định dạng số nguyên
# - cmap='Blues': Dùng bảng màu xanh
# - xticklabels/yticklabels: Nhãn cho trục X và Y
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['REAL', 'FAKE'],
            yticklabels=['REAL', 'FAKE'])

# Đặt tiêu đề và nhãn cho biểu đồ
plt.title('Confusion Matrix - Fake News Detection', fontsize=14)
plt.xlabel('Dự đoán', fontsize=12)
plt.ylabel('Thực tế', fontsize=12)

# tight_layout(): Tự động điều chỉnh để không bị cắt chữ
plt.tight_layout()

# savefig(): Lưu biểu đồ thành file ảnh PNG với độ phân giải 150 dpi
plt.savefig('reports/confusion_matrix.png', dpi=150)
print("✅ Đã lưu biểu đồ vào reports/confusion_matrix.png\n")


# ============================================================================
# BƯỚC 7: LƯU MÔ HÌNH
# ============================================================================

print("=" * 60)
print("BƯỚC 7: LƯU MÔ HÌNH")
print("=" * 60)

# joblib.dump(): Lưu mô hình và vectorizer thành file .pkl
# File này có thể được tải lại sau này để dự đoán mà không cần huấn luyện lại
joblib.dump(model, 'models/best_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print("✅ Đã lưu models/best_model.pkl")
print("✅ Đã lưu models/tfidf_vectorizer.pkl\n")


# ============================================================================
# BƯỚC 8: TEST THỬ DỰ ĐOÁN VỚI BÀI BÁO MỚI
# ============================================================================

print("=" * 60)
print("BƯỚC 8: TEST THỬ DỰ ĐOÁN VỚI BÀI BÁO MỚI")
print("=" * 60)

# Định nghĩa hàm dự đoán một bài báo mới
def predict_news(text):
    """
    Hàm dự đoán một bài báo mới:
    - Đầu vào: text (văn bản thô)
    - Đầu ra: (label, confidence) - nhãn và độ tin cậy
    """
    # Bước 1: Làm sạch văn bản bằng hàm clean_text
    cleaned = clean_text(text)
    
    # Bước 2: Biến văn bản thành vector số bằng vectorizer đã học
    # vectorizer.transform([cleaned]): Biến 1 câu thành vector (cần [] để tạo list)
    vectorized = vectorizer.transform([cleaned])
    
    # Bước 3: Dự đoán nhãn (0 hoặc 1)
    prediction = model.predict(vectorized)[0]
    
    # Bước 4: Lấy xác suất của từng lớp [P(REAL), P(FAKE)]
    probability = model.predict_proba(vectorized)[0]
    
    # Bước 5: Xác định nhãn và độ tin cậy
    label = "FAKE (Tin giả)" if prediction == 1 else "REAL (Tin thật)"
    confidence = probability[prediction] * 100
    
    return label, confidence


# Danh sách 3 bài báo test
test_articles = [
    "Breaking: Scientists discover cure for cancer overnight!",
    "The government announced new tax policy today.",
    "SHOCKING! You won't believe what this celebrity did! Share now!"
]

# Duyệt qua từng bài báo và in kết quả dự đoán
for i, article in enumerate(test_articles, 1):
    label, conf = predict_news(article)
    print(f"\n📰 Bài báo {i}: {article[:70]}...")
    print(f"   → Dự đoán: {label}")
    print(f"   → Độ tin cậy: {conf:.2f}%")


# ============================================================================
# HOÀN THÀNH
# ============================================================================

print("\n" + "=" * 60)
print("🎉 HOÀN THÀNH TOÀN BỘ QUY TRÌNH!")
print("=" * 60)
print("\nCác file đã tạo:")
print("  📁 data/processed_news.csv")
print("  📁 models/best_model.pkl")
print("  📁 models/tfidf_vectorizer.pkl")
print("  📁 reports/classification_report.txt")
print("  📁 reports/confusion_matrix.png")