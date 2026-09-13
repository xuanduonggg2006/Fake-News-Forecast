import joblib
from preprocessing import clean_text

# Tải mô hình và vectorizer đã lưu
model = joblib.load('models/best_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def predict(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    label = "FAKE (Tin giả)" if prediction == 1 else "REAL (Tin thật)"
    confidence = probability[prediction] * 100
    return label, confidence

# Test
if __name__ == "__main__":
    print("=" * 60)
    print("DỰ ĐOÁN TIN GIẢ - NHẬP BÀI BÁO ĐỂ KIỂM TRA")
    print("=" * 60)
    
    while True:
        text = input("\nNhập bài báo (hoặc 'quit' để thoát): ")
        if text.lower() == 'quit':
            break
        
        label, conf = predict(text)
        print(f"→ Kết quả: {label}")
        print(f"→ Độ tin cậy: {conf:.2f}%")
        