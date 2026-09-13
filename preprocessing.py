import re  # Nhập thư viện Regular Expression (biểu thức chính quy) để tìm kiếm và xử lý chuỗi ký tự
import nltk  # Nhập thư viện NLTK (Natural Language Toolkit) chuyên dùng xử lý ngôn ngữ tự nhiên
from nltk.corpus import stopwords  # Nhập kho dữ liệu chứa các từ dừng (stopwords) từ NLTK
from nltk.stem import PorterStemmer  # Nhập thuật toán Porter Stemmer để đưa từ về dạng gốc

# Tải dữ liệu NLTK (chỉ chạy lần đầu)
nltk.download('stopwords')  # Tải xuống danh sách các từ dừng từ máy chủ NLTK về máy tính
nltk.download('punkt')  # Tải xuống bộ dữ liệu phân tách từ/câu (tokenizer) của NLTK

stop_words = set(stopwords.words('english'))  # Lấy danh sách từ dừng tiếng Anh và chuyển thành dạng set để tra cứu nhanh hơn
stemmer = PorterStemmer()  # Khởi tạo đối tượng thực hiện thuật toán rút gọn từ gốc

def clean_text(text):  # Định nghĩa hàm làm sạch văn bản có tên là clean_text nhận vào một chuỗi (text)
    """Hàm làm sạch văn bản"""  # Phần chú thích mô tả chức năng của hàm
    text = text.lower()                                    # Chuyển toàn bộ ký tự trong văn bản thành chữ thường để chuẩn hóa
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)   # Tìm và xóa các đường dẫn URL (bắt đầu bằng http, https hoặc www)
    text = re.sub(r'\S+@\S+', '', text)                   # Tìm và xóa các địa chỉ email có chứa ký tự @
    text = re.sub(r'[^a-zA-Z\s]', '', text)               # Xóa tất cả các ký tự không phải chữ cái tiếng Anh (a-z, A-Z) và khoảng trắng (\s)
    words = text.split()  # Tách chuỗi văn bản thành một danh sách các từ riêng lẻ dựa vào khoảng trắng
    words = [stemmer.stem(w) for w in words if w not in stop_words]  # Duyệt qua từng từ: nếu từ không nằm trong danh sách từ dừng, tiến hành rút gọn gốc từ đó
    return " ".join(words)  # Ghép lại danh sách các từ đã lọc thành một chuỗi hoàn chỉnh bằng khoảng trắng và trả về kết quả