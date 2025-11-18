from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator      # mô hình ké google translate
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
import os

app = Flask(__name__)
translator = Translator()

# Đường dẫn file CSV
CSV_FILE = 'data/documents.csv'

# Biến global để lưu dữ liệu
documents_df = None
tfidf_vectorizer = None
tfidf_matrix = None

# SentenceTransformer embedding model
# switch words to vectors
def semantic_search(query, documents):
    query_embedding = model.encode(query)
    doc_embeddings = model.encode(documents)
    similarities = cosine_similarity([query_embedding], doc_embeddings)
# so sanh ket qua vector voi tu ngu input duoc nhap

    return similarities

def load_csv_data():
    """Tải và xử lý dữ liệu từ file CSV"""
    global documents_df, tfidf_vectorizer, tfidf_matrix
    
    try:
        # Đọc file CSV
        documents_df = pd.read_csv(CSV_FILE, encoding='utf-8')
        
        # Kiểm tra cột csv
        if 'content' not in documents_df.columns:
            print("Cảnh báo: CSV cần có cột 'content'")
            return False
        
        # Tạo ma trận TF-IDF vectorizer cho tìm kiếm
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents_df['content'].fillna(''))
        
        print(f"✓ Đã tải {len(documents_df)} tài liệu từ CSV")
        return True
    except FileNotFoundError:
        print(f"✗ Không tìm thấy file: {CSV_FILE}")
        return False
    except Exception as e:
        print(f"✗ Lỗi khi tải CSV: {str(e)}")
        return False

def create_sample_csv():
    """Tạo file CSV mẫu nếu chưa có"""
    os.makedirs('data', exist_ok=True)
    
    if not os.path.exists(CSV_FILE):
        sample_data = {
            'id': [1, 2, 3, 4, 5],
            'title': [
                'Python Programming',
                'Machine Learning Basics',
                'Natural Language Processing',
                'Deep Learning with PyTorch',
                'Data Science Fundamentals'
            ],
            'content': [
                'Python is a high-level programming language widely used in data science and machine learning.',
                'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
                'NLP deals with the interaction between computers and human language, enabling text analysis.',
                'Deep learning uses neural networks with multiple layers to solve complex problems.',
                'Data science combines statistics, programming, and domain knowledge to extract insights from data.'
            ],
            'category': ['Programming', 'AI', 'NLP', 'Deep Learning', 'Data Science']
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(CSV_FILE, index=False, encoding='utf-8')
        print(f"✓ Đã tạo file CSV mẫu: {CSV_FILE}")

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    """API dịch văn bản"""
    try:
        data = request.json
        text = data.get('text', '')
        target_lang = data.get('target_lang', 'vi')
        
        if not text:
            return jsonify({'error': 'Vui lòng nhập văn bản'}), 400
        
        # Dịch văn bản
        translation = translator.translate(text, dest=target_lang)
        
        return jsonify({
            'success': True,
            'original': text,
            'translated': translation.text,
            'source_lang': translation.src,
            'target_lang': target_lang
        })
    
    except Exception as e:
        return jsonify({'error': f'Lỗi dịch: {str(e)}'}), 500

@app.route('/search', methods=['POST'])
def search_documents():
    """API tìm kiếm tài liệu trong CSV"""
    try:
        data = request.json
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': 'Vui lòng nhập truy vấn tìm kiếm'}), 400
        
        if documents_df is None or tfidf_matrix is None:
            return jsonify({'error': 'Chưa tải dữ liệu CSV'}), 500
        
        # Chuyển query thành vector TF-IDF
        query_vector = tfidf_vectorizer.transform([query])
        
        # Tính độ tương đồng cosine similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Lấy top K kết quả
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Chỉ lấy kết quả có độ tương đồng > 0
                result = {
                    'id': int(documents_df.iloc[idx]['id']),
                    'title': documents_df.iloc[idx]['title'],
                    'content': documents_df.iloc[idx]['content'],
                    'category': documents_df.iloc[idx].get('category', 'N/A'),
                    'similarity': float(similarities[idx])
                }
                results.append(result)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'total_found': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': f'Lỗi tìm kiếm: {str(e)}'}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """API upload file CSV mới"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file được tải lên'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Chỉ chấp nhận file CSV'}), 400
        
        # Lưu file
        file.save(CSV_FILE)
        
        # Tải lại dữ liệu
        success = load_csv_data()
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Đã tải {len(documents_df)} tài liệu',
                'documents_count': len(documents_df)
            })
        else:
            return jsonify({'error': 'Lỗi khi xử lý file CSV'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Lỗi upload: {str(e)}'}), 500

@app.route('/stats')
def get_stats():
    """API lấy thống kê"""
    if documents_df is None:
        return jsonify({'loaded': False})
    
    return jsonify({
        'loaded': True,
        'total_documents': len(documents_df),
        'columns': list(documents_df.columns)
    })

if __name__ == '__main__':
    # Tạo file CSV mẫu nếu chưa có
    create_sample_csv()
    
    # Tải dữ liệu CSV
    load_csv_data()
    
    # Chạy app
    print("\n🚀 Server đang chạy tại: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)