import pandas as pd
import numpy as np
import json
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score


# ==========================================
# 1. 配置与清洗逻辑 
# ==========================================
BEST_TEXT_COL = 'In your own words, what kinds of tasks would you use this model for?'
RATING_COLS = [
    'How likely are you to use this model for academic tasks?',
    'Based on your experience, how often has this model given you a response that felt suboptimal?',
    'How often do you expect this model to provide responses with references or supporting evidence?',
    "How often do you verify this model's responses?"
]
MULTI_COLS = [
    'Which types of tasks do you feel this model handles best? (Select all that apply.)',
    'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'
]
MULTI_MAP = {
    "math computations": "Math computations",
    "writing or debugging code": "Writing or debugging code",
    "data processing or analysis": "Data processing or analysis",
    "explaining complex concepts simply": "Explaining complex concepts simply",
    "writing or editing essays/reports": "Writing or editing essays/reports",
    "brainstorming or generating creative ideas": "Brainstorming or generating creative ideas",
    "drafting professional text": "Drafting professional text (e.g., résumés, emails)",
    "résumés": "Drafting professional text (e.g., résumés, emails)",
    "emails": "Drafting professional text (e.g., résumés, emails)",
    "converting content between formats": "Converting content between formats (e.g., LaTeX)",
    "latex": "Converting content between formats (e.g., LaTeX)",
}
CANONICAL_MULTI_TYPES = sorted(list(set(MULTI_MAP.values())))
TARGET_COLUMN = 'label'

STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'model', 'task', 'use', 'would', 'response', 'ai', 'give', 'asks', 'asked'}

def clean_text_regex(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    cleaned = []
    for w in words:
        if w not in STOPWORDS and len(w) > 2:
            if w.endswith('ing'): w = w[:-3]
            elif w.endswith('ed'): w = w[:-2]
            elif w.endswith('s') and not w.endswith('ss'): w = w[:-1]
            if len(w) > 2: cleaned.append(w)
    return " ".join(cleaned)

def extract_rating(response):
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else 3 

def process_multiselect(series):
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
            continue
        response_lower = str(response).lower()
        found_tasks = set()
        for key, canonical_val in MULTI_MAP.items():
            if key in response_lower:
                found_tasks.add(canonical_val)
        processed.append(list(found_tasks))
    return processed

# ==========================================
# 2. 主流程
# ==========================================
print("1. 加载与清洗数据...")
try:
    df = pd.read_csv('training_data_clean.csv')
except FileNotFoundError:
    print("错误：未找到 csv 文件")
    exit()

df = df.dropna(subset=[TARGET_COLUMN])
df = df[df[TARGET_COLUMN].astype(str).str.strip() != '']
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COLUMN])
df['cleaned_text'] = df[BEST_TEXT_COL].apply(clean_text_regex)

print("2. 构建特征 (TF-IDF + OneHot + Multi)...")

# A. 文本: TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 1), 
    min_df=2, 
    sublinear_tf=True
)
X_text_sparse = vectorizer.fit_transform(df['cleaned_text'])
#转为密集矩阵 (Numpy Array)
X_text = X_text_sparse.toarray() 

# B. 评分: One-Hot
ratings_raw = []
for col in RATING_COLS:
    ratings_raw.append(df[col].apply(extract_rating).values.reshape(-1, 1))
# 强制分类 1-5
categories = [np.array([1, 2, 3, 4, 5]) for _ in range(len(RATING_COLS))]
#  sparse_output=False 直接输出密集矩阵
rating_encoder = OneHotEncoder(categories=categories, sparse_output=False, handle_unknown='ignore')
X_rating = rating_encoder.fit_transform(np.hstack(ratings_raw))

# C. 多选: Multi-Hot
mlb = MultiLabelBinarizer(classes=CANONICAL_MULTI_TYPES)
X_multi_list = []
for col in MULTI_COLS:
    processed_col = process_multiselect(df[col])
    X_multi_list.append(mlb.fit_transform(processed_col))

X_multi = np.hstack(X_multi_list)

X_final = np.hstack([X_text, X_rating, X_multi])
print(f"   特征总维度 (未筛选): {X_final.shape}")

# 3. 特征选择
print("3. 特征选择 (SelectKBest k=420)...")

selector = SelectKBest(chi2, k=420)
X_selected = selector.fit_transform(X_final, y)
print(f"   特征总维度 (筛选后): {X_selected.shape}")

# 4. 交叉验证
print("4. 验证模型性能 (Cross Validation)...")
model = MultinomialNB(alpha=0.5, fit_prior=True)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = cross_val_score(model, X_selected, y, cv=kfold, scoring=make_scorer(f1_score, average='macro', zero_division=0))
acc_scores = cross_val_score(model, X_selected, y, cv=kfold, scoring='accuracy')

print(f"\n --- 验证结果 (Validation Results) ---")
print(f"F1 Score (Macro): {f1_scores.mean():.4f} (+/- {f1_scores.std()*2:.4f})")
print(f"Accuracy:         {acc_scores.mean():.4f}")

# 5. 训练并保存
print("\n5. 在全量数据上训练并保存参数...")
model.fit(X_selected, y)

print("   - 保存贝叶斯模型参数 (概率表)...")
np.save('nb_class_log_prior.npy', model.class_log_prior_)
np.save('nb_feature_log_prob.npy', model.feature_log_prob_)
np.save('nb_classes.npy', le.classes_)

print("   - 保存特征工程参数...")
# 1. TF-IDF 词汇表
vocabulary_fixed = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
with open('nb_tfidf_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocabulary_fixed, f, indent=4)

# 2. IDF 向量
np.save('nb_tfidf_idf.npy', vectorizer.idf_)

# 3. 特征掩码
support_mask = selector.get_support()
np.save('nb_feature_mask.npy', support_mask)
