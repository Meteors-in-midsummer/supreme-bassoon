import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from snownlp import SnowNLP
from collections import Counter
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.unicode.east_asian_width',True)
# 定义停用词列表
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return set(stopwords)
# 1. 数据导入
data = pd.read_csv(r"E:\pythonProject\python练习\data\merged_file.csv")
# 2. 数据预处理
def clean_text(text, stopwords):
    # 去除HTML标签
    text = text.replace('<br />', ' ')
    # 去除特殊字符和标点
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    # 切词
    words = jieba.lcut(text)
    # 过滤停用词和空格
    filtered_words = [word for word in words if word.strip() and word not in stopwords]
    return ' '.join(filtered_words)

# 加载停用词列表
stopwords = load_stopwords(r'E:\pythonProject\python练习\data\cn_stopwords.txt')  
# 应用清洗函数
data['cleaned_comment'] = data['comments'].apply(lambda x: clean_text(x, stopwords))

# 过滤掉空评论
data = data[data['cleaned_comment'].str.len() > 0]

# 3. 情感分析
def snownlp_sentiment(text):
    s = SnowNLP(text)
    return s.sentiments

# 计算情感得分
data['sentiment_score'] = data['cleaned_comment'].apply(snownlp_sentiment)

# 4. 情感分类
def classify_sentiment(score):
    if score > 0.8:
        return 1  # 积极
    elif score < 0.6:
        return -1  # 消极
    else:
        return 0  # 中等

data['label'] = data['sentiment_score'].apply(classify_sentiment)

# 5. 情感得分直方图
plt.figure(figsize=(10, 6))
plt.hist(data['sentiment_score'], bins=30, color='skyblue', edgecolor='black')
plt.title('情感得分分布')
plt.xlabel('情感得分')
plt.ylabel('数量')
#plt.show()
all_comments = ' '.join(data['cleaned_comment'])
# 使用 jieba 进行分词
words = jieba.lcut(all_comments)
# 统计词频
word_counts = Counter(words)

# 将词频统计结果转换为 DataFrame
word_freq_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])
# 按频率降序排序
word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)
# 7. 提取关键词
def extract_keywords(comment, stopwords):
    words = jieba.lcut(comment)
    # 过滤停用词和空格
    filtered_words = [word for word in words if word.strip() and word not in stopwords]
    return filtered_words

keywords = []
for comment in data['cleaned_comment']:
    keywords.extend(extract_keywords(comment, stopwords))

# 统计关键词频率
keyword_counts = Counter(keywords)
print("\n关键词频率前20：")
print(keyword_counts.most_common(20))

# 8. 分析消极评论和积极评论
positive_comments = data[data['label'] == 1]['cleaned_comment']
negative_comments = data[data['label'] == -1]['cleaned_comment']
neutral_comments = data[data['label'] == 0]['cleaned_comment']

# 10. 统计三种评论占比
total_comments = len(data)
positive_ratio = len(positive_comments) / total_comments*100
negative_ratio = len(negative_comments) / total_comments*100
neutral_ratio = len(neutral_comments) / total_comments*100

# 11. 对消极评论展开分析
#print("\n消极评论关键词频率前10：")
negative_keywords = []
for comment in negative_comments:
    negative_keywords.extend(extract_keywords(comment, stopwords))
negative_keyword_counts = Counter(negative_keywords)
print(negative_keyword_counts.most_common(30))
#伪满皇宫景区评论情感分析占比
labels = ['积极', '中等', '消极']
values = [positive_ratio, neutral_ratio, negative_ratio]
plt.bar(labels, values, color=['#4CAF50', '#FFEB3B', '#F44336'], alpha=0.7)
plt.xlabel('情感分类')
plt.ylabel('占比 (%)')
plt.title('长春电影制片厂景区评论情感分析占比')
plt.ylim(0, 100)  # 设置y轴范围
for i, v in enumerate(values):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom')
#plt.show()
