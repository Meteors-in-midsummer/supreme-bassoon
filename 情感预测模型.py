import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
data = pd.read_csv(r'E:\\pythonProject\\python练习\\data\\result_伪满皇宫.csv')
comments = data['comments'].values  # 评论内容
# 创建一个空列表来存储情感标签
sentiment_labels = []

# 遍历每条评论并手动标注情感
for comment in comments:
    # 示例：根据您的判断为每条评论标注情感
    # 您可以使用一些简单的规则，例如包含“好”、“不错”等词汇为正面，包含“差”、“失望”等词汇为负面
    if '好' in comment or '不错' in comment or '喜欢' in comment:
        sentiment_labels.append('positive')
    elif '差' in comment or '失望' in comment or '不好' in comment:
        sentiment_labels.append('negative')
    else:
        sentiment_labels.append('neutral')  # 默认为中性

# 将情感标签添加到数据中
data['label'] = sentiment_labels
# 保存带有情感标签的数据
data.to_csv('weimanhungcomment_with_labels.csv', index=False)
# 重新加载带有情感标签的数据
data_with_labels = pd.read_csv('weimanhungcomment_with_labels.csv')
comments = data_with_labels['comments'].values  # 评论内容
labels = data_with_labels['label'].values       # 情感标签
# 将标签编码为数值
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
# 将数值标签转换为 one-hot 编码
num_classes = len(label_encoder.classes_)
one_hot_labels = to_categorical(encoded_labels, num_classes=num_classes)

# 文本分词和序列化
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)

# 填充序列以统一长度
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, one_hot_labels, test_size=0.2, random_state=42)
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),  # 嵌入层
    LSTM(64, return_sequences=True),                              # LSTM层
    LSTM(32),                                                      # 另一个LSTM层
    Dense(64, activation='relu'),                                  # 全连接层
    Dropout(0.5),                                                  # Dropout层防止过拟合
    Dense(num_classes, activation='softmax')                       # 输出层
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 打印模型结构
model.summary()
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)
# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# 保存训练好的模型
model.save('weimanhung_sentiment_model.keras')



