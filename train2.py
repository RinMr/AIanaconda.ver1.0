import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split

# ディレクトリ設定
data_dir = ""  #　画像フォルダパス
save_dir = ""  #　保存パス
comparison_dir = os.path.join(save_dir, 'comparisons')
history_dir = os.path.join(save_dir, 'histories')
loss_acc_dir = os.path.join(save_dir, 'loss_and_acc')
predictions_dir = os.path.join(save_dir, 'predictions')
models_dir = os.path.join(save_dir, 'models')

os.makedirs(comparison_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)
os.makedirs(loss_acc_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# 画像のパラメータ
image_size = (96, 96)
batch_size = 32
epochs = 150

# データの検証と可視化のためのラベル名
class_names = ['cat', 'dog', 'lion', 'monkey', 'wolf']
class_indices = {name: idx for idx, name in enumerate(class_names)}

# 画像とラベルを読み込む
images = []
labels = []

print("画像を読み込んでいます...")
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    class_idx = class_indices[class_name]
    images_in_class = 0
    for filename in os.listdir(class_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"画像を読み込めませんでした: {img_path}")
                continue
            img_resized = cv2.resize(img, image_size)
            images.append(img_resized)
            labels.append(class_idx)
            images_in_class += 1
    print(f'クラス "{class_name}" の画像を {images_in_class} 枚読み込みました。')

# NumPy配列に変換
images = np.array(images)
labels = np.array(labels)

print(f"合計で {len(images)} 枚の画像を読み込みました。")
cat_indices = np.where(labels == class_indices['cat'])[0]
cat_samples = len(cat_indices) * 2  # cat 2倍

cat_images, cat_labels = resample(images[cat_indices], labels[cat_indices], 
                                n_samples=cat_samples, random_state=42)

dog_indices = np.where(labels == class_indices['dog'])[0]
dog_samples = len(dog_indices) * 1  # dog 1倍

dog_images, dog_labels = resample(images[dog_indices], labels[dog_indices], 
                                n_samples=dog_samples, random_state=42)

# 他のクラスと結合
new_images = np.concatenate([images, cat_images, dog_images], axis=0)
new_labels = np.concatenate([labels, np.full(cat_samples, class_indices['cat']), 
                            np.full(dog_samples, class_indices['dog'])], axis=0)

print(f"データ拡張後の総画像数: {len(new_images)}, ラベル数: {len(new_labels)}")

# トレーニングとテストの分割
x_train, x_test, y_train, y_test = train_test_split(
    new_images, new_labels, test_size=0.2, stratify=new_labels, random_state=42
)

print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

# 正規化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hotエンコーディング
num_classes = len(class_names)
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

# クラスごとのサンプル数を表示
classes, counts = np.unique(y_train, return_counts=True)
print("クラスごとのトレーニングサンプル数:")
for cls, count in zip(classes, counts):
    print(f'{class_names[cls]} (Class {cls}): {count} samples')

# データ拡張
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

train_generator = train_datagen.flow(x_train, y_train_onehot, batch_size=batch_size)

# データの可視化
def plot_sample_images(x, y, class_names, num_samples=5):
    for cls in np.unique(y):
        idxs = np.where(y == cls)[0]
        idxs = np.random.choice(idxs, num_samples, replace=False)
        imgs = x[idxs]
        fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
        for i, img in enumerate(imgs):
            axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[i].axis('off')
        plt.suptitle(f'Class {cls}: {class_names[cls]}')
        plt.show()

# トレーニングデータのサンプル画像を表示
plot_sample_images(x_train, y_train, class_names)

# カスタムCNNモデルの構築（Conv2Dを使用）
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

# モデルのサマリーを表示
model.summary()

# モデルのコンパイル
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# モデル学習
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=(x_test, y_test_onehot)
)

# モデル保存
model.save(os.path.join(models_dir, 'model.h5'))

# 評価
val_loss, val_acc = model.evaluate(x_test, y_test_onehot)
with open(os.path.join(loss_acc_dir, 'loss_and_acc.txt'), 'w') as f:
    f.write(f'Validation Loss: {val_loss}\nValidation Accuracy: {val_acc}\n')

# 予測
predictions = model.predict(x_test)
y_true = np.argmax(y_test_onehot, axis=1)
y_pred = np.argmax(predictions, axis=1)

# 保存
np.save(os.path.join(predictions_dir, 'predictions.npy'), predictions)
np.save(os.path.join(predictions_dir, 'y_true.npy'), y_true)
np.save(os.path.join(predictions_dir, 'y_pred.npy'), y_pred)

# 混同行列の作成と表示
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
            horizontalalignment="center",
            color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(os.path.join(comparison_dir, 'confusion_matrix.png'))
plt.show()

# 分類レポートの出力
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(comparison_dir, 'classification_report.csv'))

print("分類モデルのトレーニングと評価が完了しました。")