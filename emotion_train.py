import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

# ディレクトリ設定
processed_data_dir = ""  #　画像フォルダパス
save_dir = ""  #　保存パス
comparison_dir = os.path.join(save_dir, 'comparisons')
history_dir = os.path.join(save_dir, 'histories')
loss_acc_dir = os.path.join(save_dir, 'loss_and_acc')
predictions_dir = os.path.join(save_dir, 'predictions')
models_dir = os.path.join(save_dir, 'models')

# ディレクトリの作成
os.makedirs(comparison_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)
os.makedirs(loss_acc_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# 画像のパラメータ
image_size = (224, 224)  # VGG16は224x224の入力サイズ
batch_size = 32
epochs = 30  #  epochs 30

# データジェネレータ
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# トレーニングデータ
train_gen = data_gen.flow_from_directory(
    directory=processed_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# 検証データ
val_gen = data_gen.flow_from_directory(
    directory=processed_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# クラスインデックスの取得と保存
class_indices = train_gen.class_indices
with open(os.path.join(models_dir, 'emotion_class_indices.pkl'), 'wb') as f:
    pickle.dump(class_indices, f)

# 事前学習済みモデルの読み込み（トップ層を除外）
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# ベースモデルの層を凍結
for layer in base_model.layers:
    layer.trainable = False

# 新しいトップ層の追加
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(len(class_indices), activation='softmax')(x)

# モデルの作成
model = Model(inputs=base_model.input, outputs=outputs)

# モデルのコンパイル
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# モデルの学習
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen
)

# 微調整（ファインチューニング）
# ベースモデルの最後の数層を解凍（学習可能に設定）
for layer in base_model.layers[-4:]:
    layer.trainable = True

# 学習率を下げて再コンパイル
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# 再度学習
history_fine = model.fit(
    train_gen,
    epochs=10,  #  epochs 10
    validation_data=val_gen
)

# モデルの保存
model.save(os.path.join(models_dir, 'emotion_model.h5'))

# モデルの評価
val_loss, val_acc = model.evaluate(val_gen)
with open(os.path.join(loss_acc_dir, 'emotion_loss_and_acc.txt'), 'w') as f:
    f.write(f'Validation Loss: {val_loss}\nValidation Accuracy: {val_acc}\n')

# 予測の生成
predictions = model.predict(val_gen)
y_true = val_gen.classes
y_pred = np.argmax(predictions, axis=1)

# 予測結果の保存
np.save(os.path.join(predictions_dir, 'emotion_predictions.npy'), predictions)
np.save(os.path.join(predictions_dir, 'emotion_y_true.npy'), y_true)
np.save(os.path.join(predictions_dir, 'emotion_y_pred.npy'), y_pred)

# 分類レポートの保存
report = classification_report(y_true, y_pred, target_names=list(class_indices.keys()))
with open(os.path.join(comparison_dir, 'emotion_classification_report.txt'), 'w') as f:
    f.write(report)

# 混同行列の保存
conf_matrix = confusion_matrix(y_true, y_pred)
np.save(os.path.join(comparison_dir, 'emotion_confusion_matrix.npy'), conf_matrix)

# 学習履歴のプロットと保存
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.savefig(os.path.join(history_dir, 'emotion_training_history.png'))
plt.close()

print("分類モデルのトレーニングが完了しました。結果が保存されました。")
