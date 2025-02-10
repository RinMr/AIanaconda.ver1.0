import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# ディレクトリ設定
base_dir = ""  #  h5ファイルパス
predictions_dir = os.path.join(base_dir, "predictions")
models_dir = os.path.join(base_dir, "models")
loss_acc_dir = os.path.join(base_dir, "loss_and_acc")
comparison_dir = os.path.join(base_dir, "comparisons")

# ファイル読み込み
y_true = np.load(os.path.join(predictions_dir, "emotion_y_true.npy"))
y_pred = np.load(os.path.join(predictions_dir, "emotion_y_pred.npy"))

with open(os.path.join(models_dir, "emotion_class_indices.pkl"), "rb") as f:
    class_indices = pickle.load(f)
    class_names = {v: k for k, v in class_indices.items()}
    class_names = [class_names[i] for i in range(len(class_names))]

# 分類レポートの作成
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(comparison_dir, "emotion_classification_report.txt"), sep="\t")

# 混同行列の作成と保存
conf_matrix = confusion_matrix(y_true, y_pred)
np.save(os.path.join(comparison_dir, "emotion_confusion_matrix.npy"), conf_matrix)

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
plt.savefig(os.path.join(comparison_dir, "emotion_confusion_matrix.png"))
plt.show()

print("分類レポートの作成が完了しました。")
