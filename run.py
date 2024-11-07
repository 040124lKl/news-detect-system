import os
import sys
from other_code_files.Iterable_Sequence import PreVector, ImageSequence, CombinedSequence
import tensorflow as tf
import requests
import numpy as np
import pandas as pd
# 以上为依赖包引入部分, 请根据实际情况引入
# 引入的包需要安装的, 请在requirements.txt里列明, 最好请列明版本

# 以下为逻辑函数, main函数的入参和最终的结果输出不可修改
def main(to_pred_dir, result_save_path):
    run_py = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py)

    to_pred_dir = os.path.abspath(to_pred_dir)
    testa_csv_path = os.path.join(to_pred_dir, "testa_x", "testa_x.csv")
    testa_html_dir = os.path.join(to_pred_dir, "testa_x", "html")
    testa_image_dir = os.path.join(to_pred_dir, "testa_x", "image")

    #=-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==
    # 以下区域为预测逻辑代码, 下面的仅为示例
    # 请选手根据实际模型预测情况修改
    vector_size = 300
    batch_size = 16
    to_pred = pd.read_csv(testa_csv_path)
    id_1 = to_pred['id']
    vector_pre = PreVector(testa_html_dir, testa_csv_path, batch_size, vector_size)
    image_sequence = ImageSequence(testa_image_dir, batch_size)
    combinedsequence = CombinedSequence(vector_pre, image_sequence, predict=1)
    model = tf.keras.models.load_model('model')
    predictions = model.predict(combinedsequence, verbose=1)
    # 设置阈值
    threshold = 0.2
    # 将预测结果转换为标签
    predicted_labels = (predictions > threshold).astype(int)
    test = pd.DataFrame([id_1, predicted_labels], columns=['id', 'label'])



    #=-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==
   
    # 结果输出到result_save_path
    test.to_csv(result_save_path, index=None)

if __name__ == "__main__":
    # 以下代码请勿修改, 若因此造成的提交失败由选手自负
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv
    main(to_pred_dir, result_save_path)
