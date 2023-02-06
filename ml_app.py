import joblib # 모델 내보내기
import os

# 모델 불러오기
model_file = './models/logis_iris_model_230126.pkl'
loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))

