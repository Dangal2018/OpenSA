import pickle
import numpy as np
#加载模型和保存模型


# 加载模型
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    return model

# 保存模型
def save_model(model):
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

# 预测

def model_predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

if __name__ == '__main__':
    datapath = "0117dami2.csv"
    data1 = np.loadtxt(open(datapath, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    data = data1[:, :-1]
    label = data1[:, -1]
    model = load_model()
    y_pred = model_predict(model, data)

    print("predict results:")
    print("预测值 真实值")
    for i in range(0, len(y_pred)):
        print(y_pred[i], label[i])
