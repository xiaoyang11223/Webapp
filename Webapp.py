import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 页面内容设置
# 页面名称
st.set_page_config(page_title="MINS", layout="wide")
# 标题
st.title('The machine-learning based model to predict MINS')
# 文本
st.write('This is a web app to predict the prob of MINS based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction.')

st.markdown('## Input Data:')
# 隐藏底部水印
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            <![]()yle>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)

def option_name(x):
    if x == 0:
        return "no"
    if x == 1:
        return "yes"
@st.cache
def predict_quality(model, df):
    y_pred = model.predict(df, prediction_type='Probability')
    return y_pred[:, 1]

# 导入模型
model = joblib.load('D:\\Python\\老年心衰\\30d特因再入院\\模型\\catb.pkl')
st.sidebar.title("Features")

# 设置各项特征的输入范围和选项
Smoke = st.sidebar.selectbox(label='Smoke', options=[0, 1], format_func=lambda x: option_name(x), index=0)
Drink = st.sidebar.selectbox(label='Drink', options=[0, 1], format_func=lambda x: option_name(x), index=0)
SBP = st.sidebar.number_input(label='SBP', min_value=50.0,
                                  max_value=246.0,
                                  value=50.0,
                                  step=1.0)
DBP = st.sidebar.number_input(label='DBP', min_value=24.0,
                                  max_value=180.0,
                                  value=50.0,
                                  step=1.0)

Pulse = st.sidebar.number_input(label='Pulse', min_value=16.0,
                                  max_value=200.0,
                                  value=16.0,
                                  step=1.0)
Temperature = st.sidebar.number_input(label='Temperature', min_value=35.0,
                                  max_value=40.5,
                                  value=35.0,
                                  step=0.1)

RR = st.sidebar.number_input(label='RR', min_value=10.0,
                                  max_value=150.0,
                                  value=35.0,
                                  step=1.0)
Insurance = st.sidebar.selectbox(label='Insurance', options=[1,2,3,4], index=1)
Gender = st.sidebar.selectbox(label='Gender', options=[0, 1], format_func=lambda x: option_name(x), index=0)
Age = st.sidebar.number_input(label='Age', min_value=65.0,
                                  max_value=100.0,
                                  value=70.0,
                                  step=1.0)

LOS = st.sidebar.number_input(label='LOS', min_value=1.0,
                                  max_value=89.0,
                                  value=2.0,
                                  step=1.0)

Hypertension = st.sidebar.selectbox(label='Hypertension', options=[0, 1], format_func=lambda x: option_name(x), index=0)
CAD = st.sidebar.selectbox(label='CAD', options=[0, 1], format_func=lambda x: option_name(x), index=0)
NYHA3 = st.sidebar.selectbox(label='NYHA3', options=[0, 1], format_func=lambda x: option_name(x), index=0)##主要针对0,1二分类变量
NYHA2 = st.sidebar.selectbox(label='NYHA2', options=[0, 1], format_func=lambda x: option_name(x), index=0)
NYHA4 = st.sidebar.selectbox(label='NYHA4', options=[0, 1], format_func=lambda x: option_name(x), index=0)
Stroke = st.sidebar.selectbox(label='Stroke', options=[0, 1], format_func=lambda x: option_name(x), index=0)
Respiratory_failure = st.sidebar.selectbox(label='Respiratory.failure', options=[0, 1], format_func=lambda x: option_name(x), index=0)
HHD = st.sidebar.selectbox(label='HHD', options=[0, 1], format_func=lambda x: option_name(x), index=0)
Ischemic_cardiomyopathy = st.sidebar.selectbox(label='Ischemic.cardiomyopathy', options=[0, 1], format_func=lambda x: option_name(x), index=0)
AF = st.sidebar.selectbox(label='AF', options=[0, 1], format_func=lambda x: option_name(x), index=0)
Carotid_arteriosclerosis = st.sidebar.selectbox(label='Carotid.arteriosclerosis', options=[0, 1], format_func=lambda x: option_name(x), index=0)
Osteoporosis = st.sidebar.selectbox(label='Osteoporosis', options=[0, 1], format_func=lambda x: option_name(x), index=0)
Hypoproteinemia = st.sidebar.selectbox(label='Hypoproteinemia', options=[0, 1], format_func=lambda x: option_name(x), index=0)
Hyperuricemia = st.sidebar.selectbox(label='Hyperuricemia', options=[0, 1], format_func=lambda x: option_name(x), index=0)
CRF = st.sidebar.selectbox(label='CRF', options=[0, 1], format_func=lambda x: option_name(x), index=0)
NTproBNP = st.sidebar.number_input(label='NT.proBNP', min_value=5.0,
                                  max_value=67000.0,
                                  value=70.0,
                                  step=100.0)
D_dimer = st.sidebar.number_input(label='D.dimer', min_value=0.01,
                                  max_value=3000.0,
                                  value=70.0,
                                  step=10.0)
GGT = st.sidebar.number_input(label='GGT', min_value=3.0,
                                  max_value=900.0,
                                  value=3.0,
                                  step=10.0)
ALT = st.sidebar.number_input(label='ALT', min_value=1.0,
                                  max_value=1000.0,
                                  value=1.0,
                                  step=10.0)
Neu = st.sidebar.number_input(label='Neu.', min_value=3.3,
                                  max_value=98.1,
                                  value=5.5,
                                  step=0.1)
LDH = st.sidebar.number_input(label='LDH', min_value=62.0,
                                  max_value=5550.0,
                                  value=62.0,
                                  step=10.0)
LDL_cholesterol = st.sidebar.number_input(label='LDL.cholesterol', min_value=0.08,
                                  max_value=8.40,
                                  value=6.2,
                                  step=0.01)
PT = st.sidebar.number_input(label='PT', min_value=8.7,
                                  max_value=120.0,
                                  value=10.2,
                                  step=0.1)
TT = st.sidebar.number_input(label='TT', min_value=10.4,
                                  max_value=300.0,
                                  value=10.4,
                                  step=0.1)

INR = st.sidebar.number_input(label='INR', min_value=0.69,
                                  max_value=11.90,
                                  value=6.5,
                                  step=0.1)
AST = st.sidebar.number_input(label='AST', min_value=4.4,
                                  max_value=1500.0,
                                  value=22.0,
                                  step=1.0)

Urea = st.sidebar.number_input(label='Urea', min_value=1.2,
                                  max_value=59.3,
                                  value=33.0,
                                  step=1.0)
Uric_acid = st.sidebar.number_input(label='Uric.acid', min_value=52.7,
                                  max_value=1100.0,
                                  value=272.0,
                                  step=1.0)

TC = st.sidebar.number_input(label='TC', min_value=0.98,
                                  max_value=11.87,
                                  value=5.5,
                                  step=1.0)

Cl = st.sidebar.number_input(label='Cl', min_value=64.0,
                                  max_value=134.0,
                                  value=64.0,
                                  step=0.1)

APTT = st.sidebar.number_input(label='APTT', min_value=15.0,
                                  max_value=180.0,
                                  value=27.0,
                                  step=1.0)



LYM= st.sidebar.number_input(label='LYM.', min_value=1.0,
                                  max_value=92.0,
                                  value=27.0,
                                  step=1.0)

Triglyceride = st.sidebar.number_input(label='Triglyceride', min_value=0.06,
                                  max_value=16.0,
                                  value=0.1,
                                  step=0.2)

LEU = st.sidebar.number_input(label='LEU', min_value=0.2,
                                  max_value=40.0,
                                  value=11.0,
                                  step=0.2)
Albumin = st.sidebar.number_input(label='Albumin', min_value=17.3,
                                  max_value=54.2,
                                  value=17.3,
                                  step=0.1)
P = st.sidebar.number_input(label='P', min_value=0.10,
                                  max_value=4.43,
                                  value=0.1,
                                  step=0.1)

RBC= st.sidebar.number_input(label='RBC', min_value=0.85,
                                  max_value=8.45,
                                  value=5.42,
                                  step=0.01)

HCT= st.sidebar.number_input(label='HCT', min_value=8.0,
                                  max_value=65.0,
                                  value=33.0,
                                  step=1.0)
Fibrinogen = st.sidebar.number_input(label='Fibrinogen', min_value=0.10,
                                  max_value=12.75,
                                  value=0.10,
                                  step=0.01)
Creatinine = st.sidebar.number_input(label='Creatinine', min_value=17.0,
                                  max_value=1000.0,
                                  value=17.0,
                                  step=10.0)
Cholinesterase = st.sidebar.number_input(label='Cholinesterase', min_value=0.01,
                                  max_value=22.0,
                                  value=0.01,
                                  step=0.01)
GLU = st.sidebar.number_input(label='GLU', min_value=0.46,
                                  max_value=46.0,
                                  value=0.46,
                                  step=0.01)
PLT = st.sidebar.number_input(label='PLT', min_value=3.0,
                                  max_value=600.0,
                                  value=152.0,
                                  step=1.0)
Hemoglobin = st.sidebar.number_input(label='Hemoglobin', min_value=28.0,
                                  max_value=195.0,
                                  value=52.0,
                                  step=1.0)

Ca = st.sidebar.number_input(label='Ca', min_value=0.60,
                                  max_value=4.12,
                                  value=0.60,
                                  step=0.01)
Na = st.sidebar.number_input(label='Na', min_value=107.3,
                                  max_value=165.1,
                                  value=142.2,
                                  step=0.1)
K = st.sidebar.number_input(label='K', min_value=0.10,
                                  max_value=4.43,
                                  value=0.1,
                                  step=0.1)




























features = {'Smoke': Smoke, 'Drink': Drink,
            'SBP': SBP, 'DBP': DBP,
            'Pulse': Pulse, 'Temperature': Temperature,
            'RR': RR, 'Insurance': Insurance,
            'Gender': Gender, 'Age': Age,
            'LOS': LOS, 'Hypertension': Hypertension,
            'CAD': CAD, 'NYHA3': NYHA3,
            'NYHA2': NYHA2, 'NYHA4': NYHA4,
            'Stroke': Stroke, 'Respiratory.failure': Respiratory_failure,
            'HHD': HHD, 'Ischemic.cardiomyopathy': Ischemic_cardiomyopathy,
            'AF': AF,'Carotid.arteriosclerosis': Carotid_arteriosclerosis, 'Osteoporosis': Osteoporosis,
            'Hypoproteinemia': Hypoproteinemia, 'Hyperuricemia': Hyperuricemia,
            'CRF': CRF, 'NT.proBNP': NTproBNP,
            'D.dimer': D_dimer, 'GGT': GGT,
            'ALT': ALT, 'Neu.': Neu,
            'LDH': LDH, 'LDL.cholesterol': LDL_cholesterol,
            'PT': PT, 'TT': TT,
            'INR': INR, 'AST': AST,
            'Urea': Urea, 'Uric.acid': Uric_acid,
            'TC': TC, 'Cl': Cl,
            'APTT': APTT,'LYM.': LYM, 'Triglyceride': Triglyceride,
            'LEU': LEU, 'Albumin': Albumin,
            'P': P, 'RBC': RBC,
            'HCT': HCT, 'Fibrinogen': Fibrinogen,
            'Creatinine': Creatinine, 'Cholinesterase': Cholinesterase,
            'GLU': GLU, 'PLT': PLT,
            'Hemoglobin': Hemoglobin, 'Ca': Ca,
            'Na': Na, 'K': K,
            }

features_df = pd.DataFrame([features])
#显示输入的特征
st.table(features_df)

from skimage import io,data

#显示预测结果与shap解释图
if st.button('Predict'):
    prediction = predict_quality(model, features_df)
    st.write("the probability of MINS:")
    st.success(round(prediction[0], 4))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    shap.force_plot(explainer.expected_value, shap_values[0], features_df, matplotlib=True, show=False)
    plt.subplots_adjust(top=0.67,
                        bottom=0.0,
                        left=0.1,
                        right=0.9,
                        hspace=0.2,
                        wspace=0.2)
    plt.savefig('test_shap.png')
    st.image('test_shap.png', caption='Individual prediction explaination', use_column_width=True)