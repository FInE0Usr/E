import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image

# Function to load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv('Obesity prediction.csv')
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Split the data into features and target
    X = df.drop('Obesity', axis=1)
    y = df['Obesity']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model, label encoders, and scaler
    joblib.dump(model, 'obesity_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, label_encoders, scaler

# Function for the Machine Learning content page
def ml_content_page():
    st.title("Machine Learning Content")
    st.image("https://img2.pic.in.th/pic/E969E099-2388-4D2F-ADE4-3C17107010E6.jpeg", width=600, use_container_width=False)
    st.write("""
    ### Step 1: นำเข้าไลบรารีที่จำเป็น
    - ใช้ pandas สำหรับจัดการข้อมูล
    - ใช้ sklearn สำหรับการแบ่งชุดข้อมูล, แปลงข้อมูล, และสร้างโมเดล
    
    ### Step 2: โหลดชุดข้อมูล
    - อัปโหลดไฟล์ CSV ใน Google Colab
    - อ่านข้อมูลเข้า pandas DataFrame
    
    ### Step 3: เตรียมข้อมูล (Data Preprocessing)
    1.เข้ารหัสค่าข้อความ (Categorical Encoding) → แปลงข้อมูลที่เป็นข้อความเป็นตัวเลข
    2.แยกคุณลักษณะ (Features) และ เป้าหมาย (Target Variable)
      - X = ข้อมูลคุณลักษณะ
      - y = ค่าผลลัพธ์ (Obesity)
    3.แบ่งข้อมูลออกเป็นชุดฝึก (Train) และชุดทดสอบ (Test) (80:20)
    4.ปรับมาตรฐานข้อมูล (Standardization) → ทำให้ข้อมูลอยู่ในช่วงที่เหมาะสม

    ### Step 4: ฝึกโมเดล (Train Model)
    -ใช้ Random Forest Classifier (100 ต้นไม้) เพื่อฝึกโมเดล
    
    ### Step 5: ทดสอบและประเมินผล (Evaluate Model)
    -ใช้โมเดลทำนายผล
    -คำนวณค่าความแม่นยำ (accuracy_score)
    -แสดง Classification Report เพื่อตรวจสอบประสิทธิภาพ
    
    ### Step 6: บันทึกโมเดล (Save Model) [Optional]
    -ใช้ joblib บันทึกโมเดลลงไฟล์ (obesity_model.pkl) เพื่อใช้ในอนาคต
    """)

# Function for the Machine Learning demo page
def ml_demo_page():
    st.title("Machine Learning Demo")
    st.write("### Predict Obesity Level")
    
    # Load the model, label encoders, and scaler
    model = joblib.load('obesity_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Input fields for user data
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    height = st.number_input("Height (in meters)", min_value=0.5, max_value=2.5, value=1.65)
    weight = st.number_input("Weight (in kg)", min_value=10, max_value=200, value=70)
    family_history = st.selectbox("Family History of Obesity", ["yes", "no"])
    favc = st.selectbox("Frequent Consumption of High-Caloric Food", ["yes", "no"])
    fcvc = st.slider("Frequency of Consuming Vegetables (1-3)", 1, 3, 2)
    ncp = st.slider("Number of Main Meals per Day (1-4)", 1, 4, 3)
    caec = st.selectbox("Consumption of Food Between Meals", ["Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Do You Smoke?", ["yes", "no"])
    ch2o = st.slider("Daily Water Consumption (1-3)", 1, 3, 2)
    scc = st.selectbox("Do You Monitor Calorie Intake?", ["yes", "no"])
    faf = st.slider("Physical Activity Frequency (0-3)", 0, 3, 1)
    tue = st.slider("Time Using Technology Devices (0-2)", 0, 2, 1)
    calc = st.selectbox("Consumption of Alcohol", ["no", "Sometimes", "Frequently"])
    mtrans = st.selectbox("Transportation Method", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])
    
    # Create a dictionary from the input data
    input_data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables using the loaded label encoders
    for column in input_df.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            input_df[column] = label_encoders[column].transform(input_df[column])
    
    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_df)
    
    # Predict the obesity level
    if st.button("Predict Obesity Level"):
        prediction = model.predict(input_scaled)
        predicted_label = label_encoders['Obesity'].inverse_transform(prediction)
        st.success(f"Predicted Obesity Level: **{predicted_label[0]}**")

# Function for the Neural Network content page
def nn_content_page():
    st.title("Neural Network Content")
    st.image("https://img5.pic.in.th/file/secure-sv1/DE703F19-71F1-416B-9B93-369E0193F1FA.jpeg", width=600, use_container_width=False)
    st.write("""
    
    ### 1.ติดตั้ง Kaggle API
    -ใช้เพื่อดาวน์โหลดชุดข้อมูลจาก Kaggle
    ### 2.อัปโหลดไฟล์ kaggle.json
    -ไฟล์นี้ใช้สำหรับการยืนยันตัวตนกับ Kaggle API
    ### 3.ดาวน์โหลดชุดข้อมูล
    -ดาวน์โหลดชุดข้อมูล one-piece-image-classifier จาก Kaggle
    ### 4.แตกไฟล์ zip
    -แตกไฟล์ชุดข้อมูลที่ดาวน์โหลดมา
    ### 5.ติดตั้ง TensorFlow และ Keras
    -สำหรับการสร้างและฝึกโมเดล Neural Network
    ### 6.กำหนดไดเรกทอรี
    -กำหนดไดเรกทอรีสำหรับข้อมูลฝึก (train_dir) และข้อมูลตรวจสอบ (validation_dir)

    """)
    st.image("https://img2.pic.in.th/pic/A338E49D-5340-48E9-891C-070F6D4DAB05.jpeg", width=600, use_container_width=False)
    st.write("""

    ### นำเข้าไลบรารี
    -นำเข้า TensorFlow และ ImageDataGenerator จาก Keras เพื่อจัดการข้อมูลภาพ
    ### กำหนดไดเรกทอรี
    -กำหนดไดเรกทอรีสำหรับข้อมูลฝึก (train_dir) และข้อมูลตรวจสอบ (validation_dir)
    ### สร้าง ImageDataGenerator
    -สำหรับข้อมูลฝึก: เพิ่มการ augmentation (เช่น การหมุน, การย้าย, การซูม) เพื่อเพิ่มความหลากหลายของข้อมูล
    -สำหรับข้อมูลตรวจสอบ: ปรับขนาดภาพเท่านั้น (rescale)
    ### สร้าง Data Generator
    -train_generator: สร้าง generator สำหรับข้อมูลฝึก โดยปรับขนาดภาพเป็น 150x150 และแบ่งเป็น batch ขนาด 32
    -validation_generator: สร้าง generator สำหรับข้อมูลตรวจสอบ โดยปรับขนาดภาพเป็น 150x150 และแบ่งเป็น batch ขนาด 32

    """)
    st.image("https://img2.pic.in.th/pic/79B7FCF4-9025-45FE-B1A1-82C7FC9A68F6.jpeg", width=600, use_container_width=False)
    st.write("""

    ### นำเข้าไลบรารี
    -นำเข้า Sequential model และ layer ต่างๆ จาก Keras เช่น Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    
    ### สร้างโมเดล CNN
    -Conv2D: เลเยอร์ convolutional สำหรับสกัดคุณลักษณะจากภาพ
    -MaxPooling2D: เลเยอร์ pooling เพื่อลดขนาดข้อมูล
    -Flatten: แปลงข้อมูล 2D เป็น 1D
    -Dense: เลเยอร์ fully connected สำหรับการจำแนก
    -Dropout: เพื่อป้องกัน overfitting
    
    ### คอมไพล์โมเดล
    -ใช้ optimizer adam และ loss function categorical_crossentropy สำหรับการฝึก
    ### แสดงสรุปโมเดล
    -แสดงโครงสร้างของโมเดล
    ### ฝึกโมเดล
    -ใช้ train_generator และ validation_generator เพื่อฝึกโมเดลเป็นเวลา 30 epochs
   
    """)
    st.image("https://img5.pic.in.th/file/secure-sv1/A4185653-181C-4392-993D-34F6CA69A075_4_5005_c.jpeg", width=600, use_container_width=False)
    st.write("""

    ### บันทึกโมเดล
    -ใช้ model.save() เพื่อบันทึกโมเดลที่ฝึกไว้ในรูปแบบไฟล์ HDF5 (one_piece_character_classifier.h5)
    -มีคำเตือนเกี่ยวกับการบันทึกโมเดลในรูปแบบ HDF5 ซึ่งเป็นรูปแบบที่ TensorFlow รองรับ
    ### เชื่อมต่อ Google Drive
    -ใช้ drive.mount() เพื่อเชื่อมต่อ Google Drive กับ Google Colab โดยให้ผู้ใช้เข้าสู่ระบบและอนุญาตการเข้าถึง
    """)

# Function for the Neural Network demo page
def nn_demo_page():
    st.title("Neural Network Demo")
    st.write("### One Piece Character Classifier")
    st.write("### อัปโหลดภาพตัวละครใน One Piece เพื่อทำนายว่าเป็นใคร (Luffy, Sanji, Zoro)")

    # โหลดโมเดล
    @st.cache_resource  # Cache โมเดลเพื่อเพิ่มประสิทธิภาพ
    def load_model():
        model = tf.keras.models.load_model('one_piece_character_classifier.h5')
        return model

    model = load_model()

    # ฟังก์ชันทำนายภาพ
    def predict_character(img):
        # ปรับขนาดภาพให้ตรงกับขนาดที่โมเดลต้องการ (150x150)
        img = img.resize((150, 150))
        # แปลงภาพเป็น array และปรับค่า pixel ให้อยู่ระหว่าง 0 ถึง 1
        img_array = np.array(img)  # แปลง PIL.Image เป็น numpy array
        img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติ batch
        img_array = img_array.astype('float32') / 255.0  # ปรับค่า pixel ให้อยู่ระหว่าง 0 ถึง 1

        # ทำนาย
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        return predicted_class

    # ชื่อคลาส (ตัวละคร)
    class_names = ['Luffy', 'Sanji', 'Zoro']  # เปลี่ยนเป็นชื่อคลาสของคุณ

    # ส่วนติดต่อผู้ใช้
    st.title('One Piece Character Classifier')
    st.write('อัปโหลดภาพตัวละครใน One Piece เพื่อทำนายว่าเป็นใคร')

    # อัปโหลดภาพ
    uploaded_file = st.file_uploader("เลือกภาพ...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # แสดงภาพที่อัปโหลด
        image = Image.open(uploaded_file)
        st.image(image, caption='ภาพที่อัปโหลด', use_column_width=True)
        st.write("")

        # ทำนายภาพ
        st.write("กำลังทำนาย...")
        predicted_class = predict_character(image)
        st.write(f"ผลลัพธ์: **{class_names[predicted_class[0]]}**")

# Main function to run the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Machine Learning Content", "Machine Learning Demo", "Neural Network Content", "Neural Network Demo"])
    
    if page == "Machine Learning Content":
        ml_content_page()
    elif page == "Machine Learning Demo":
        ml_demo_page()
    elif page == "Neural Network Content":
        nn_content_page()
    elif page == "Neural Network Demo":
        nn_demo_page()

# Run the app
if __name__ == "__main__":
    main()
