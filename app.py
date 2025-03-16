import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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
