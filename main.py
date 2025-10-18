import streamlit as st
import tensorflow as tf
import keras
from keras import preprocessing
import numpy as np
import pandas as pd

import os
import warnings
# TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)



st.set_page_config(
    page_title = 'Food Image Recognition',
    page_icon = '🍔'
)

# <----- Homepage ---->
homepage = """
<div id = 'parent1'>
    <div id = 'blank1'></div>
    <div class = 'nav_bar'>
        <div class = 'first_head'>
            FIRCE
        </div>
        <div class = 'btns'>
            <ul class = 'btn_nav'>
                <li><a href='#blank1'>Home</a></li>
                <li><a href='#blank2'>How it Works?</a></li>
                <li><a href='#blank3'>FNV</a></li>
                <li><a href='#blank4'>Junk/Fast Food</a></li>
                <li><a href='#blank5'>About</a></li>
                <li><a href='#blank6'>FAQs</a></li>
                <li><a href='#blank7'>Contact</a></li>
            </ul>
        </div>
    </div>
    <img class = 'logo' src="data:image/jpeg;base64,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" alt="image4">
    <div class="main_head">
            <div class=" head_1">
                FOOD IMAGE RECOGNITION
            </div>
            <div class=" head_2">
                & CALORIE ESTIMATION
            </div>
        </div>
    </div>
    <div id = 'black_bg1'></div>
    <div class = 'parent_sub_head1'>
        <div class="sub_head">
            Welcome to Our Platform!
        </div>
        <div class="desc">
            At FIRCE, we are passionate about empowering individuals to make informed decisions about their nutrition. Our cutting-edge technology combines the power of image recognition with advanced algorithms to revolutionize how you track your food intake and make healthier choices.
        </div>
    </div>
    <div id = 'black_bg2'></div>
    <div class="parent_sub_head2">
        <div class="sub_head">
            What We Offer:
        </div>
        <div class="desc">
            <strong>- Food Detection :</strong> Upload an image of any fruit or vegetable, and our intelligent system will detect it and tell that what fruit or vegetable it is.
        </div>
        <div class="desc">
            <strong>- Accurate Calorie Estimation :</strong> Our intelligent system will analyze it to provide accurate calorie estimation and nutritional information. Say goodbye to manual logging and guesswork!
        </div>
        <div class="desc">
            <strong>- Easy-to-Use Interface :</strong> Our user-friendly interface makes it effortless to navigate through the process of uploading images, receiving instant results, and accessing additional resources for a deeper understanding of nutrition.
        </div>
        <div class="desc">
            <strong>- Personalized Recommendations :</strong> Based on your dietary preferences and goals, receive personalized recommendations and insights to help you achieve your health and wellness objectives.
        </div>
        <div id = 'blank2'></div>
    </div>
</div>
"""
st.markdown(homepage, unsafe_allow_html=True)

homepage_css = """
<style>

    [data-testid="stAppViewContainer"] {
        background-size: cover;
        background-attachment: fixed;
        background-image: linear-gradient(rgba(0,0,0,0.5),rgba(0,0,0,0.5)), url("https://img.freepik.com/premium-photo/fruits-berries-summer_82893-12212.jpg?w=996");
    }

    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }

    *{
        margin: 0;
        padding: 0;
        font-family: "Poppins", sans-serif;
        color: ghostwhite;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        scroll-behavior: smooth;
    }

    .st-emotion-cache-1avcm0n{
        display: none;
    }

    #parent1{
        height: 900px;
        display: flex;
    }

    .first_head {
        font-size: 40px;
        font-style: normal;
        position: absolute;
        left: 50px;
        color: gold;
    }

    .nav_bar{
        background-color: black;
        position: fixed;
        top: 0px;
        width: 100%;
        height: 60px;
        z-index: 1000;
        left: 0px;
    }

    .btns{
        display: flex;
        position: absolute;
        left: 770px;
        top: 13px;
        font-size: 15px;
        width: 90%;
    }

    .btn_nav{
        display: flex;
    }

    .btn_nav li {
        width: 40%;
        list-style: none;
        white-space: nowrap;
        word-wrap: normal;
        height: 30px;
    }

    .btn_nav li a {
        text-decoration: none;
        color: inherit;
        color: gold;
    }

    .main_head{
        font-size: 70px;
        position: absolute;
        top: 110px;
        right: 480px;
    }

    .head_1{
        width: 200%;
        color: gold;
    }

    .head_2{
        margin-top: -37px;
        width: 170%;
        color: gold;
    }

    #black_bg1{
        background-color: rgba(0, 0, 0, 0.4);
        width: 128%;
        height: 180px;
        position: absolute;
        top: 400px;
        left: -270px;
    }

    .parent_sub_head1{
        position: relative;
        top: -487px;
        left: -245px;
    }

    .sub_head{
        font-size: 30px;
        text-decoration: underline;
    }

    .desc{
        font-size: 18px;
        text-align: justify;
        width: 120%;
        margin-top: 12px;
    }

    #black_bg2{
        background-color: rgba(0, 0, 0, 0.4);
        width: 128%;
        height: 400px;
        position: relative;
        top: -365px;
        left: -270px;
    }

    .parent_sub_head2{
        position: relative;
        top: -751px;
        left: -240px;
        font-size: 15px;
        text-align: justify;
        width: 98%;
    }

    #blank1{
        height: 1px;
        width: 1%;
        position: absolute;
        top: 320px;
    }

    #blank2{
        height: 1px;
        width: 1%;
        position: absolute;
        top: 440px;
    }

    .logo{
        height: 96px;
        position: fixed;
        top: 72px;
        left: 1440px;
        border-radius: 25px;
    }

</style>
"""
st.markdown(homepage_css, unsafe_allow_html=True)

# <---- Working ---->
working = """
<div class = 'parent2'>
    <div id = 'black_bg3'></div>
    <div class = 'parent2_child1'>
        <div class = 'sub_head'>
            How it Works?
        </div>
        <div class = 'desc2'>
            Curious about how our food image recognition and calorie estimation platform functions? Let us walk you through the process step by step:
        </div>
        <div class = 'parent2_points'>
            <strong>1. Upload Your Image</strong>
        </div>
        <div class = 'desc2'>
            Begin by uploading an image of your meal using our intuitive interface. Whether it's a snapshot from your phone or a picture from your computer, our platform can analyze a wide range of food images.
        </div>
        <div class = 'parent2_points'>
            <strong>2. Image Recognition</strong>
        </div>
        <div class = 'desc2'>
            Once your image is uploaded, our advanced image recognition technology goes to work. Our algorithms meticulously scan the image to identify individual food items, taking into account factors such as size, shape, color, and texture.
        </div>
        <div class = 'parent2_points'>
            <strong>3. Nutritional Analysis</strong>
        </div>
        <div class = 'desc2'>
            With the food items identified, our platform then cross-references them with an extensive database of nutritional information. This database contains data on thousands of foods, including their calorie content, macronutrient composition, vitamins, and minerals.
        </div>
        <div class = 'parent2_points'>
            <strong>4. Calorie Estimation</strong>
        </div>
        <div class = 'desc2'>
            Using the nutritional information obtained from the database, our system calculates the calorie content of each food item in your image. By summing up the calories from all the identified items, we provide you with an accurate estimation of the total calorie count for your meal.
        </div>
        <div class = 'parent2_points'>
            <strong>5. Instant Results</strong>
        </div>
        <div class = 'desc2'>
            In just a matter of seconds, you'll receive instant results directly on your screen. Our user-friendly interface presents the analyzed image alongside detailed nutritional information, making it easy for you to understand and interpret the results.
        </div>
    </div>
    <div class = 'parent2_child2'>
        <div style="font-size: 30px;">
            <strong>Ready to Get Started?</strong>
        </div>
        <div>
            Experience the convenience, accuracy, and reliability of our food image recognition and calorie estimation platform today. Upload an image of your meal and discover a new way to track your nutrition effortlessly.
        </div>
    </div>
</div>
"""
st.markdown(working, unsafe_allow_html=True)

working_css = """
<style>
    #black_bg3{
        background-color: rgba(0, 0, 0, 0.4);
        width: 130%;
        height: 850px;
        position: relative;
        top: -602px;
        left: -269px;
    }

    .parent2_child1{
        position: absolute;
        left: -240px;
        top: -582px;
    }

    .parent2_points{
        margin-top: 20px;
    }

    .desc2{
        font-size: 18px;
        text-align: justify;
        width: 89%;
        margin-top: 12px;
    }

    .parent2_child2{
        font-size: 40;
        position: relative;
        top: -570px;
        left: -268px;
        width: 127%;
        text-align: justify;
    }

</style>
"""
st.markdown(working_css, unsafe_allow_html=True)

# <---- Fruit/Vegetable ---->
# Prediction Model Function
def model_prediction(test_image, confidence_threshold=0.8):
    model = keras.models.load_model("C:/Users/HP/Desktop/FIRS/Project/trained_model_data.h5", compile=False)
    image = preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])                   # Convert single image to batch
    prediction = model.predict(input_arr)
    predicted_class_prob = np.max(prediction)
    predicted_class_index = np.argmax(prediction)
    if predicted_class_prob >= confidence_threshold:
        return predicted_class_index, predicted_class_prob
    else:
        return None, predicted_class_prob

fruit_1 = """
<div id = 'black_bg4'></div>
<div class = 'parent3_1'>
    <div class = 'sub_head'>
        Fruit/Vegetable Prediction Model
    </div>
    <div id = 'blank3'></div>
    <div class = 'parent3_desc1'>
        Upload an image of any fruit or vegetable below and let our advanced image recognition model analyze it to provide you with an accurate calorie estimation along with its benefits.
    </div>
</div>
"""
st.markdown(fruit_1, unsafe_allow_html=True)

test_image = st.file_uploader('')

fruit_2 = """
    <div class = 'parent3_desc2'>
        (Once the image is uploaded, our model will identify the food item present in the image..)
    </div>
"""

st.markdown(fruit_2, unsafe_allow_html=True)

confidence_threshold = 0.8

if st.button('Predict', key='predict1'):
    if test_image is None:
        st.markdown("<p class='st-ay my-custom-success3'>Please upload an image.</p>", unsafe_allow_html=True)
    
    else:
        # st.image(test_image, caption='Uploaded Image', use_column_width=True)
        result_index, confidence_score = model_prediction(test_image)
        
        if result_index is not None:
            with open("C:/Users/HP/Desktop/FIRS/Project/labels.txt") as f:
                content = f.readlines()
            label = [i.strip() for i in content]
            if result_index < len(label):
                if confidence_score >= confidence_threshold:
                    st.markdown(f"<p class='st-ay my-custom-success'>It's {label[result_index]}</p>", unsafe_allow_html=True)

                    # Calorie Estimation
                    # Retrieve fruit/vegetable name corresponding to predicted class index
                    with open("C:/Users/HP/Desktop/FIRS/Project/labels.txt") as f:
                        labels = f.readlines()

                    predicted_fruit_vegetable = labels[result_index].strip()  # Remove trailing newline character

                    calorie_data = pd.read_csv('C:/Users/HP/Desktop/FIRS/Project/Calorie Dataset.csv')

                    calorie_dict = dict(zip(calorie_data['Fruit/Vegetable'], calorie_data['Calorie (kcal)']))
                    fiber_dict = dict(zip(calorie_data['Fruit/Vegetable'], calorie_data['Dietary Fiber (g)']))
                    water_dict = dict(zip(calorie_data['Fruit/Vegetable'], calorie_data['Water (g)']))
                    fat_dict = dict(zip(calorie_data['Fruit/Vegetable'], calorie_data['Total Fat (g)']))
                    protein_dict = dict(zip(calorie_data['Fruit/Vegetable'], calorie_data['Protein (g)']))
                    carbo_dict = dict(zip(calorie_data['Fruit/Vegetable'], calorie_data['Carbohydrates (g)']))
                    vitamin_dict = dict(zip(calorie_data['Fruit/Vegetable'], calorie_data['Vitamins']))

                    if predicted_fruit_vegetable in calorie_dict:

                        calorie_info = calorie_dict[predicted_fruit_vegetable]
                        fiber_info = fiber_dict[predicted_fruit_vegetable]
                        water_info = water_dict[predicted_fruit_vegetable]
                        fat_info = fat_dict[predicted_fruit_vegetable]
                        protein_info = protein_dict[predicted_fruit_vegetable]
                        carbo_info = carbo_dict[predicted_fruit_vegetable]
                        vitamin_info = vitamin_dict[predicted_fruit_vegetable]

                        calorie_info_div = f"""
                            <div class="calorie-info">
                                <p><strong>Calorie :</strong> {calorie_info} kcal</p>
                                <p><strong>Dietary Fiber :</strong> {fiber_info} g</p>
                                <p><strong>Water :</strong> {water_info} g</p>
                                <p><strong>Total Fat :</strong> {fat_info} g</p>
                                <p><strong>Protein :</strong> {protein_info} g</p>
                                <p><strong>Carbohydrates :</strong> {carbo_info} g</p>
                                <p><strong>Vitamins :</strong> {vitamin_info}</p>
                            </div>
                            """
                        st.markdown(calorie_info_div, unsafe_allow_html=True)

                    else:
                        st.error(f'Calorie information for {predicted_fruit_vegetable} not found in the dataset.')

                    # Benefits
                    food_benefits_df = pd.read_csv("C:/Users/HP/Desktop/FIRS/Project/Benefits Dataset.csv")

                    if predicted_fruit_vegetable in food_benefits_df['Fruit/Vegetable'].values:
                        benefits = food_benefits_df.loc[food_benefits_df['Fruit/Vegetable'] == predicted_fruit_vegetable, 'Benefits'].iloc[0]
                        benefits_list = benefits.split(';')  # Split benefits into a list
                        bnft_label = """
                            <div class = 'bnft_label'><strong>Benefits :</strong></div>
                        """
                        st.markdown(bnft_label, unsafe_allow_html=True)

                        for benefit in benefits_list:
                            benefit_div = f"""<div class = 'bnft'>
                                <p>{benefit.strip()}</p>
                            </div>
                            """
                            st.markdown(benefit_div, unsafe_allow_html=True)
                    else:
                        st.error(f'Benefits for {predicted_fruit_vegetable} not found in the dataset.')
                else:
                    st.markdown("<p class='st-ay my-custom-success3'>The uploaded image could not be recognized.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='st-ay my-custom-success3'>The uploaded image could not be recognized.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='st-ay my-custom-success3'>The uploaded image could not be recognized.</p>", unsafe_allow_html=True)


fruit_css = """
<style>
    .parent3_1{
        position: relative;
        top: -508px;
        left: -248px;
        width: 127%;
        text-align: justify;
    }

    #black_bg4{
        background-color: rgba(0, 0, 0, 0.4);
        width: 130%;
        height: 329px;
        position: absolute;
        top: -522px;
        left: -270px;
    }

    #blank3{
        height: 1px;
        width: 1%;
        position: relative;
        top: -150px;
    }

    .parent3_desc1{
        font-size: 18px;
        text-align: justify;
        width: 97%;
        margin-top: 12px;
    }

    .st-emotion-cache-1erivf3{
        position: absolute;
        width: 60%;
        top: -489px;
        left: -252px;
    }

    .parent3_desc2{
        font-size: 15px;
        text-align: justify;
        width: 101%;
        position: relative;
        top: -447px;
        left: -249px;
    }

    .st-emotion-cache-19rxjzo ef3psqc12{
        position: relative;
        left: -270px;
    }

    .st-emotion-cache-fis6aj{
        position: absolute;
        top: -483px;
        left: 326px;
        width: 37%;
    }

    .stButton button:nth-child(1) {
        position: relative;
        left: -249px;
        top: -439px;
    }

    .my-custom-success {
        font-size: 18px;
        text-align: center;
        margin-top: 20px;
        position: relative;
        left: -16px;
        top: -510px;
        border: 2px solid green;
        width: 20%;
        border-radius: 5px;
    }

    .calorie-info {
        position: absolute;
        top: -450px;
        left: -180px; 
    }

    .bnft_label{
        margin-bottom: 25px;
        position: relative;
        top: -465px;
        left: 300px;
        text-decoration: underline;
    }

    .bnft{
        position: relative;
        top: -460px;
        left: 300px;
    }

</style>
"""
st.markdown(fruit_css, unsafe_allow_html=True)

# <---- Junk/Fast Food ---->
# Prediction Model Function
def model_prediction2(test_image2, confidence_threshold2=0.8):
    model2 = keras.models.load_model("C:/Users/HP/Desktop/FIRS/Project/trained_model_junk_food.h5", compile=False)
    image2 = preprocessing.image.load_img(test_image2, target_size=(64, 64))
    input_arr2 = preprocessing.image.img_to_array(image2)
    input_arr2 = np.array([input_arr2])  # Convert single image to batch
    prediction2 = model2.predict(input_arr2)
    predicted_class_prob2 = np.max(prediction2)
    predicted_class_index2 = np.argmax(prediction2)
    if predicted_class_prob2 >= confidence_threshold2:
        return predicted_class_index2, predicted_class_prob2
    else:
        return None, predicted_class_prob2

ff_1 = """
<div id = 'black_bg5'></div>
<div class = 'parent4_1'>
    <div class = 'sub_head'> Junk/Fast Food Prediction Model </div>
    <div id = 'blank4'></div>
    <div class = 'parent3_desc1'>
        Upload an image of any junk/fast food below and let our advanced image recognition model analyze it to provide you with an accurate calorie estimation.
    </div>
</div>
"""
st.markdown(ff_1, unsafe_allow_html=True)


ff_2 = """
    <div id = 'blank4_2'></div>
    <div class = 'parent4_desc2'>
        (Once the image is uploaded, our model will identify the food item present in the image..)
    </div>
"""
st.markdown(ff_2, unsafe_allow_html=True)

test_image2 = st.file_uploader('', key = 'image2')

ff_3 = """
    <div id = 'blank4_3'></div>
"""
st.markdown(ff_3, unsafe_allow_html=True)

confidence_threshold2 = 0.8

if st.button('Predict', key='predict2'):
    if test_image2 is None:
        st.markdown("<p class='st-ay my-custom-success3'>Please upload an image.</p>", unsafe_allow_html=True)
    else:
        result_index2, confidence_score2 = model_prediction2(test_image2)
        if result_index2 is not None:
            with open("C:/Users/HP/Desktop/FIRS/Project/ff_labels.txt") as f:
                content2 = f.readlines()
            label2 = [i2.strip() for i2 in content2]
            if result_index2 < len(label2):
                if confidence_score2 >= confidence_threshold2:
                    st.markdown(f"<p class='st-ay my-custom-success2'>It's {label2[result_index2]}</p>", unsafe_allow_html=True)

                    # Calorie Estimation
                    # Retrieve junk/fast food name corresponding to predicted class index
                    with open("C:/Users/HP/Desktop/FIRS/Project/ff_labels.txt") as f:
                        labels2 = f.readlines()

                    predicted_food = labels2[result_index2].strip()  # Remove trailing newline character

                    calorie_data2 = pd.read_csv('C:/Users/HP/Desktop/FIRS/Project/FF Calorie Dataset.csv')

                    calorie_dict2 = dict(zip(calorie_data2['Food Item'], calorie_data2['Calories (kcal)']))
                    saturated_dict = dict(zip(calorie_data2['Food Item'], calorie_data2['Saturated Fat (g)']))
                    trans_dict = dict(zip(calorie_data2['Food Item'], calorie_data2['Trans Fat (g)']))
                    sugar_dict = dict(zip(calorie_data2['Food Item'], calorie_data2['Sugar (g)']))
                    protein_dict2 = dict(zip(calorie_data2['Food Item'], calorie_data2['Protein (g)']))

                    if predicted_food in calorie_dict2:
                        calorie_info2 = calorie_dict2[predicted_food]
                        saturated_info = saturated_dict[predicted_food]
                        trans_info = trans_dict[predicted_food]
                        sugar_info = sugar_dict[predicted_food]
                        protein_info2 = protein_dict2[predicted_food]

                        calorie_info_div2 = f"""
                            <div class="calorie-info2">
                                <p><strong>Calorie :</strong> {calorie_info2} kcal</p>
                                <p><strong>Saturated Fat :</strong> {saturated_info} g</p>
                                <p><strong>Trans Fat :</strong> {trans_info} g</p>
                                <p><strong>Sugar :</strong> {sugar_info} g</p>
                                <p><strong>Protein :</strong> {protein_info2} g</p>
                            </div>
                            """
                        st.markdown(calorie_info_div2, unsafe_allow_html=True)

                    else:
                        st.error(f'Calorie information for {predicted_food} not found in the dataset.')
                else:
                    st.markdown("<p class='st-ay my-custom-success3'>The uploaded image could not be recognized.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='st-ay my-custom-success3'>The uploaded image could not be recognized.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='st-ay my-custom-success3'>The uploaded image could not be recognized.</p>", unsafe_allow_html=True)


ff_css = """
<style>
    .parent4_1{
        position: relative;
        top: -308px;
        left: -248px;
        width: 127%;
        text-align: justify;
    }

    #black_bg5{
        background-color: rgba(0, 0, 0, 0.4);
        width: 130%;
        height: 317px;
        position: absolute;
        top: -322px;
        left: -270px;
    }

    #blank4{
        height: 1px;
        width: 1%;
        position: relative;
        top: -150px;
    }

    #blank4_2{
        height: 165px;
        width: 1%;
        position: relative;
        top: -150px;
    }

    #blank4_3{
        height: 25px;
        width: 1%;
        position: relative;
        top: -150px;
    }

    .st-emotion-cache-1erivf3{
        position: absolute;
        width: 60%;
        top: -489px;
        left: -252px;
    }

    .parent4_desc2{
        font-size: 15px;
        text-align: justify;
        width: 101%;
        position: relative;
        top: -377px;
        left: -249px;
    }

    .st-emotion-cache-19rxjzo ef3psqc12{
        position: relative;
        left: -270px;
    }

    .st-emotion-cache-fis6aj{
        position: absolute;
        top: -483px;
        left: 326px;
        width: 37%;
    }

    .stButton button:nth-child(1) {
        position: relative;
        left: -249px;
        top: -439px;
    }

    .my-custom-success2 {
        font-size: 18px;
        text-align: center;
        margin-top: 20px;
        position: relative;
        left: -16px;
        top: -510px;
        border: 2px solid green;
        width: 30%;
        border-radius: 5px;
    }

    .my-custom-success3 {
        font-size: 18px;
        text-align: center;
        margin-top: 20px;
        position: relative;
        left: -60px;
        top: -510px;
        border: 2px solid red;
        width: 53%;
        border-radius: 5px;
    }

    #image2 {
        position: relative;
        top: -350px;
    }

    .calorie-info2 {
        position: absolute;
        top: -450px;
        left: -180px; 
    }

</style>
"""
st.markdown(ff_css, unsafe_allow_html=True)

# <----- Diet Tracker Button ---->
diet_tracker_btn = """
    <div>
        <a href="#blank_diet"><img class='diet_tracker_photo' src="https://th.bing.com/th/id/R.38ac3f5cd7344b38430dcf65a0aea6c0?rik=4FL%2byCSY%2bUkz4w&riu=http%3a%2f%2ficonbug.com%2fdata%2fad%2f256%2ff1320fedfc185a488c5d4806c3f01abe.png&ehk=DjpkeowaR1iLY7vcmprZzlAmOyDQRiHnBlQEAK18VYs%3d&risl=&pid=ImgRaw&r=0" alt="image"></a>
    </div>
"""
st.markdown(diet_tracker_btn, unsafe_allow_html=True)

# <----- Diet Planner ---->
# Function to recommend food items based on desired calorie intake
def recommend_foods_for_calorie_intake(calorie_intake, dataset):
    filtered_food_items = dataset[dataset['Calories (kcal/100g)'] <= calorie_intake]
    recommended_food_items = filtered_food_items.sort_values(by='Calories (kcal/100g)', ascending=False)
    return recommended_food_items

# Function to recommend food items based on desired protein intake
def recommend_foods_for_protein_intake(protein_intake, dataset):
    filtered_food_items = dataset[dataset['Protein (g/100g)'] >= protein_intake]
    recommended_food_items = filtered_food_items.sort_values(by='Protein (g/100g)', ascending=True)
    return recommended_food_items

def display_recommendations(recommended_foods):
    if recommended_foods is not None and not recommended_foods.empty:
        st.subheader("Recommended Food Items:")
        st.write(recommended_foods)
    else:
        st.error("Unable to recommend foods. Please check the input value.")

diet_1 = """
    <div id = 'blank_diet'></div>
    <div id = 'black_bg_diet'></div>
    <div class = 'parent_diet'>
        <div class = 'sub_head_diet'>
            Diet Planner
        </div>
    </div>
"""
st.markdown(diet_1, unsafe_allow_html=True)

diet_dataset = pd.read_csv("C:/Users/HP/Desktop/FIRS/Project/Diet Planner.csv")

# Radio button for selecting recommendation type
recommendation_type = st.radio("Choose recommendation type:", ("Calorie", "Protein"))

if recommendation_type == "Calorie":
    calorie_intake = st.number_input("Enter the amount of calorie you want to eat at the moment (in kcal):", min_value=0, step=1)
    if st.button("Get Recommendations"):
        recommended_foods = recommend_foods_for_calorie_intake(calorie_intake, diet_dataset)
        display_recommendations(recommended_foods)
elif recommendation_type == "Protein":
    protein_intake = st.number_input("Enter your desired amount of protein intake at the moment (in grams):", min_value=0, step=1)
    if st.button("Get Recommendations"):
        recommended_foods = recommend_foods_for_protein_intake(protein_intake, diet_dataset)
        display_recommendations(recommended_foods)

# <----- Diet Tracker ---->
diet_2 = """
    <div id = 'blank_diet2'></div>
    <div id = 'black_bg_diet2'></div>
    <div class = 'parent_diet'>
        <div class = 'sub_head_diet'>
            Diet Tracker
        </div>
    </div>
"""
st.markdown(diet_2, unsafe_allow_html=True)

diet_total_calories = 0
diet_total_protein = 0
diet_total_carbohydrates = 0
diet_total_fat = 0
diet_total_fiber = 0
diet_total_sugar = 0
for meal_time in ['Breakfast', 'Lunch', 'Dinner']:
    st.subheader(f"{meal_time}")
    search_query = st.multiselect(f"Search for {meal_time} food item:", options=diet_dataset['Food Item'].unique(), key=f"{meal_time}_search")
    if search_query:
        for diet_food_item in search_query:
            selected_diet_food_data = diet_dataset[diet_dataset['Food Item'] == diet_food_item].iloc[0]
            diet_total_calories += selected_diet_food_data['Calories (kcal/100g)']
            diet_total_protein += selected_diet_food_data['Protein (g/100g)']
            diet_total_carbohydrates += selected_diet_food_data['Carbohydrates (g/100g)']
            diet_total_fat += selected_diet_food_data['Fat (g/100g)']
            diet_total_fiber += selected_diet_food_data['Dietary Fiber (g/100g)']
            diet_total_sugar += selected_diet_food_data['Sugar (g/100g)']

nutrition_info_div = f"""
    <div class="nutrition-info">
        <p class = 'p_diet_head'><strong>Total Nutrition for the Day :</strong></p>
        <p><strong>Total Calories: </strong>{diet_total_calories:.2f} kcal</p>
        <p><strong>Total Protein: </strong>{diet_total_protein:.2f} g</p>
        <p><strong>Total Carbohydrates: </strong>{diet_total_carbohydrates:.2f} g</p>
        <p><strong>Total Fat: </strong>{diet_total_fat:.2f} g</p>
        <p><strong>Total Dietary Fiber: </strong>{diet_total_fiber:.2f} g</p>
        <p><strong>Total Sugar: </strong>{diet_total_sugar:.2f} g</p>
    </div>
    """
st.markdown(nutrition_info_div, unsafe_allow_html=True)


diet_planner_css = """
<style>
    .st-emotion-cache-zt5igj {
        position: relative;
        top: -420px;
        left: -296px;
    }

    #blank_diet{
        height: 100px;
        width: 1%;
        position: relative;
        top: -437px;
    }

    #blank_diet2{
        height: 100px;
        width: 1%;
        position: relative;
        top: -150px;
    }

    #black_bg_diet{
        background-color: rgba(0, 0, 0, 0.4);
        width: 130%;
        height: 286px;
        position: absolute;
        top: -341px;
        left: -270px;
    }

    #black_bg_diet2{
        background-color: rgba(0, 0, 0, 0.4);
        width: 130%;
        height: 780px;
        position: absolute;
        top: -341px;
        left: -270px;
    }

    .row-widget.stRadio {
        position: relative;
        top: -423px;
        left: -249px;
    }

    .stNumberInput {
        position: relative;
        top: -436px;
        left: -250px;
    }

    .row-widget.stMultiSelect {
        position: relative;
        top: -430px;
        left: -249px;
    }

    .glideDataEditor.gdg-wmyidgi {
        position: relative;
        left: -269px;
        top: -415px;
    }

    .nutrition-info {
        position: absolute;
        top: -382px;
        left: -214px; 
    }

    .p_diet_head {
        font-size: 30px;
        text-decoration: underline;
        margin-bottom: 18px;
        position: relative;
        left: -31px;
    }

    .sub_head_diet{
        font-size: 30px;
        text-decoration: underline;
        position: relative;
        top: -430px;
        left: -248px;
    }

    .diet_tracker_photo{
        height: 125px;
        position: fixed;
        top: 407px;
        left: 1398px;
        border-radius: 25px;
        z-index: 1;
    }
</style>
"""
st.markdown(diet_planner_css, unsafe_allow_html=True)

# <----- About ---->
about = """
    <div id = 'blank5'></div>
    <div class = 'about_head'>About FIRCE</div>
    <div class = 'parent_5'>
        <div class = 'p5_child'>
            <div class = 'p5_child1_para'>
                <p class = 'p5_child_head'>Our Mission</p>
                At FIRCE, our mission is simple, to empower individuals to make healthier choices by providing them with the tools and information they need to understand and track their nutrition effectively. We believe that technology has the potential to transform the way we eat and live, and we're committed to leveraging it for the betterment of society.
            </div>
            <img class = 'p5_child1_img' src="https://tse3.mm.bing.net/th/id/OIG2.GMWouSU48g7nDM0DfDH1?pid=ImgGn" alt="image4">
        </div>
        <div class = 'p5_child'>
            <img class = 'p5_child2_img' src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTySMFsxHWwxixwqftYXdTeP8K_xiTQzI1xaw&s" alt="image4">
            <div class = 'p5_child2_para'>
                <p class = 'p5_child2_head'>Our Vision</p>
                We envision a world where everyone has access to accurate, reliable, and user-friendly nutrition information at their fingertips. Whether you're trying to manage your weight, improve your dietary habits, or simply make more informed food choices, we want to be your trusted companion on your journey towards better health and wellness.
            </div>
        </div>
        <div class = 'p5_child'>
            <div class = 'p5_child3_para'>
                <p class = 'p5_child_head'>Why Choose FIRCE?</p>
                <p class = 'points'>Accuracy: Our platform is powered by advanced machine learning algorithms and image recognition technology, ensuring accurate and reliable results with every analysis.</p>
                <p class = 'points'>Convenience: Say goodbye to manual food tracking and guesswork. With F.I.R.C.E, you can quickly and easily estimate the calorie content of your meals using just a simple image upload.</p>
            </div>
            <img class = 'p5_child3_img' src="data:image/jpeg;base64,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" alt="image4">
        </div>
        <div class = 'p5_child'>
            <img class ='p5_child4_img' src="https://media.gettyimages.com/id/857146092/photo/sea-of-hands.jpg?s=612x612&w=gi&k=20&c=eF96aafY7tRi6Vp8JpkYHs8C7tFwTqdKqJTQu7f7Zqs=" alt="image4">
            <div class = 'p5_child4_para'>
                <p class = 'p5_child2_head'>Join Our Community</p>
                Join the growing community of health-conscious individuals who have embraced FIRCE as their preferred tool for tracking nutrition and making healthier choices. Whether you're a fitness enthusiast, a nutrition professional, or someone looking to improve their diet, there's a place for you in our community.
            </div>
        </div>
        <div class = 'p5_child'>
            <div class = 'p5_child5_para'>
                <p class = 'p5_child_head'>Contact Us</p>
                Have questions, feedback, or suggestions? We'd love to hear from you! Get in touch with our friendly support team via <a href="https://sbbsuniversity.ac.in/">https://sbbsuniversity.ac.in/</a> and we'll be happy to assist you.
            </div>
            <img class = 'p5_child5_img' src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfENXouj0SEU5T_kBGMZ6p17biFr1SIP3IKg&s" alt="image4">
        </div>
    </div>
"""
st.markdown(about, unsafe_allow_html=True)

about_css = """
<style>
    .about_head{
        position: relative;
        top: -350px;
        left: -265px;
        font-size: 40px;
        text-decoration: underline;
    }

    #blank5{
        height: 276px;
        width: 1%;
        position: relative;
        top: -156px;
    }

    .p5_child{
        position: relative;
        top: -287px;
        left: -264px;
        display: grid;
        grid-template-columns: auto auto;
        margin-bottom: 30px;
    }

    .p5_child_head {
        font-size: 35px;
    }

    .p5_child1_img{
        height: 175px;
        width:  175px;
        position: relative;
        top: 18px;
        left: 180px;
        border-radius: 25px;
    }

    .p5_child1_para {
        width: 122%;
        text-align: justify;
        font-size: 18px;
    }

    .p5_child2_img{
        height: 175px;
        width:  175px;

        border-radius: 25px;
        position: relative;
        top: 36px;
    }

    .p5_child2_para {
        width: 122%;
        text-align: justify;
        position: relative;
        top: 24px;
        left: 71px;
        font-size: 18px;
    }

    .p5_child2_head {
        text-align: left;
        # font-size: x-large;
        font-size: 35px;

    }

    .p5_child3_para {
        width: 122%;
        text-align: justify;
        position: relative;
        top: 80px;
        font-size: 18px;
    }

    .points{
        font-size: 18px;
    }

    .p5_child3_img{
        height: 175px;
        width:  175px;
        border-radius: 25px;
        position: relative;
        top: 95px;
        left: 185px;
    }

    .p5_child4_img{
        height: 175px;
        width:  175px;
        border-radius: 25px;
        position: relative;
        top: 110px;
    }

    .p5_child4_para {
        width: 122%;
        text-align: justify;
        position: relative;
        top: 110px;
        left: 71px;
        font-size: 18px;
    }

    .p5_child5_para {
        width: 122%;
        text-align: justify;
        position: relative;
        top: 140px;
        font-size: 18px;
    }

    .p5_child5_img{
        height: 175px;
        width:  175px;
        border-radius: 25px;
        position: relative;
        top: 153px;
        left: 185px;
    }


</style>
"""
st.markdown(about_css, unsafe_allow_html=True)

# <----- FAQs ---->
faqs = """
    <div id = 'blank6'></div>
    <div class = 'faq_head'>FAQs</div>
    
    <details>
        <summary>How accurate is the calorie estimation provided by your platform?</summary>
        <p>Our platform utilizes advanced machine learning algorithms and a vast database of nutritional information to provide accurate calorie estimations. While we strive for precision, it's important to note that the accuracy may vary depending on factors such as image quality and food composition.</p>
    </details>

    <details>
        <summary>What types of foods can your platform recognize?</summary>
        <p>Our platform is capable of recognizing fruits & vegetables. However, there may be some limitations, particularly with complex dishes or obscure food items.</p>
    </details>

    <details>
        <summary>How do I ensure the best results when uploading images?</summary>
        <p>To achieve the most accurate results, we recommend uploading clear, well-lit images with minimal clutter. Ensure that the food items are fully visible and not obscured by other objects or packaging.</p>
    </details>

    <details>
        <summary>Can I access my previous analysis results?</summary>
        <p>At this time, we do not offer a feature to store or access previous analysis results. However, we are continuously working to enhance our platform and may consider implementing such functionality in the future.</p>
    </details>

    <details>
        <summary>How often is your nutritional database updated?</summary>
        <p>We regularly update our nutritional database to ensure that it remains current and accurate. However, the frequency of updates may vary depending on factors such as the availability of new data and changes in food composition.</p>
    </details>

    <details>
        <summary>Can your platform provide nutritional information beyond calorie estimation?</summary>
        <p>Yes, in addition to calorie estimation, our platform can provide information on macronutrient composition (such as protein, carbohydrates, and fat), vitamins, minerals, and more.</p>
    </details>

    <details>
        <summary>Is your platform suitable for individuals with specific dietary requirements or restrictions?</summary>
        <p>While our platform can provide general nutritional information for a wide range of foods, it may not be suitable for individuals with specific dietary requirements or restrictions. We recommend consulting with a healthcare professional or nutritionist for personalized dietary advice.</p>
    </details>

    <details class = 'last_detail'>
        <summary>How can I provide feedback or report issues with your platform?</summary>
        <p>We welcome feedback from our users and encourage you to reach out to our support team with any questions, suggestions, or concerns. You can contact us via hardeepsood903@gmail.com , and we'll be happy to assist you.</p>
    </details>
"""
st.markdown(faqs, unsafe_allow_html=True)

faqs_css = """
<style>

    .faq_head{
        position: relative;
        top: -168px;
        left: -265px;
        font-size: 35px;
        text-decoration: underline;
    }

    #blank6{
        height: 1px;
        width: 1%;
        position: relative;
        top: -250px;
    }

    details {
        margin-bottom: 20px;
        padding: 10px;
        background-color: rgba(0,0,0,0.4);
        position: relative;
        top: -130px;
        left: -265px;
    }

    .last_detail{
        margin-bottom: -266px;
    }

    summary {
        cursor: pointer;
        font-weight: bold;
    }

    summary::marker {
        content: "▸";
        color: #666;
        font-size: 14px;
        margin-right: 5px;
    }

    summary::-webkit-details-marker {
        display: none;
    }

    p {
        margin: 0;
    }

</style>
"""
st.markdown(faqs_css, unsafe_allow_html=True)

# <----- Contact ---->
contact = """
    <div id = 'blank7'></div>
    <footer>
        <img class = 'footer_img' src="data:image/jpeg;base64,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" alt="firce logo">
        <div class = 'address'>
            <div class = 'footer_head'>
                Address
            </div>
            <div class = 'footer_desc'>
                <p class = 'footer_desc'>VPO Rampur</p>
                <p class = 'footer_desc'>Distt. S.B.S.Nagar</p>
                <p class = 'footer_desc'>Punjab</p>
            </div>
        </div>
        <div class = 'contact'>
            <div class = 'footer_head'>
                Contact
            </div>
            <div class = 'footer_desc'>
                <p class = 'footer_desc'><strong>E-mail :<a></strong>hardeepsood903@gmail.com</a></p>
                <p class = 'footer_desc'><strong>Phone number :</strong>+91 81988 20997</p>
            </div>
        </div>
        <div class = 'footer_about'>
            <div class = 'footer_head'>
                About
            </div>
            <div class = 'footer_desc'>
                Our mission is simple, to empower individuals to make healthier choices by providing them with the tools and information they need to understand and track their nutrition effectively.
            </div>
        </div>
    </footer>
"""
st.markdown(contact, unsafe_allow_html=True)

contact_css = """
<style>
    footer{
        height: 162px;
        background-color: black;
        position: relative;
        left: -401px;
        bottom: -159px;
        width: 215%;
        display: grid;
        grid-template-columns: 19% auto auto 27%;
        padding-top: 20px;
    }

    .footer_img{
        height: 137px;
        position: relative;
        top: -6px;
        left: 15px;
        border-radius: 25px;
    }

    .footer_desc{
        font-size: 12px;
        color: burlywood;
    }

    #blank7{
        height: 1px;
        width: 1%;
        position: relative;
        top: -250px;
    }

    .footer_head {
        font-size: 30px;
    }

</style>
"""
st.markdown(contact_css, unsafe_allow_html=True)



