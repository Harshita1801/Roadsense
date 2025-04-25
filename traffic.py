# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:04:59 2024

@author: harsh
"""

import os
os.chdir("C:/Users/harsh/Downloads/archive (2)")
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import base64

# Label Overview
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

classes_descriptions = {
    'Speed limit (20km/h)': "Indicates the maximum speed limit of 20 kilometers per hour. Typically found in school zones or residential areas.",
    'Speed limit (30km/h)': "Indicates the maximum speed limit of 30 kilometers per hour. Common in urban areas with heavy pedestrian traffic.",
    'Speed limit (50km/h)': "Indicates the maximum speed limit of 50 kilometers per hour. Standard limit in many urban and suburban roads.",
    'Speed limit (60km/h)': "Indicates the maximum speed limit of 60 kilometers per hour. Often used on main roads within cities and towns.",
    'Speed limit (70km/h)': "Indicates the maximum speed limit of 70 kilometers per hour. Found on some rural and suburban roads.",
    'Speed limit (80km/h)': "Indicates the maximum speed limit of 80 kilometers per hour. Common on rural roads and minor highways.",
    'End of speed limit (80km/h)': "Signifies the end of the 80 kilometers per hour speed limit zone.",
    'Speed limit (100km/h)': "Indicates the maximum speed limit of 100 kilometers per hour. Typically found on major highways and rural roads.",
    'Speed limit (120km/h)': "Indicates the maximum speed limit of 120 kilometers per hour. Common on motorways and high-speed roads.",
    'No passing': "Prohibits overtaking and passing other vehicles in the designated zone.",
    'No passing veh over 3.5 tons': "Restricts vehicles over 3.5 tons from overtaking and passing other vehicles.",
    'Right-of-way at intersection': "Indicates that vehicles must yield to other traffic at the intersection.",
    'Priority road': "Marks the beginning of a road where traffic has priority over intersecting roads.",
    'Yield': "Requires drivers to slow down and yield the right-of-way to other traffic at the intersection.",
    'Stop': "Requires drivers to come to a complete stop and proceed only when safe.",
    'No vehicles': "Prohibits all motor vehicles from entering the road or area.",
    'Veh > 3.5 tons prohibited': "Restricts vehicles over 3.5 tons from entering the road or area.",
    'No entry': "Indicates that entry is prohibited in the direction of the sign.",
    'General caution': "Warns drivers of potential hazards or changes in driving conditions ahead.",
    'Dangerous curve left': "Alerts drivers to a sharp left curve ahead.",
    'Dangerous curve right': "Alerts drivers to a sharp right curve ahead.",
    'Double curve': "Warns of successive curves, first to the left and then to the right (or vice versa).",
    'Bumpy road': "Indicates an uneven or rough road surface ahead.",
    'Slippery road': "Warns of a road surface that may be slippery when wet or icy.",
    'Road narrows on the right': "Indicates that the road will narrow on the right side ahead.",
    'Road work': "Alerts drivers to road construction or maintenance work ahead.",
    'Traffic signals': "Warns of upcoming traffic signals where drivers must be prepared to stop.",
    'Pedestrians': "Alerts drivers to a pedestrian crossing where they should slow down and yield.",
    'Children crossing': "Warns of a crossing area frequented by children, such as near schools.",
    'Bicycles crossing': "Alerts drivers to a bicycle crossing where cyclists may be present.",
    'Beware of ice/snow': "Warns of potential icy or snowy road conditions ahead.",
    'Wild animals crossing': "Alerts drivers to areas where wild animals may cross the road.",
    'End speed + passing limits': "Indicates the end of all previously imposed speed and passing restrictions.",
    'Turn right ahead': "Informs drivers that they must turn right at the next intersection.",
    'Turn left ahead': "Informs drivers that they must turn left at the next intersection.",
    'Ahead only': "Indicates that traffic is only allowed to proceed straight ahead.",
    'Go straight or right': "Informs drivers that they may continue straight or turn right at the intersection.",
    'Go straight or left': "Informs drivers that they may continue straight or turn left at the intersection.",
    'Keep right': "Directs traffic to stay or move to the right side of the road or lane.",
    'Keep left': "Directs traffic to stay or move to the left side of the road or lane.",
    'Roundabout mandatory': "Indicates that all traffic must circulate around the roundabout.",
    'End of no passing': "Indicates the end of a no-passing zone for all vehicles.",
    'End no passing veh > 3.5 tons': "Indicates the end of a no-passing zone for vehicles over 3.5 tons.",
    'not identified': "Sign is not identifed "
}


descriptions = [
    " Please reduce your speed to comply with this limit.",
    "Ensure you are driving within this speed for safety.",
    "Adjust your speed accordingly to avoid penalties.",
    "Maintain this speed to adhere to traffic regulations.",
    "Drive within this limit to ensure road safety.",
    "Follow this limit to avoid fines and ensure safety.",
    "You can now adjust your speed accordingly.",
    "Maintain this speed for a safe driving experience.",
    "Ensure your speed does not exceed this limit.",
    "Remain in your lane and do not overtake other vehicles.",
    "Heavier vehicles must stay in their lane.",
    "You have the priority at this crossing.",
    "You have the right of way; other vehicles must yield.",
    "Prepare to give way to other vehicles at the intersection.",
    "Come to a complete stop and proceed only when it is safe.",
    "This road is closed to all vehicle traffic.",
    "Heavier vehicles are not permitted.",
    "Do not enter this road; it is closed to traffic.",
    "Be alert for potential hazards on the road.",
    "Slow down and navigate carefully.",
    "Reduce speed and proceed with caution.",
    "Prepare for two consecutive curves on the road.",
    "Drive slowly to avoid damage to your vehicle.",
    "Reduce speed and drive carefully to avoid skidding.",
    "Be cautious as the road width decreases.",
    "Slow down and follow any temporary signs or signals.",
    "Be prepared to stop or follow signal instructions.",
    "Slow down and be ready to stop for pedestrians.",
    "Drive carefully and be prepared to stop for children.",
    "Watch for cyclists and give them the right of way.",
    "Drive cautiously to avoid slipping or skidding.",
    "Slow down and watch for animals on the road.",
    "Resume normal driving but stay cautious.",
    "Prepare to make a right turn as indicated.",
    "Get ready to make a left turn at the upcoming junction.",
    "Continue straight; do not turn at this point.",
    "You may continue straight or make a right turn.",
    "You can proceed straight or make a left turn.",
    "Stay to the right side of the road.",
    "Remain on the left side of the road.",
    "Enter the roundabout and follow its direction.",
    "Overtaking is now permitted where safe.",
    "Heavier vehicles may overtake."
]




model = load_model('model_vgg.h5')

model.make_predict_function()

def predict_label(img_name):
    image1 = cv2.imread(img_name)

    image_fromarray = Image.fromarray(image1, 'RGB')
    resize_image = image_fromarray.resize((50, 50))
    expand_input = np.expand_dims(resize_image,axis=0)
    input_data = np.array(expand_input)
    input_data = input_data/255

    
    pred = model.predict(input_data)
    result = pred.argmax()
    # print(result,pred[0][result])
    if( pred[0][result]>0.85):
        return (classes[result],descriptions[result])
    else:
        return ("not identified"," ")
  

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_image(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
    }}
    h1 {{
        color: #000000; /* Change this to any other color if needed */
    }}
    h2 {{
        color: #000000; /* Change this to any other color if needed */
    }}
    h3 {{
        color: #000000; /* Change this to any other color if needed */
    }}
    .stFileUploader label {{
        color: #FF6347 !important; /* Light red color for the file uploader */
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
  
def main():
    
    # Set the background image
    set_background_image('imageback.jpg')
    
    
    # giving a title
    #st.title('RoadSense: Traffic sign recognition')
    st.markdown("<h1 style='text-align: center;'>RoadSense: Traffic Sign Recognition</h1>", unsafe_allow_html=True)

    
    st.markdown(
        """
        <style>
        .file_uploader_label {
            color: #FF6347 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Open and display the image
        imag = Image.open(uploaded_file)
        st.image(imag, caption='Uploaded Image.', use_column_width=True)
       
        # code for Prediction
        predicted = ''
        desc = ''
       
        # creating a button for Prediction
        if st.button('Recognize Traffic Sign'):
            predicted,desc=predict_label(uploaded_file.name)
        
        #st.success(predicted)
        st.markdown(f"<h2 style='text-align: center;'>{predicted}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{desc}<br><br>({classes_descriptions[predicted]})</h3>", unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()
    