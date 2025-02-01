import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from skimage import transform

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Background color */
    .stApp {
        background-color: #1a1a2e;  /* Dark navy blue */
        color: white;
        font-family: Verdana, sans-serif;
    }
    /* Header style */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4cc9f0;  /* Light blue */
        font-family: Verdana, sans-serif;
    }
    /* File uploader style */
    .stFileUploader label {
        color: #ffffff;  /* White text */
        font-family: Verdana, sans-serif;
    }
    /* Buttons and sliders */
    .stButton button, .stSlider, .stSelectbox {
        background-color: #ffffff;  /* White */
        color: #1a1a2e;  /* Dark navy blue text */
        font-family: Verdana, sans-serif;
    }
    /* Prediction text style */
    .stMarkdown p {
        color: #4cc9f0;  /* Light blue */
        font-family: Verdana, sans-serif;
    }
    /* Column headers */
    .stMarkdown h2 {
        color: #00f5d4;  /* Blue-green */
        font-family: Verdana, sans-serif;
    }
    /* Frame around the application */
    .main-container {
        padding: 20px;
        border-radius: 15px;
        background-color: #16213e;  /* Slightly darker navy */
        font-family: Verdana, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def preprocess_image(image):
    image = image.resize((28, 28))
    image = image.convert('L')
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32') / 255.0
    return image

def predict(image):
    processed_image = preprocess_image(image)
    prediction = np.argmax(model.predict(processed_image), axis=1)[0]
    return prediction

def to_2d(sub_v, augment=False):
  w,h,z = sub_v.shape
  img = []
  for s in range(10, 90, 5):
    img.append(sub_v[:,:,s])
  arr = np.zeros((w*4, h*4))
  for i, s in enumerate(img):
    if augment:
      s= transform.rotate(s, np.random.uniform(-360, 360))
    arr[i//4*w:i//4*w+w, i%4*h:i%4*h+h] = s
  return arr

def create_frame(axis, index):
    if axis == 'x':
        data = volume[index, :, :]
        x_label, y_label = 'Y Axis', 'Z Axis'
    elif axis == 'y':
        data = volume[:, index, :]
        x_label, y_label = 'X Axis', 'Z Axis'
    elif axis == 'z':
        data = volume[:, :, index]
        x_label, y_label = 'X Axis', 'Y Axis'
    return go.Heatmap(z=data, colorscale='Viridis'), x_label, y_label

st.title('PETAI')
st.write("An AI-powered Alzheimer's disease detection web-based tool, based on FDG PET brain scans")
st.markdown('<p style="color: #ffffff; font-family: Verdana, sans-serif;">(Upload an FDG PET brain scan)</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Select an image...", type="npy")

if uploaded_file is not None:

    f = uploaded_file.name.split("3d")[0]
    im1 = np.load(uploaded_file)
    hm = Image.open(f"{f}.jpg")

    volume = np.swapaxes(im1, 0,2)
    volume = np.swapaxes(volume, 0,1)

    initial_slice = 50

    heatmap, x_label, y_label = create_frame('z', initial_slice)
    fig = go.Figure(data=heatmap)

    frames = [go.Frame(data=[create_frame('z', i)[0]], name=str(i)) for i in range(volume.shape[2])]

    fig.update(frames=frames)

    sliders = [dict(
        active=initial_slice,
        currentvalue={"prefix": "Slice: "},
        pad={"t": 50},
        steps=[dict(
            label=str(i),
            method='animate',
            args=[[str(i)], dict(mode='immediate', frame=dict(duration=300, redraw=True), transition=dict(duration=0))]
        ) for i in range(volume.shape[2])]
    )]

    fig.update_layout(
        sliders=sliders,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=520,
        height=520,
    )

    col1, spacer, col2 = st.columns([6,1,6])

    with col1:
            st.header("3D Illustration of an Uploaded Brain Scan File")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
            st.header("Prediction")
            st.write("with highlighted important brain features")
            st.image(hm, width=450)

    st.write("")

    label = "unknown"

    if f.split("_")[1]=="CN":
       label = "Healthy brain"
    elif f.split("_")[1]=="MCI":
       label = "Mild cognitive impaired suspect"
    elif f.split("_")[1]=="AD":
       label = "Dementia suspect"
    with col2:
        st.write(f'Prediction : {label}')

st.markdown('</div>', unsafe_allow_html=True)
