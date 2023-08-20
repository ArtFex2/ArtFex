import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import tensorflow as tf
import requests
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def main():
    st.set_page_config(page_title="Artfex", page_icon="🖖", layout="wide")
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style/style.css")


    st.title('AI Image Detection')
    st.write('Upload any image that you think fits into one of the classes and see if the prediction is correct.')

    file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
    if file:
        left_column, right_column= st.columns(2)
        with left_column:
            image = Image.open(file)
            st.image(image, width= 700)

            # Resize the image to (32, 32) pixels
            resized_image = image.resize((32, 32))

            img_array = np.array(resized_image) / 255.0

            # Make sure the img_array has shape (32, 32, 3)
            if img_array.shape != (32, 32, 3):
                st.text('Error: Invalid image shape. Please ensure the uploaded image is in RGB format and has dimensions (32, 32).')
                return

            img_array = img_array.reshape((1, 32, 32, 3))

            model = tf.keras.models.load_model('improved_cifar10_model.h5')

            predictions = model.predict(img_array)
            cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        with right_column:
            lottie_coding4 = load_lottieurl("https://lottie.host/6137d1db-b72d-4a75-8c89-582da835f2e8/g7XYjR39bS.json")
            st_lottie(lottie_coding4, height=620, key='coding3')


        left_column, right_column= st.columns(2)
        with right_column:
            fig, ax = plt.subplots()
            y_pos = np.arange(len(cifar10_classes))
            ax.barh(y_pos, predictions[0], align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cifar10_classes)
            ax.invert_yaxis()
            ax.set_xlabel('Probability')
            ax.set_title('Predictions')

            st.pyplot(fig)
        with left_column:
            lottie_coding3 = load_lottieurl("https://lottie.host/623fd5ac-7cbb-4edc-a9c2-7659b8bde2aa/JRNFRczBqR.json")
            st_lottie(lottie_coding3, height=580, key='coding')
    else:
        st.text('You have not uploaded an image yet!')





if __name__ == '__main__':
    main()

lottie_coding2 = load_lottieurl("https://lottie.host/2658fd65-387a-4ed0-ac76-d55dcff90ef8/sCuBfhJE9u.json")
with st.container():
        st.write("---")
        st.header("Get in Touch With Us!")
        st.write("##")

        conatct_form = """
        <form action="https://formsubmit.co/artfex.info@gmail.com" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder="Your name" required>
         <input type="email" name="email" placeholder="Your email" required>
         <textarea name="massage" placeholder="Your message here" required></textarea>
         <button type="submit">Send</button>
        </form>
        """



        left_column, right_column= st.columns(2)

        with left_column:
            st.markdown(conatct_form, unsafe_allow_html=True)
        with right_column:
            st_lottie(lottie_coding2, height=300, key='coding5')
