import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image
import tensorflow as tf


def main():
    st.title('AI')
    st.write('Upload any image that you think fits into one of the classes and see if the prediction is correct.')

    file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

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

        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_classes))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_classes)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('Predictions')

        st.pyplot(fig)

    else:
        st.text('You have not uploaded an image yet!')


if __name__ == '__main__':
    main()
