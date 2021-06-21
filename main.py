import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from tensorflow.keras.models import model_from_json
from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict(path):
    model = load_model("tomato_final.h5")

    # testfol = os.listdir("E:\Downloads\Testing")
    #
    # for i in testfol:
    #     print(i)
    # print(len(testfol))

    # image_path = path + '*.jpg'
    image_path = 'E:\\Downloads\\Testing\\' + path

    def imprepro(img):
        img = cv2.resize(img, (56, 56))
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        img = img.reshape(1, 56, 56, 1)

        return img

    def prediction(img):
        img = imprepro(img)
        ans = model.predict_classes(img)
        li = ['Tomato___Late_blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Target_Spot',
              'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Septoria_leaf_spot', 'Tomato___healthy',
              'Tomato___Early_blight', 'Tomato___Bacterial_spot', 'Tomato___Leaf_Mold', 'Tomato___Tomato_mosaic_virus']

        return li[int(ans)]

    basepath = "E:\Downloads\Testing"
    pl = 1
    # for j in os.listdir(basepath):
    #     print(pl)
    #     plt.figure()
    #     plt.subplot(len(os.listdir(basepath)), len(os.listdir(basepath)), pl)
    #     pl = pl + 1
    #     impath = basepath + "/" + j
    #     img = cv2.imread(impath)
    #     img = np.array(img)
    #
    #     plt.imshow(img)
    #     print(prediction(img))

    img = cv2.imread(image_path)
    img = np.array(img)
    # plt.imshow(img)
    print(prediction(img))
    ans = prediction(img)
    return ans
