from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

import cv2
import numpy

class SimpleVGG():
    # Constructeur
    def __init__(self):
        # Appel du constructeur de la classe parente
        super().__init__()
        self.vgg = VGG16()
        

    # méthode qui rertaille une image numpy en 224x224, en conservant les proportions
    def resize_image(self, image):
        # Récupération des dimensions de l'image
        row, col, channel = image.shape
        # Calcul des nouvelles dimensions
        if row > col:
            new_row = 224
            new_col = int(col * 224 / row)
        else:
            new_col = 224
            new_row = int(row * 224 / col)

        # Retaillage de l'image
        image = cv2.resize(image, dsize=(new_col, new_row), interpolation=cv2.INTER_CUBIC)

        # Ajout de marges noires
        if row > col:
            delta = 224 - new_col
            left = int(delta / 2)
            right = delta - left
            image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            delta = 224 - new_row
            top = int(delta / 2)
            bottom = delta - top
            image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image

    # Méthode qui permet de prédire la classe d'une image
    # A partir d'une image passée en paramètre
    # L'image est d'abord retaillée pour correspondre à l'entrée de VGG
    # Puis elle est préparée pour correspondre aux entrées de VGG
    # Enfin, on prédit la classe de l'image
    # et l'on retourne les 5 classes les plus probables 
    def predict_image(self, image):

        if not isinstance(image, numpy.ndarray):
            raise Exception("L'image doit être un tableau numpy")

        # Retaillage de l'image
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = self.resize_image(image)
        
        # Préparation de l'image pour correspondre aux entrées de VGG
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        
        # Prédiction des probabilités de chaque classe
        output = self.vgg.predict(image, verbose=0)
        
        # traitement des sorties : classement et ajout de l'intitulé
        labels = decode_predictions(output, top=5)

        return labels
