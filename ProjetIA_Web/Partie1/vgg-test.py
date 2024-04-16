import cv2
from simple_vgg import SimpleVGG

# Chargement de l'image
image = cv2.imread('Air-France-Avion-A350-900.jpg')
print('**************************************************************')
print(type(image))

# Chargement du modèle VGG
model = SimpleVGG()

# Prédiction des probabilités de chaque classe
vgg_labels = model.predict_image(image)

# affichage des 5 premières classes
for label in vgg_labels[0]:
    print('%s (%.2f%%)' % (label[1], label[2]*100))

# Affichage de l'image
cv2.imshow("test",image)

# Attente d'une touche avant de fermer la fenêtre
cv2.waitKey()
