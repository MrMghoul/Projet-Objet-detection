# Projet combiné de détection d'objet avec YOLO et VGG

Ce projet utilise deux techniques puissantes de vision par ordinateur pour traiter et analyser des images en temps réel. D'une part, YOLO (You Only Look Once) pour la détection d'objets, et d'autre part, VGG (Visual Geometry Group) pour la classification d'objets dans des images.

## Dépendances

- **OpenCV**: Utilisé pour la capture vidéo, la lecture, le dessin sur les images et l'affichage.
- **simple_yolo**: Une implémentation simplifiée de YOLO.
- **simple_vgg**: Une implémentation simplifiée de VGG.

## Installation

Assurez-vous que toutes les dépendances sont installées. Vous pouvez les installer via pip ou conda en fonction de votre gestionnaire de paquets.

## Utilisation

Pour démarrer les différents scripts de ce projet, suivez ces étapes :

1. **Pour la détection d'objets avec YOLO**:
   - Exécutez le script `yolo-test.py`.

2. **Pour la classification d'images avec VGG**:
   - Exécutez le script `vgg-test.py`.

## Fonctionnement des scripts

### Détection d'objets avec YOLO (`yolo-test.py`)

- Importe les bibliothèques nécessaires.
- Définit une fonction `draw_boxes_in_image` qui dessine des boîtes englobantes et des étiquettes pour chaque objet détecté.
- Initialise YOLO avec la classe `StdYOLO` de `simple_yolo`.
- Initialise la capture vidéo à partir de la webcam.
- Dans une boucle infinie, capture des images, détecte les objets, dessine les boîtes et affiche l'image.

### Classification d'images avec VGG (`vgg-test.py`)

- Importe les bibliothèques nécessaires.
- Charge une image depuis le disque.
- Initialise VGG avec la classe `SimpleVGG` de `simple_vgg`.
- Utilise VGG pour prédire les probabilités des classes pour l'image.
- Affiche les 5 classes les plus probables avec leurs probabilités.
