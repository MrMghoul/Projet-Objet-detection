import cv2
import time

cap = cv2.VideoCapture(0)
window_name = "FPS Loop"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

start_time = time.time()
frames = 0

# limitation de la durée de la capture
seconds_to_measure = 2

# boucle de capture
while start_time + seconds_to_measure > time.time():
    success, img = cap.read()

    # ajout d'un délai pour simuler un traitement
    time.sleep(0.1)  

    # affichage de l'image
    cv2.imshow(window_name, img)
    cv2.waitKey(1)

    frames = frames + 1

cv2.destroyAllWindows()

# affichage de la fréquence d'images capturées (FPS pour Frame Per Second)
print(
    f"{frames} images captuées en {seconds_to_measure} secondes. FPS: {frames/seconds_to_measure}"
)
