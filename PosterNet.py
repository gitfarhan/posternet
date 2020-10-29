from tensorflow.keras.models import load_model
import numpy as np
import cv2
import click
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = load_model('poster.model')
le = pickle.loads(open('poster.pickle', "rb").read())

def predict(poster_name):
    poster = cv2.imread(poster_name)
    poster = cv2.resize(poster, (32, 32))
    poster = poster.astype("float") / 255.0
    poster = img_to_array(poster)
    poster = np.expand_dims(poster, axis=0)
    preds = model.predict(poster)[0]
    j = np.argmax(preds)
    label = le.classes_[j]

    result = f"{poster_name} predicted genre: {label}"
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('PosterNet')
    img = mpimg.imread(poster_name)
    plt.imshow(img)
    plt.title(f"genre: {label}")
    plt.show()

    print(result)

@click.command()
@click.option('--postername', '-p')
def main(postername):
    click.echo(predict(poster_name=postername))

if __name__ == '__main__':
    main()