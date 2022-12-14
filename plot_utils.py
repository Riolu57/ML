import numpy as np
import matplotlib.pyplot as plt

def plot_loss(history):
    # list all data in history
    # summarize history for accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_predictions(preds, actual, horizon):
    preds = np.array(preds).reshape((horizon, 12))
    plt.plot(preds[:, 0], preds[:, 1], '-+', color='blue', label='prediction')
    plt.plot(actual[:, 0], actual[:, 1], '-o', color='blue', label='actual', alpha=0.5)

    plt.plot(preds[:, 4], preds[:, 5], '-+', color='green', label='prediction')
    plt.plot(actual[:, 4], actual[:, 5], '-o', color='green', label='actual', alpha=0.5)

    plt.plot(preds[:, 8], preds[:, 9], '-+', color='orange', label='prediction')
    plt.plot(actual[:, 8], actual[:, 9], '-o', color='orange', label='actual', alpha=0.5)
    plt.legend()
    plt.show()

