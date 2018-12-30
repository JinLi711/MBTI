import matplotlib.pyplot as plt

def smooth_curve(points, factor=0.8):
    """
    Smooth out a list of points

    :param points: list of y coordinate points
    :type  points: list
    :param factor: smoothness factor
    :type  factor: float
    :return: Smoothed out elements of a list
    :rtype:  factor: list
    """

    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_loss_acc (history, plot_type='loss'):
    """
    Plot either the loss or accuracy.
    
    :param history: The history of the training from Keras
    :type  history: keras.callbacks.History
    :param plot_type: either loss or acc
    :type  plot_type: str
    """
    
    history_dict = history.history
    for name, values in history_dict.items():
        # the name will look something like: val_IE_loss
        split = name.split('_')
        if (len (split) == 3) and (split[2] == plot_type):
            epochs = range(len(values))
            
            train_name = name[4:] # something like IE_loss
            train_values = history_dict[train_name]
            plt.figure()
            
            # the values are validation values
            plt.plot(epochs, smooth_curve (values), 'b', label=name)
            plt.plot(epochs, smooth_curve (train_values), 'bo', label=train_name)
            plt.title(split[1]) # something like IE
            plt.legend()
    plt.show()


def conf_mx_rates (y, y_pred):
    """
    Given labels and predictions, creates a confusion matrix of error rates.
    Each row is an actual class, while each column is a predicted class.
    The whiter the square, the more the image is misclassified

    :param y: The labels
    :type  y: pandas.core.series.Series
    :param y_pred: The predictions based on the ML algorithm.
    :type  y_pred: pandas.core.series.Series
    """
    
    from sklearn.metrics import confusion_matrix
    
    conf_mx = confusion_matrix (y, y_pred)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()