import matplotlib.pyplot as plt

def plot_image(X, title = 'No title'):
    # Reshape the image
    image = X.reshape(3,32,32)
    # Transpose the image
    image = image.transpose(1,2,0)
    # Display the image
    plt.imshow(image)
    plt.title(title)
    return