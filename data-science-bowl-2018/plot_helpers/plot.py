import matplotlib.pyplot as plt

def imshow(img):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    img = ax.imshow(img)
    fig.colorbar(img)
