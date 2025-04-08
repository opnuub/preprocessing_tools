from glob import glob
import matplotlib.pyplot as plt
import os

ROOT_DIR = os.getcwd()
true_dir, pred_dir = os.path.join(ROOT_DIR, 'true'), os.path.join(ROOT_DIR, 'predict')


def vis_image(path_to_image, pred_image):
    # load image
    image = plt.imread(path_to_image)
    pred_image = plt.imread(pred_image)
    height, width, channels = image.shape
    # create the subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*width/150, height/150))
    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.gca().invert_yaxis()
#    ax.set_axis_off()
    # display the image on both subplots
    ax1.imshow(image)
    ax1.set_title('Ground Truth', fontsize=14)
    ax2.imshow(pred_image)
    ax2.set_title('Prediction', fontsize=14)
    os.makedirs('comparison')
    fig.savefig(os.path.join(ROOT_DIR, 'comparison', f'{path_to_image[5:]}'))
    return fig, ax1, ax2

if __name__ == '__main__':
    files = glob(pred_dir + "\\*.jpg")
    files = [i.replace("\\", "/").split("/")[-1].split(".jpg")[0] for i in files]

    for file in files: 
        if os.path.exists(os.path.join(ROOT_DIR, 'true', f'{file}_masked.jpg')):
            vis_image(f'true/{file}_masked.jpg', os.path.join(ROOT_DIR, 'predict', f'{file}.jpg'))
        else:
            vis_image(f'true/{file}.jpg', os.path.join(ROOT_DIR, 'predict', f'{file}.jpg'))
    