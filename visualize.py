import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_loss_and_accuracy(hist_non_leaky, hist_leaky):
    n = len(hist_leaky['train']['loss'])

    loss_patch_train = mpatches.Patch(color='red', label='Training Loss')
    loss_patch_dev = mpatches.Patch(color='blue', label='Dev Loss')
    acc_patch_train = mpatches.Patch(color='red', label='Training Accuracy')
    acc_patch_dev = mpatches.Patch(color='blue', label='Dev Accuracy')

    fig = plt.figure(0, figsize=(10, 10))
    plt.suptitle('Train Statistics')

    ax1 = fig.add_subplot(221)
    ax1.set_title('Loss Leaky')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.plot(list(range(n)), hist_leaky['train']['loss'], color='red')
    ax1.plot(list(range(n)), hist_leaky['dev']['loss'], color='blue')
    ax1.legend(handles=[loss_patch_train, loss_patch_dev])

    ax2 = fig.add_subplot(222)
    ax2.set_title('Accuracy Leaky')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.plot(list(range(n)), hist_leaky['train']['acc'], color='red')
    ax2.plot(list(range(n)), hist_leaky['dev']['acc'], color='blue')
    ax2.legend(handles=[acc_patch_train, acc_patch_dev])

    ax3 = fig.add_subplot(223)
    ax3.set_title('Loss Non Leaky')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.plot(list(range(n)), hist_non_leaky['train']['loss'], color='red')
    ax3.plot(list(range(n)), hist_non_leaky['dev']['loss'], color='blue')
    ax3.legend(handles=[loss_patch_train, loss_patch_dev])

    ax4 = fig.add_subplot(224)
    ax4.set_title('Accuracy Non Leaky')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Accuracy')
    ax4.plot(list(range(n)), hist_non_leaky['train']['acc'], color='red')
    ax4.plot(list(range(n)), hist_non_leaky['dev']['acc'], color='blue')
    ax4.legend(handles=[acc_patch_train, acc_patch_dev])
    
    plt.tight_layout()
    plt.savefig('./images/result.png')
    plt.show()
