"""
Author: Yingru Liu
Institute: Stony Brook University
This file contains the visualization of the alpha matrices of all the saved models.
"""

from TAL_Layers import Loss2Euclidean, LossCosine, LossTraceNorm
from TAAN import TAAN
import os
import matplotlib.pyplot as plt

Categories = ['Cosine', 'Euclidean', 'LowRank', 'NoVMTL']
model = TAAN(Lambda=0.0, dimBasis=32, lr=1e-5)

if __name__ == '__main__':
    if not os.path.exists('Images2N'):
        os.mkdir('Images2N')
    for VMTL in Categories:
        for k in range(10, 45, 1):
            savedModelPath = os.path.join(VMTL, 'TAAN-Lambda-{:.2f}'.format(0.01 * k))
            if not os.path.exists(savedModelPath):
                continue
            print(savedModelPath)
            model.loadModel(os.path.join(savedModelPath, 'model_params.pt'))
            Euclidean = [Loss2Euclidean(model.alphas[i], model.b[i])[1].data.cpu().numpy()
                         for i in range(len(model.alphas))]
            f, axarr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(20, 13))
            #
            axarr[0, 0].imshow(Euclidean[0])
            axarr[0, 0].set_title('conv1', fontsize=30)
            axarr[0, 0].axis('off')
            #
            axarr[0, 1].imshow(Euclidean[1])
            axarr[0, 1].set_title('conv2', fontsize=30)
            axarr[0, 1].axis('off')
            #
            axarr[0, 2].imshow(Euclidean[2])
            axarr[0, 2].set_title('conv3', fontsize=30)
            axarr[0, 2].axis('off')
            #
            axarr[1, 0].imshow(Euclidean[3])
            axarr[1, 0].set_title('conv4', fontsize=30)
            axarr[1, 0].axis('off')
            #
            axarr[1, 1].imshow(Euclidean[4])
            axarr[1, 1].set_title('dense1', fontsize=30)
            axarr[1, 1].axis('off')
            #
            axarr[1, 2].axis('off')
            plt.close()
            savedPath = os.path.join('Images2N', VMTL + '-TAAN-Lambda-{:.2f}.pdf'.format(0.01 * k))
            f.savefig(savedPath, bbox_inches='tight')
            pass
