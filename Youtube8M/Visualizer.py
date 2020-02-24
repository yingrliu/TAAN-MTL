"""
Author: Yingru Liu
Institute: Stony Brook University
"""

import sys
sys.path.append('..')

import argparse, os
import networkx as nx
import matplotlib.pyplot as plt
from Youtube8M.models import *
from sklearn.cluster import SpectralClustering
from Layers.TAL_pytorch import LossDis, LossCosine
from Youtube8M.torch_loader import MultiTask_Dataloader

parser = argparse.ArgumentParser(description='Evaluate TAAN by various splitting.')
# Model Setting of TAAN.
parser.add_argument('--hidden_feature', required=False, metavar='HIDDEN', default=1024, type=int,
                    help='the dimension of hidden output.')
parser.add_argument('--taan_constant', required=False, metavar='CONSTANT_T', default=1.0, type=float,
                    help='coefficient of TAAN regularization.')
parser.add_argument('--basis', required=False, metavar='BASIS', default=64, type=int,
                    help='number of basis functions.')
parser.add_argument('--path', required=False, metavar='PATH', default=None, type=str,
                    help='path of checkpoints.')
#
parser.add_argument('--gpu', required=False, metavar='GPU', default='0', type=str, help='ID of GPU.')

class Visualizer():
    def __init__(self, args):
        # model setting.
        self.args = args
        _, task_classes = MultiTask_Dataloader('train', 12, shuffle=True)
        self.model = TAAN(hidden_feature=args.hidden_feature, task_classes=task_classes,
                          basis=args.basis, c=args.taan_constant).cuda()
        # compute num of parameter.
        num_params = 0
        for param in self.model.parameters():
            size = list(param.size())
            num_params += np.prod(size)
        print('Model has %d parameters.' % num_params)
        return

    def compute_distance(self, path):
        # load the trained TAAN.
        print("Load trained model.")

        self.model.load_state_dict(torch.load(path))
        # compute the distance matrices.
        matrices = []
        # Layer 1.
        dis_layer1 = LossDis(self.model.layer1.alpha, self.model.layer1.basis)[1].detach().cpu().numpy()
        matrices.append(dis_layer1)
        # Layer 2.
        dis_layer2 = LossDis(self.model.layer2.alpha, self.model.layer2.basis)[1].detach().cpu().numpy()
        matrices.append(dis_layer2)
        # Layer 3.
        dis_layer3 = LossDis(self.model.layer3.alpha, self.model.layer3.basis)[1].detach().cpu().numpy()
        matrices.append(dis_layer3)
        return matrices

    def compute_cosine(self, path):
        # load the trained TAAN.
        print("Load trained model.")
        self.model.load_state_dict(torch.load(path))
        # compute the distance matrices.
        matrices = []
        # Layer 1.
        cos_layer1 = LossCosine(self.model.layer1.alpha, self.model.layer1.basis)[1].detach().cpu().numpy()
        matrices.append(cos_layer1)
        # Layer 2.
        cos_layer2 = LossCosine(self.model.layer2.alpha, self.model.layer2.basis)[1].detach().cpu().numpy()
        matrices.append(cos_layer2)
        # Layer 3.
        cos_layer3 = LossCosine(self.model.layer3.alpha, self.model.layer3.basis)[1].detach().cpu().numpy()
        matrices.append(cos_layer3)
        return matrices


if __name__ == "__main__":
    args = parser.parse_args()
    args.path = os.path.join(args.path, 'TAAN.pth')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    visualizer = Visualizer(args)
    matrices = visualizer.compute_distance(path=args.path)
    x_positions = np.arange(0, 16)
    x_labels = ['Arts & Entertainment', 'Games', 'Autos & Vehicles', 'Sports',
                  'Food & Drink', 'Computers & Electronics', 'Business & Industrial', 'Pets & Animals',
                  'Hobbies & Leisure', 'Beauty & Fitness', 'Shopping', 'Internet & Telecom',
                  'Home & Garden', 'Science', 'Travel', 'Law & Government']
    y_labels = ['1. Arts & Entertainment', '2. Games', '3. Autos & Vehicles',
                '4. Sports',
                '5. Food & Drink',
                '6. Computers & Electronics',
                '7. Business & Industrial',
                '8. Pets & Animals',
                '9. Hobbies & Leisure',
                '10. Beauty & Fitness',
                '11. Shopping',
                '12. Internet & Telecom',
                '13. Home & Garden',
                '14. Science',
                '15. Travel',
                '16. Law & Government']
    f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))
    axarr[0].imshow(matrices[0])
    axarr[0].set_xticks(x_positions)
    axarr[0].set_yticks(x_positions)
    axarr[0].set_xticklabels(x_labels, rotation=90)
    axarr[0].set_yticklabels(y_labels)
    axarr[0].set_title('Layer 1')
    axarr[1].imshow(matrices[1])
    axarr[1].set_xticks(x_positions)
    axarr[1].set_yticks(x_positions)
    axarr[1].set_xticklabels(x_labels, rotation=90)
    axarr[1].set_yticklabels(y_labels)
    axarr[1].set_title('Layer 2')
    axarr[2].imshow(matrices[2])
    axarr[2].set_xticks(x_positions)
    axarr[2].set_yticks(x_positions)
    axarr[2].set_xticklabels(x_labels, rotation=90)
    axarr[2].set_yticklabels(y_labels)
    axarr[2].set_title('Layer 3')
    plt.show()