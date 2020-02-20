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
parser.add_argument('--rbf', required=False, metavar='RBF', default=0, type=int,
                    help='number of RBF kernel in GMM.')
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
                          basis=args.basis, c=args.taan_constant, regularization=args.regularize,
                          RBF=args.rbf).cuda()
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
        pi = self.model.layer1.rbf_pi if self.model.layer1.rbf_pi.size()[0] == 1 \
            else F.softmax(self.model.layer1.rbf_pi)
        scale = self.model.layer1.rbf_scale if self.model.layer1.rbf_pi.size()[0] == 1 \
            else F.softplus(self.model.layer1.rbf_scale)
        dis_layer1 = LossDis(self.model.layer1.alpha, self.model.layer1.basis, pi,
                                self.model.layer1.rbf_loc, scale)[1].detach().cpu().numpy()
        matrices.append(dis_layer1)
        # Layer 2.
        pi = self.model.layer2.rbf_pi if self.model.layer2.rbf_pi.size()[0] == 1 \
            else F.softmax(self.model.layer2.rbf_pi)
        scale = self.model.layer2.rbf_scale if self.model.layer2.rbf_pi.size()[0] == 1 \
            else F.softplus(self.model.layer2.rbf_scale)
        dis_layer2 = LossDis(self.model.layer2.alpha, self.model.layer2.basis, pi,
                             self.model.layer2.rbf_loc, scale)[1].detach().cpu().numpy()
        matrices.append(dis_layer2)
        # Layer 3.
        pi = self.model.layer3.rbf_pi if self.model.layer3.rbf_pi.size()[0] == 1 \
            else F.softmax(self.model.layer3.rbf_pi)
        scale = self.model.layer3.rbf_scale if self.model.layer3.rbf_pi.size()[0] == 1 \
            else F.softplus(self.model.layer3.rbf_scale)
        dis_layer3 = LossDis(self.model.layer3.alpha, self.model.layer3.basis, pi,
                             self.model.layer3.rbf_loc, scale)[1].detach().cpu().numpy()
        matrices.append(dis_layer3)
        return matrices

    def compute_cosine(self, path):
        # load the trained TAAN.
        print("Load trained model.")
        self.model.load_state_dict(torch.load(path))
        # compute the distance matrices.
        matrices = []
        # Layer 1.
        pi = self.model.layer1.rbf_pi if self.model.layer1.rbf_pi.size()[0] == 1 \
            else F.softmax(self.model.layer1.rbf_pi)
        scale = self.model.layer1.rbf_scale if self.model.layer1.rbf_pi.size()[0] == 1 \
            else F.softplus(self.model.layer1.rbf_scale)
        cos_layer1 = LossCosine(self.model.layer1.alpha, self.model.layer1.basis, pi,
                             self.model.layer1.rbf_loc, scale)[1].detach().cpu().numpy()
        matrices.append(cos_layer1)
        # Layer 2.
        pi = self.model.layer2.rbf_pi if self.model.layer2.rbf_pi.size()[0] == 1 \
            else F.softmax(self.model.layer2.rbf_pi)
        scale = self.model.layer2.rbf_scale if self.model.layer2.rbf_pi.size()[0] == 1 \
            else F.softplus(self.model.layer2.rbf_scale)
        cos_layer2 = LossCosine(self.model.layer2.alpha, self.model.layer2.basis, pi,
                             self.model.layer2.rbf_loc, scale)[1].detach().cpu().numpy()
        matrices.append(cos_layer2)
        # Layer 3.
        pi = self.model.layer3.rbf_pi if self.model.layer3.rbf_pi.size()[0] == 1 \
            else F.softmax(self.model.layer3.rbf_pi)
        scale = self.model.layer3.rbf_scale if self.model.layer3.rbf_pi.size()[0] == 1 \
            else F.softplus(self.model.layer3.rbf_scale)
        cos_layer3 = LossCosine(self.model.layer3.alpha, self.model.layer3.basis, pi,
                             self.model.layer3.rbf_loc, scale)[1].detach().cpu().numpy()
        matrices.append(cos_layer3)
        return matrices

    def display(self, matrices):
        # todo:
        G = nx.DiGraph()
        network_edges = []
        em_edges = [(11, 11+16), (11+16, 11+32), (12, 12+16), (12+16, 12+32)]
        em_edges_1 = [(14, 14 + 16), (14 + 16, 14 + 32), (15, 15 + 16), (15 + 16, 15 + 32)]
        color_map = [c/16 for c in range(1, 17)] * 3
        color_table = list(range(9))
        labels = {}
        for i in range(16*3):
            labels[i+1] = i % 16 + 1
        """Layer 1."""
        matrix = matrices[0]
        G.add_nodes_from(list(range(1, 17)))
        sc = SpectralClustering(3, affinity='precomputed', n_init=2000, assign_labels='discretize')
        sc.fit_predict(matrix)
        clusters = sc.labels_
        for i in range(4):
            idx = np.where(clusters == i)[0].tolist()
            for j in idx:
                color_map[j] = color_table[i]
        pos = nx.circular_layout(G)
        """Layer 2"""
        matrix = matrices[1]
        G.add_nodes_from(list(range(17, 33)))
        sc = SpectralClustering(3, affinity='precomputed', n_init=2000, assign_labels='discretize')
        sc.fit_predict(matrix)
        clusters = sc.labels_
        for i in range(4):
            idx = np.where(clusters == i)[0].tolist()
            for j in idx:
                color_map[j+16] = color_table[i+3]
        for i in range(16):
            G.add_edge(i+1, i+17)
            network_edges.append((i+1, i+17))
            pos[i+1+16] = pos[i+1] + [0.3, 4]
        """Layer 3"""
        matrix = matrices[2]
        G.add_nodes_from(list(range(33, 33+16)))
        sc = SpectralClustering(3, affinity='precomputed', n_init=2000, assign_labels='discretize')
        sc.fit_predict(matrix)
        clusters = sc.labels_
        for i in range(4):
            idx = np.where(clusters == i)[0].tolist()
            for j in idx:
                color_map[j + 32] = color_table[i+6]
        for i in range(16):
            G.add_edge(i+17, i+33)
            network_edges.append((i+17, i+33))
            pos[i+33] = pos[i+17] + [-0.3, 4]
        #
        """draw graph."""
        nx.draw_networkx_nodes(G, pos, node_color=color_map, cmap=plt.cm.tab10)
        nx.draw_networkx_edges(G, pos, edgelist=network_edges, width=2, alpha=1.0)
        nx.draw_networkx_edges(G, pos, edgelist=em_edges, width=8, alpha=0.5, edge_color='b')
        nx.draw_networkx_edges(G, pos, edgelist=em_edges_1, width=8, alpha=0.5, edge_color='r')
        nx.draw_networkx_labels(G, pos, labels, font_size=12)
        plt.show()
        return
"""
# add network edge.
        for m in range(3):
            if m > 0:
                for i in range(16):
                    G.add_edges_from([(i+1+16*(m-1), i+1+16*m)])
                    network_list.append((i+1+16*(m-1), i+1+16*m))
"""

if __name__ == "__main__":
    args = parser.parse_args()
    args.regularize = 'distance'
    args.path = os.path.join('checkpoints', 'TAAN_64_cosine', 'TAAN.pth')
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
    #visualizer.display(matrices)