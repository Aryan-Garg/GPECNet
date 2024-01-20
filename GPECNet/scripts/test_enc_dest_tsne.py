import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse
import copy
sys.path.append("../utils/")
import matplotlib.pyplot as plt
import numpy as np
from models import *
from social_utils import *
import yaml
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--load_file', '-lf', default="run7.pt")
parser.add_argument('--num_trajectories', '-nt', default=20) #number of trajectories to sample
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--root_path', '-rp', default="./")

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)


checkpoint = torch.load('../saved_models/{}'.format(args.load_file), map_location=device)
hyper_params = checkpoint["hyper_params"]

print(hyper_params)

def test(test_dataset):

#    model.eval()
#    assert best_of_n >= 1 and type(best_of_n) == int
#    test_loss = 0

    with torch.no_grad():
        for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
            traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
            x = traj[:, :hyper_params["past_length"], :]
#            print(x.shape)   #([2829, 8, 2])
            y = traj[:, hyper_params["past_length"]:, :]
            y = y.cpu().numpy()
            # reshape the data
            x = x.contiguous().view(-1, x.shape[1]*x.shape[2])
#            print(x.shape)  #[2829, 16]
            x = x.to(device)
            future = y[:, :-1, :]
            dest = y[:, -1, :]
##            print(dest.shape)     #(2829, 2)
##            print(initial_pos.shape)  #[2829, 2]
##            print(mask.shape)  #[2829, 2829]
#            all_l2_errors_dest = []
#            all_guesses = []
#            for index in range(best_of_n):
#
#                dest_recon = model.forward(x, initial_pos, device=device)
#                dest_recon = dest_recon.cpu().numpy()
#                all_guesses.append(dest_recon)
#
#                l2error_sample = np.linalg.norm(dest_recon - dest, axis = 1)
#                all_l2_errors_dest.append(l2error_sample)
#
#            all_l2_errors_dest = np.array(all_l2_errors_dest)
#            all_guesses = np.array(all_guesses)
#            # average error
#            l2error_avg_dest = np.mean(all_l2_errors_dest)
#
#            # choosing the best guess
#            indices = np.argmin(all_l2_errors_dest, axis = 0)
#
#            best_guess_dest = all_guesses[indices,np.arange(x.shape[0]),  :]
#
#            # taking the minimum error out of all guess
#            l2error_dest = np.mean(np.min(all_l2_errors_dest, axis = 0))
#
#            # back to torch land
#            best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)
#
#            # using the best guess for interpolation
#            interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
#            interpolated_future = interpolated_future.cpu().numpy()
#            best_guess_dest = best_guess_dest.cpu().numpy()
#
#            # final overall prediction
#            predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)
#            predicted_future = np.reshape(predicted_future, (-1, hyper_params["future_length"], 2))
#
#            # ADE error
#            l2error_overall = np.mean(np.linalg.norm(y - predicted_future, axis = 2))
#
#            l2error_overall /= hyper_params["data_scale"]
#            l2error_dest /= hyper_params["data_scale"]
#            l2error_avg_dest /= hyper_params["data_scale"]
#
#            print('Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(l2error_dest, l2error_avg_dest))
#            print('Test time error overall (ADE) best: {:0.3f}'.format(l2error_overall))

    return dest

def fit(X):
    n_samples = X.shape[0]
    
    # Compute euclidean distance
    distances = pairwise_distances(X, metric='euclidean', squared=True)
    
    # Compute joint probabilities p_ij from distances.
    perplexity = 30
    P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)
    n_components = 2
    # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)
    
    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(n_components - 1, 1)
    
    return _tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded)

def _tsne(P, degrees_of_freedom, n_samples, X_embedded):
    params = X_embedded.ravel()
    obj_func = _kl_divergence
    n_components = 2
    params = _gradient_descent(obj_func, params, [P, degrees_of_freedom, n_samples, n_components])   
    X_embedded = params.reshape(n_samples, n_components)
    return X_embedded

def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components):
    X_embedded = params.reshape(n_samples, n_components)
    
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    MACHINE_EPSILON = np.finfo(np.double).eps
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    
    # Kullback-Leibler divergence of P and Q
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    
    # Gradient: dC/dY
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    return kl_divergence, grad


def _gradient_descent(obj_func, p0, args, it=0, n_iter=10,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7):
    
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it
    
    for i in range(it, n_iter):
        error, grad = obj_func(p, *args)
        grad_norm = linalg.norm(grad)
        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update
        print("[t-SNE] Iteration %d: error = %.7f," " gradient norm = %.7f"% (i + 1, error, grad_norm))
        
        if error < best_error:
                best_error = error
                best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break
        
        if grad_norm <= min_grad_norm:
            break
    return p


N = args.num_trajectories #number of generated trajectories
#    model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
#    model = model.double().to(device)
#    model.load_state_dict(checkpoint["model_state_dict"])
test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)

#    for traj in test_dataset.trajectory_batches:
#        traj -= traj[:, :1, :]
#        traj *= hyper_params["data_scale"]

    #average ade/fde for k=20 (to account for variance in sampling)
num_samples = 1
average_ade, average_fde = 0, 0
for i in range(num_samples):
    dest = test(test_dataset)
    
 
#print(dest)
#print(dest.shape)       
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#palette = sns.color_palette("bright", 10)
##    X, y = load_digits(return_X_y=True) #(1797, 64) (1797,)
#    #print(X.shape)
#    #print(y.shape)
#    
X_embedded = fit(dest)
#sns.scatterplot(X_embedded[:,0], X_embedded[:,1],  legend='full', palette=palette)
plt.figure(figsize=(11.7,8.27))
plt.scatter(X_embedded[:,0], X_embedded[:,1], marker='*', c='b', label='32', s=10)
plt.legend()
plt.show()