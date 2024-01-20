import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse
sys.path.append("../utils/")
from utils.social_utils import *
import yaml
from utils.models_new import *
from utils.models import *
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--config_filename', '-cfn', type=str, default='optimal.yaml')
parser.add_argument('--save_file', '-sf', type=str, default='PECNET_social_model.pt')
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--root_path', '-rp', default="./")
parser.add_argument('--num_trajectories', '-nt', default=20) #number of trajectories to sample


args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)

with open(args.config_filename, 'r') as file:
    try:
        hyper_params = yaml.load(file, Loader = yaml.FullLoader)
    except:
        hyper_params = yaml.load(file)
file.close()
print(hyper_params)

checkpoint = torch.load('C:/Users/hp/Desktop/PECNet_new/PECNet-master-next_sigma_gpu_student_twin/saved_models/PECNET_social_model1.pt', map_location=device)


def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.3):  #2000 or 20000
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule
        
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def plot_grad_flow(named_parameters, e, i):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
#            print(p.grad.abs().mean().shape)
            ave_grads.append(p.grad.abs().mean())
            
#    print(ave_grads)
#    layers_selected = layers[-16:]
#    ave_grads_selected = ave_grads[-16:]
    
    layers_selected = layers[:17]
#    print(len(layers_selected))
    ave_grads_selected = ave_grads[:17]
    
    plt.plot(ave_grads_selected, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads_selected)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads_selected), 1), layers_selected, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads_selected))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
#    plt.show()
    if (e == 1) and i == 34 :
#        print((ave_grads_selected))
        plt.show()
    if (e % 5 == 0) and i == 34:
#        print((ave_grads_selected))
        plt.show()

def train(train_dataset, e):

    model_new.train()
    model.eval()
    train_loss = 0
    total_rcl, total_kld, total_adl, total_kld_traj, count = 0, 0, 0, 0, 0
    mu_l2, mu_inf, var_l2, var_inf, mu_l2_b, mu_inf_b, var_l2_b, var_inf_b = 0,0,0,0,0,0,0,0

    criterion = nn.MSELoss()

    beta = frange_cycle_linear(650)
#    beta[350:] = 1

    for i, (traj, mask, initial_pos) in enumerate(zip(train_dataset.trajectory_batches, train_dataset.mask_batches, train_dataset.initial_pos_batches)):
        traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
        x = traj[:, :hyper_params['past_length'], :]  #past trajectory
        y = traj[:, hyper_params['past_length']:, :]  #future trajectory

        x = x.contiguous().view(-1, x.shape[1]*x.shape[2]) # (x,y,x,y ... )
        x = x.to(device)
        dest_org = y[:, -1, :]
        dest = y[:, -1, :].to(device)
        future  = y[:, :-1, :].contiguous().view(y.size(0),-1).to(device)   #[513, 22]
        
        with torch.no_grad():
            all_l2_errors_dest_point = []
            all_guesses_point = []
            for index_point in range(20):
                dest_recon_point = model.forward(x, initial_pos, device=device)
#                dest_recon_point = dest_recon_point.cpu().numpy()
                all_guesses_point.append(dest_recon_point.cpu().numpy())
                l2error_sample_point = np.linalg.norm(dest_recon_point - dest_org, axis = 1)
                all_l2_errors_dest_point.append(l2error_sample_point)
                
            all_l2_errors_dest_point = np.array(all_l2_errors_dest_point)
            all_guesses_point = np.array(all_guesses_point)
            # average error
            l2error_avg_dest_point = np.mean(all_l2_errors_dest_point)
            # choosing the best guess
            indices_point = np.argmin(all_l2_errors_dest_point, axis = 0)
            best_guess_dest_point = all_guesses_point[indices_point,np.arange(x.shape[0]),  :]
            # taking the minimum error out of all guess
            l2error_dest_point = np.mean(np.min(all_l2_errors_dest_point, axis = 0))
            # back to torch land
            best_guess_dest_point = torch.DoubleTensor(best_guess_dest_point).to(device)
            # using the best guess for interpolation
            interpolated_future_point = model.predict(x, best_guess_dest_point, mask, initial_pos)
            interpolated_future_point_org = interpolated_future_point  #[513, 22]
#            print('shape', interpolated_future_point_org.shape)
            interpolated_future_point = interpolated_future_point.cpu().numpy()
            best_guess_dest_point = best_guess_dest_point.cpu().numpy()
            # final overall prediction
            predicted_future_point = np.concatenate((interpolated_future_point, best_guess_dest_point), axis = 1)
            predicted_future_point = np.reshape(predicted_future_point, (-1, hyper_params["future_length"], 2))
            # ADE error
            l2error_overall_point = np.mean(np.linalg.norm(y - predicted_future_point, axis = 2))
            l2error_overall_point /= hyper_params["data_scale"]
            l2error_dest_point /= hyper_params["data_scale"]
            l2error_avg_dest_point /= hyper_params["data_scale"]
        
            print('original_best_dest_error_train', l2error_dest_point)
            print('original_trajectory_error_train', l2error_overall_point)
        
       
        dest_recon, mu, var, interpolated_future, mu_f, var_f = model_new.forward(x, initial_pos, dest=dest, mask=mask, future = future, future_pred = interpolated_future_point_org, device=device)
        mu_l2_norm = np.linalg.norm(mu_f.cpu().detach().numpy(), axis=1)
        mu_inf_norm = np.linalg.norm(mu_f.cpu().detach().numpy(), np.inf, axis=1)
        var_l2_norm = np.linalg.norm(torch.exp(var_f).cpu().detach().numpy(), axis=1)
        var_inf_norm = np.linalg.norm(torch.exp(var_f).cpu().detach().numpy(), np.inf, axis=1)
        
#        print('mu_min', torch.min(mu))
#        print('var_min', torch.min(torch.exp(var)))
#        print('var_mean', torch.mean(torch.exp(var)))
#        print('var_max', torch.max(torch.exp(var)))
#        
#        print('mu_f_min', torch.min(mu_f))
#        print('var_f_min', torch.min(torch.exp(var_f)))
#        print('var_f_mean', torch.mean(torch.exp(var_f)))
#        print('var_f_max', torch.max(torch.exp(var_f)))
        
        
        
        rcl, kld, adl, kld_f = calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future, mu_f, var_f, e)
        loss = rcl + kld*hyper_params["kld_reg"] + adl*hyper_params["adl_reg"] + kld_f * beta[e] 
        
        
        optimizer.zero_grad()
        loss.backward()
#        plot_grad_flow(model.named_parameters(), e, i)
#        if i > 0:
#            plot_grad_flow(model.named_parameters())
        optimizer.step()
        train_loss += loss.item()
        total_rcl += rcl.item()
        total_kld += kld.item()
        total_adl += adl.item()
        total_kld_traj += kld_f.item() 
        mu_l2 += np.mean(mu_l2_norm)
        mu_inf += np.mean(mu_inf_norm)
        var_l2 += np.mean(var_l2_norm)
        var_inf += np.mean(var_inf_norm)
        count+= 1
        
    mu_l2_b = mu_l2/count
    mu_inf_b = mu_inf/count
    var_l2_b = var_l2/count
    var_inf_b = var_inf/count
    
    print('var_l2_b', var_l2_b)
    print('mu_l2_b', mu_l2_b)
    
    printout_file = open('results.txt', 'a')
    printout_file.write("\n var_l2_b: %f\n"%var_l2_b)
    printout_file.write("\n mu_l2_b: %f\n"%mu_l2_b)
    printout_file.close()
    return train_loss, total_rcl, total_kld, total_adl, total_kld_traj, mu_l2_b, mu_inf_b, var_l2_b, var_inf_b


def test(test_dataset, best_of_n = 1):
    '''Evalutes test metrics. Assumes all test data is in one batch'''

    model_new.eval()
    model.eval()
    assert best_of_n >= 1 and type(best_of_n) == int

    with torch.no_grad():
        for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
            traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
            x = traj[:, :hyper_params['past_length'], :]
            y = traj[:, hyper_params['past_length']:, :]
            z = traj[:, hyper_params['past_length']:-1, :]
            z = z.view(-1, z.shape[1]*z.shape[2])
            y = y.cpu().numpy()

            # reshape the data
            x = x.view(-1, x.shape[1]*x.shape[2])
            x = x.to(device)

            dest = y[:, -1, :]
            
            
            all_l2_errors_dest_point = []
            all_guesses_point = []
            for index_point in range(20):

                dest_recon_point = model.forward(x, initial_pos, device=device)
                dest_recon_point = dest_recon_point.cpu().numpy()
                all_guesses_point.append(dest_recon_point)

                l2error_sample_point = np.linalg.norm(dest_recon_point - dest, axis = 1)
                all_l2_errors_dest_point.append(l2error_sample_point)

            all_l2_errors_dest_point = np.array(all_l2_errors_dest_point)
            all_guesses_point = np.array(all_guesses_point)
            # average error
            l2error_avg_dest_point = np.mean(all_l2_errors_dest_point)

            # choosing the best guess
            indices_point = np.argmin(all_l2_errors_dest_point, axis = 0)

            best_guess_dest_point = all_guesses_point[indices_point,np.arange(x.shape[0]),  :]

            # taking the minimum error out of all guess
            l2error_dest_point = np.mean(np.min(all_l2_errors_dest_point, axis = 0))

            # back to torch land
            best_guess_dest_point = torch.DoubleTensor(best_guess_dest_point).to(device)

            # using the best guess for interpolation
            interpolated_future_point = model.predict(x, best_guess_dest_point, mask, initial_pos)
            interpolated_future_point_org = interpolated_future_point
            interpolated_future_point = interpolated_future_point.cpu().numpy()
            best_guess_dest_point = best_guess_dest_point.cpu().numpy()
            
#            test_output.append(interpolated_future_point)

            # final overall prediction
            predicted_future_point = np.concatenate((interpolated_future_point, best_guess_dest_point), axis = 1)
            predicted_future_point = np.reshape(predicted_future_point, (-1, hyper_params["future_length"], 2))

            # ADE error
            l2error_overall_point = np.mean(np.linalg.norm(y - predicted_future_point, axis = 2))

            l2error_overall_point /= hyper_params["data_scale"]
            l2error_dest_point /= hyper_params["data_scale"]
            l2error_avg_dest_point /= hyper_params["data_scale"]
            
            print('original_best_dest_error_test', l2error_dest_point)
            print('original_trajectory_error_test', l2error_overall_point)
            
            
            all_l2_errors_dest = []
            all_guesses = []
            for _ in range(best_of_n):

                dest_recon = model_new.forward(x, initial_pos, mask=mask, device=device)
                dest_recon = dest_recon.cpu().numpy()
                all_guesses.append(dest_recon)

                l2error_sample = np.linalg.norm(dest_recon - dest, axis = 1)
#                print(l2error_sample.shape) #(2829)
#                print('l2error_sample', np.min(l2error_sample))
                all_l2_errors_dest.append(l2error_sample)

            all_l2_errors_dest = np.array(all_l2_errors_dest)
#            print(all_l2_errors_dest.shape)   #(20, 2829)
            all_guesses = np.array(all_guesses)
#            print(all_guesses.shape)  #(20, 2829, 2)
            # average error
            l2error_avg_dest = np.mean(all_l2_errors_dest)

            # choosing the best guess
            indices = np.argmin(all_l2_errors_dest, axis = 0)
#            print('indices', indices.shape)
#            print('x_shape', x.shape[0]) #(2829)

            best_guess_dest = all_guesses[indices,np.arange(x.shape[0]),  :]
#            best_guess_dest = best_guess_dest.cpu().numpy()

            # taking the minimum error out of all guess
            l2error_dest = np.mean(np.min(all_l2_errors_dest, axis = 0))
            all_l2_errors_dest_f = []
            all_guesses_f = []
#            all_sigma = []
#            pred_sigma_x_20, pred_sigma_y_20, pred_x_20, pred_y_20 = [], [], [], []
            for _ in range(20):
                
                best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)
                # using the best guess for interpolation
                interpolated_future = model_new.predict(x, best_guess_dest, mask, initial_pos, interpolated_future_point_org)
                interpolated_future = interpolated_future.cpu().detach().numpy()
#                interpolated_future_sigma = interpolated_future_sigma.cpu().numpy()
#                interpolated_future_sigma_x = interpolated_future_sigma[:, 0::2] + interpolated_future_sigma[:, 1::2]   #(2829, 11)
#                pred_x_20_tmp = interpolated_future[:, 0::2]
#                pred_y_20_tmp = interpolated_future[:, 1::2]
#                pred_sigma_x_20_tmp = interpolated_future_sigma[:, 0::2]
#                pred_sigma_y_20_tmp = interpolated_future_sigma[:, 1::2]
                
#                interpolated_future_sigma_x = interpolated_future_sigma_x .cpu().numpy()
#                best_guess_dest = best_guess_dest.cpu().numpy()

                # final overall prediction
                predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)
                predicted_future = np.reshape(predicted_future, (-1, hyper_params['future_length'], 2)) # making sure
#                print(predicted_future.shape) #(2829, 12, 2)
                all_guesses_f.append(predicted_future)
                # ADE error
                l2error_sample_f = np.linalg.norm(y - predicted_future, axis = 2)
#                print(l2error_sample_f.shape)  #(2829, 12)
#                print('l2error_sample_f', np.min(l2error_sample_f))
                all_l2_errors_dest_f.append(l2error_sample_f)
#                all_sigma.append(interpolated_future_sigma_x)
#                pred_x_20.append(pred_x_20_tmp)
#                pred_y_20.append(pred_y_20_tmp)
#                pred_sigma_x_20.append(pred_sigma_x_20_tmp)
#                pred_sigma_y_20.append(pred_sigma_y_20_tmp)
                
            
#            all_sigma = np.array(all_sigma) #(10, 2829, 11)
#            pred_x_20, pred_y_20, pred_sigma_x_20, pred_sigma_y_20 = np.array(pred_x_20), np.array(pred_y_20), np.array(pred_sigma_x_20), np.array(pred_sigma_y_20)
#            print(pred_x_20.shape) #(20, 2829, 11)
#            all_sigma = np.mean(all_sigma, axis = 2)
#            indices_sigma = np.argmin(all_sigma, axis = 0)
            
#            final_x_max_index = np.argmin(pred_sigma_x_20, axis=0)
#            final_y_max_index = np.argmin(pred_sigma_y_20, axis=0)
#            final_x_pred = np.zeros_like(final_x_max_index)
#            final_y_pred = np.zeros_like(final_y_max_index)
#            
#            
#            for k in range(pred_x_20.shape[0]):
#                new_x = np.where(final_x_max_index==k, 1, 0)
#                new_y = np.where(final_y_max_index==k, 1, 0)
#                final_x_pred = final_x_pred + pred_x_20[k] * new_x
#                final_y_pred = final_y_pred + pred_y_20[k] * new_y
            
            
#            final_pred = np.zeros_like(z)
#            final_pred[:, 0::2] = final_x_pred
#            final_pred[:, 1::2] = final_y_pred
#            final_predicted_future = np.concatenate((final_pred, best_guess_dest), axis = 1)
#            final_predicted_future = np.reshape(final_predicted_future, (-1, hyper_params['future_length'], 2)) 
            
            all_l2_errors_dest_f = np.array(all_l2_errors_dest_f)
#            print(all_l2_errors_dest_f.shape)  #(20, 2829, 12)
            all_guesses_f = np.array(all_guesses_f)
#            print(all_guesses_f.shape)  #(20, 2829, 12, 2)
            # average error
            l2error_overall = np.mean(all_l2_errors_dest_f)
#            print('l2error_overall', l2error_overall.shape)
            # choosing the best guess
#            indices_f = np.argmin(all_l2_errors_dest_f, axis = 0)

#            best_guess_dest_f = all_guesses_f[indices_f,np.arange(x.shape[0]),  :]
            # taking the minimum error out of all guess
            l2error_dest_f = np.mean(np.min(all_l2_errors_dest_f, axis = 0))
            
            scale = 1.86
            
#            final_l2error_overall = np.mean(np.linalg.norm(y - final_predicted_future, axis = 2)) 
#            final_l2error_overall/= scale
#            print('our newly estimated error', final_l2error_overall)
#            
#            l2error_dest_f_estimated = np.mean(all_l2_errors_dest_f[indices_sigma, np.arange(x.shape[0])])
#            l2error_dest_f_estimated/= scale
#            print('our estimated error', l2error_dest_f_estimated)
            
            l2error_dest_max = np.mean(np.max(np.mean(all_l2_errors_dest_f, axis=2), axis = 0))
            l2error_dest_max/= scale
            print('average maximum', l2error_dest_max)
            
            
            l2error_overall /= scale
            l2error_dest_f /= scale
            l2error_dest /= scale
            l2error_avg_dest /= scale
            
            print('average', l2error_overall)
            
            printout_file = open('results.txt', 'a')
#            printout_file.write("\n our estimated error: %f\n"%l2error_dest_f_estimated)
            printout_file.write("\n average maximum: %f\n"%l2error_dest_max)
            printout_file.write("\n average: %f\n"%l2error_overall)
            printout_file.close()

            print('Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(l2error_dest, l2error_avg_dest))
            print('Test time error overall (ADE) best: {:0.3f}'.format(l2error_dest_f))

    return l2error_dest_f, l2error_dest, l2error_avg_dest

model_new = PECNet_new(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
model_new = model_new.double().to(device)
hyper_params_org = checkpoint["hyper_params"]
model = PECNet(hyper_params_org["enc_past_size"], hyper_params_org["enc_dest_size"], hyper_params_org["enc_latent_size"], hyper_params_org["dec_size"], hyper_params_org["predictor_hidden_size"], hyper_params_org['non_local_theta_size'], hyper_params_org['non_local_phi_size'], hyper_params_org['non_local_g_size'], hyper_params_org["fdim"], hyper_params_org["zdim"], hyper_params_org["nonlocal_pools"], hyper_params_org['non_local_dim'], hyper_params_org["sigma"], hyper_params_org["past_length"], hyper_params_org["future_length"], args.verbose)
model = model.double().to(device)
model.load_state_dict(checkpoint["model_state_dict"])

#model.load_state_dict(checkpoint["model_state_dict"])

optimizer = optim.Adam(model_new.parameters(), lr= 0.0003)


train_dataset = SocialDataset(set_name="train", b_size=hyper_params["train_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)
test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)

scale = 1.86

# shift origin and scale data
for traj in train_dataset.trajectory_batches:
    traj -= traj[:, :1, :]
    traj *= scale
for traj in test_dataset.trajectory_batches:
    traj -= traj[:, :1, :]
    traj *= scale


best_test_loss = 50 # start saving after this threshold
best_endpoint_loss = 50
N = hyper_params["n_values"]


training_loss = np.zeros(hyper_params['num_epochs'])
rcl_loss, kld_loss, adl_loss, kld_traj_loss, ADE_training_loss = np.zeros(hyper_params['num_epochs']), np.zeros(hyper_params['num_epochs']), np.zeros(hyper_params['num_epochs']), np.zeros(hyper_params['num_epochs']), np.zeros(hyper_params['num_epochs'])
ADE_test_loss = np.zeros(hyper_params['num_epochs'])
mu_l2_epoch, mu_inf_epoch, var_l2_epoch, var_inf_epoch = np.zeros(hyper_params['num_epochs']), np.zeros(hyper_params['num_epochs']), np.zeros(hyper_params['num_epochs']),  np.zeros(hyper_params['num_epochs'])



for e in range(650):
    train_loss, rcl, kld, adl, kld_traj, mu_l2_b, mu_inf_b, var_l2_b, var_inf_b = train(train_dataset, e)
    test_loss, final_point_loss_best, final_point_loss_avg = test(test_dataset, best_of_n = N)
    printout_file = open('results.txt', 'a')
    
    training_loss[e] = train_loss
    rcl_loss[e] = rcl
    kld_loss[e] = kld
    adl_loss[e] = adl
    kld_traj_loss[e] = kld_traj
    mu_l2_epoch[e] = mu_l2_b
    mu_inf_epoch[e] = mu_inf_b
    var_l2_epoch[e] = var_l2_b
    var_inf_epoch[e] = var_inf_b

    if best_test_loss > test_loss:
        print("Epoch: ", e)
        print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_loss))
        printout_file.write("Epoch: %d\n"%e)
        printout_file.write("################## BEST PERFORMANCE######: %f\n"%test_loss)
        best_test_loss = test_loss
        if best_test_loss < 100.25:
            save_path = args.save_file
            torch.save({
                        'hyper_params': hyper_params,
                        'model_new_state_dict': model_new.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, save_path)
            print("Saved model_new to:\n{}".format(save_path))

    if final_point_loss_best < best_endpoint_loss:
        best_endpoint_loss = final_point_loss_best

    print("Train Loss", train_loss)
    print("RCL", rcl)
    print("KLD", kld)
    print("ADL", adl)
    print("KLD_Traj", kld_traj)
    print("Test ADE", test_loss)
    print("Test Average FDE (Across  all samples)", final_point_loss_avg)
    print("Test Min FDE", final_point_loss_best)
    print("Test Best ADE Loss So Far (N = {})".format(N), best_test_loss)
    print("Test Best Min FDE (N = {})".format(N), best_endpoint_loss)
    printout_file.write("Train Loss: %f\n"%train_loss)
    printout_file.write("RCL: %f\n"%rcl)
    printout_file.write("KLD: %f\n"%kld)
    #printout_file.write("KLD_traj: %f\n"%kld_traj)
    printout_file.write("ADL: %f\n"%adl)
    printout_file.write("KLD_Traj: %f\n"%kld_traj)
    printout_file.write("Test ADE: %f\n"%test_loss)
    printout_file.write("Test Average FDE (Across  all samples): %f\n"%final_point_loss_avg)
    printout_file.write("Test Min FDE: %f\n"%final_point_loss_best)
    printout_file.write("Test Best ADE Loss So Far: %f\n"%best_test_loss)
    printout_file.write("Test Best Min FDE So Far: %f\n"%best_endpoint_loss)
    printout_file.close()


with open('training_loss.npy', 'wb') as f:
    np.save(f, training_loss)
    
with open('rcl_loss.npy', 'wb') as f:
    np.save(f, rcl_loss)
    
with open('kld_loss.npy', 'wb') as f:
    np.save(f, kld_loss)
    
with open('adl_loss.npy', 'wb') as f:
    np.save(f, adl_loss)
    
with open('kld_traj_loss.npy', 'wb') as f:
    np.save(f, kld_traj_loss)
    
    
with open('mu_l2_epoch.npy', 'wb') as f:
    np.save(f, mu_l2_epoch)
    
with open('mu_inf_epoch.npy', 'wb') as f:
    np.save(f, mu_inf_epoch)
    
with open('var_l2_epoch.npy', 'wb') as f:
    np.save(f, var_l2_epoch)
    
with open('var_inf_epoch.npy', 'wb') as f:
    np.save(f, var_inf_epoch)

