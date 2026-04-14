import numpy as np
import torch
import random
import pickle
import io
import os
import sys
import importlib
import types
from PIL import Image
from pathlib import Path

TRAJFLOW = Path(__file__).resolve().parent
if str(TRAJFLOW) not in sys.path:
    sys.path.insert(0, str(TRAJFLOW))

from TrajFlow.flowModels import TrajFlow, Future_Encoder, Future_Decoder, Future_Seq2Seq, Scene_Encoder, Future_Decoder_Control, Future_Seq2Seq_Control


class _CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

class trajflow_meszaros():
    '''
    TrajFlow is a single agent prediction model that combine Normalizing Flows with
    GRU-based autoencoders.
    
    The model was implemented into the framework by its original creators, and 
    the model was first published under:
        
    Mészáros, A., Alonso-Mora, J., & Kober, J. (2023). Trajflow: Learning the 
    distribution over trajectories. arXiv preprint arXiv:2304.05166.
    '''    
    def __init__(self):
        # Set model kwargs
        self.model_kwargs = {}

        # Load gpu
        if torch.cuda.is_available():
            self.device = torch.device('cuda', index=0) 
            torch.cuda.set_device(0)
        else:
            self.device = torch.device('cpu')
        
        self.dt = 0.1
        self.has_map = True
        self.has_graph = True
        
        self.num_timesteps_in = 10
        self.num_timesteps_out = 20
        
        self.agents_to_predict = 'predefined'
        
        self.t_e_quantile = np.linspace(0.1, 0.9, 9)
        self.num_samples_path_pred = 1
                
        path = __file__
        model_name = 'model_weights.npy'
        self.model_file = os.path.dirname(path) + os.sep + 'Model_weights' + os.sep + model_name

        # Initialize actual model
        self.setup_method()

        # Load the model weights
        self.load_method()

        # Prepare images
        image_name = 'L-GAP_image_6_px_per_meter.png'
        image_file = os.path.dirname(path) + os.sep + 'Data' + os.sep + image_name
        image = Image.open(image_file)
        rgb_image = image.convert('RGB')
        self.img = torch.from_numpy(np.array(rgb_image, dtype=np.uint8)).to(device = self.device)
        self.px_per_meter = 6
    
    def define_default_kwargs(self):
        if not ('batch_size' in self.model_kwargs.keys()):
            self.model_kwargs['batch_size'] = 128

        if not ('hs_rnn' in self.model_kwargs.keys()):
            self.model_kwargs['hs_rnn'] = 16

        if not ('n_layers_rnn' in self.model_kwargs.keys()):
            self.model_kwargs['n_layers_rnn'] = 3

        if not ('fut_enc_sz' in self.model_kwargs.keys()):
            self.model_kwargs['fut_enc_sz'] = 20

        if not ('scene_encoding_size' in self.model_kwargs.keys()):
            self.model_kwargs['scene_encoding_size'] = 64

        if not ('obs_encoding_size' in self.model_kwargs.keys()):
            self.model_kwargs['obs_encoding_size'] = 64

        # TODO: Add the GNN encoding size (currently 32)

        if not ('beta_noise' in self.model_kwargs.keys()):
            self.model_kwargs['beta_noise'] = 0 # 0.2 (P) / 0.002

        if not ('gamma_noise' in self.model_kwargs.keys()):
            self.model_kwargs['gamma_noise'] = 0 # 0.02 (P) / 0.002

        if not ('alpha' in self.model_kwargs.keys()):
            self.model_kwargs['alpha'] = 10 # 10 (P) / 3

        if not ('s_min' in self.model_kwargs.keys()):
            self.model_kwargs['s_min'] = 0.8 # 0.3 (P) / 0.8 

        if not ('s_max' in self.model_kwargs.keys()):
            self.model_kwargs['s_max'] = 1.2 # 1.7 (P) / 1.2

        if not ('sigma' in self.model_kwargs.keys()):  
            self.model_kwargs['sigma'] = 0.5 # 0.5 (P) / 0.2

        if not ('fut_ae_epochs' in self.model_kwargs.keys()):
            self.model_kwargs['fut_ae_epochs'] = 5000
        
        if not ('fut_ae_lr' in self.model_kwargs.keys()):
            self.model_kwargs['fut_ae_lr'] = 5e-4

        if not ('fut_ae_lr_decay' in self.model_kwargs.keys()):
            self.model_kwargs['fut_ae_lr_decay'] = 1.0

        if not ('fut_ae_wd' in self.model_kwargs.keys()):
            self.model_kwargs['fut_ae_wd'] = 1e-4

        if not ('flow_epochs' in self.model_kwargs.keys()):
            self.model_kwargs['flow_epochs'] = 500

        if not ('flow_lr' in self.model_kwargs.keys()):
            self.model_kwargs['flow_lr'] = 1e-3

        if not ('flow_lr_decay' in self.model_kwargs.keys()):
            self.model_kwargs['flow_lr_decay'] = 0.98

        if not ('flow_wd' in self.model_kwargs.keys()):
            self.model_kwargs['flow_wd'] = 1e-5

        if not ('vary_input_length' in self.model_kwargs.keys()):
            self.model_kwargs['vary_input_length'] = False

        if not ('scale_AE' in self.model_kwargs.keys()):
            self.model_kwargs['scale_AE'] = False

        if not ('scale_NF' in self.model_kwargs.keys()):
            self.model_kwargs['scale_NF'] = False

        if not ('pos_loss' in self.model_kwargs.keys()):
            self.model_kwargs['pos_loss'] = True

        if not("decoder_type" in self.model_kwargs.keys()):
            self.model_kwargs["decoder_type"] = "none"

        if not('seed' in self.model_kwargs.keys()):
            self.model_kwargs['seed'] = 0

        if not('interactions' in self.model_kwargs.keys()):
            self.model_kwargs['interactions'] = True


        self.hs_rnn = self.model_kwargs['hs_rnn']
        self.n_layers_rnn = self.model_kwargs['n_layers_rnn']
        self.fut_enc_sz = self.model_kwargs['fut_enc_sz'] 

        self.scene_encoding_size = self.model_kwargs['scene_encoding_size']
        self.obs_encoding_size = self.model_kwargs['obs_encoding_size'] 
        
        self.beta_noise = self.model_kwargs['beta_noise']
        self.gamma_noise = self.model_kwargs['gamma_noise']
        
        self.alpha = self.model_kwargs['alpha']
        self.s_min = self.model_kwargs['s_min']
        self.s_max = self.model_kwargs['s_max']
        self.sigma = self.model_kwargs['sigma']

        self.fut_ae_epochs = self.model_kwargs['fut_ae_epochs']
        self.fut_ae_lr = self.model_kwargs['fut_ae_lr']
        self.fut_ae_lr_decay = self.model_kwargs['fut_ae_lr_decay']
        self.fut_ae_wd = self.model_kwargs['fut_ae_wd']

        self.flow_epochs = self.model_kwargs['flow_epochs']
        self.flow_lr = self.model_kwargs['flow_lr']
        self.flow_lr_decay = self.model_kwargs['flow_lr_decay']
        self.flow_wd = self.model_kwargs['flow_wd']

        self.vary_input_length = self.model_kwargs['vary_input_length']
        self.scale_AE = self.model_kwargs['scale_AE']
        self.scale_NF = self.model_kwargs['scale_NF']
        self.pos_loss = self.model_kwargs['pos_loss']

        # Get Decoder Type
        self.decoder_type = self.model_kwargs["decoder_type"] 

        self.interactions = self.model_kwargs['interactions']

    
    def setup_method(self):        
        # set random seeds
        self.define_default_kwargs()
        seed = self.model_kwargs['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.batch_size = self.model_kwargs['batch_size']
        
        # Required attributes of the model
        self.min_t_O_train = self.num_timesteps_out
        self.max_t_O_train = 100
        self.predict_single_agent = True
        self.can_use_map = True
        self.can_use_graph = False
        # If self.can_use_map, the following is also required
        self.target_width = 257
        self.target_height = 156
        self.grayscale = True
        
        self.norm_rotation = True

        self.std_pos_ped = 1
        self.std_pos_veh = 1 

        
    def load_method(self):        
        fut_model_file = self.model_file[:-4] + '_AE'
        flow_dist_file = self.model_file[:-4] + '_NF'
        self.fut_model = _CPU_Unpickler(open(fut_model_file, 'rb')).load()
        self.flow_dist = _CPU_Unpickler(open(flow_dist_file, 'rb')).load()
        self.fut_model.to(self.device)
        self.flow_dist.to(self.device)
        if hasattr(self.fut_model, 'device'):
            self.fut_model.device = self.device
        if hasattr(self.flow_dist, 'device'):
            self.flow_dist.device = self.device

        
    def _repeat_rowwise(self, x, n):
        org_dim = x.size(-1)
        x = x.repeat(1, 1, n)
        return x.view(-1, n, org_dim)


    def _interpolate_image(self, imgs_rot, pos_old, image):
        
        useful = ((0 <= pos_old[...,0]) & (pos_old[...,0] <= image.shape[1] - 1) &
                  (0 <= pos_old[...,1]) & (pos_old[...,1] <= image.shape[0] - 1))
        
        useful_ind, useful_row, useful_col = torch.where(useful)
        pos_old = pos_old[useful_ind, useful_row, useful_col,:]
        
        pos_up  = torch.ceil(pos_old).to(dtype = torch.int64)
        pos_low = torch.floor(pos_old).to(dtype = torch.int64)
        
        imgs_rot_uu = image[pos_up[:,1],  pos_up[:,0]]
        imgs_rot_ul = image[pos_up[:,1],  pos_low[:,0]]
        imgs_rot_lu = image[pos_low[:,1], pos_up[:,0]]
        imgs_rot_ll = image[pos_low[:,1], pos_low[:,0]]
        
        del pos_up, pos_low
        
        pos_fac = torch.remainder(pos_old, 1)
        
        imgs_rot_u = imgs_rot_uu * (pos_fac[:,[0]]) + imgs_rot_ul * (1 - pos_fac[:,[0]])
        imgs_rot_l = imgs_rot_lu * (pos_fac[:,[0]]) + imgs_rot_ll * (1 - pos_fac[:,[0]])
        
        del imgs_rot_uu, imgs_rot_ul, imgs_rot_lu, imgs_rot_ll
        
        imgs_rot_v = imgs_rot_u * (pos_fac[:,[1]]) + imgs_rot_l * (1 - pos_fac[:,[1]])
        
        if imgs_rot.shape[-1] == 1:
            imgs_rot[useful] = imgs_rot_v.mean(-1, keepdims = True).to(dtype = image.dtype)
        else:
            imgs_rot[useful] = imgs_rot_v.to(dtype = image.dtype)
        
        return imgs_rot


    def return_batch_images(self, center, rot_angle, target_width, target_height, grayscale):

        if grayscale:
            Imgs = torch.zeros((len(center), target_height, target_width, 1), dtype = torch.uint8)
        else:
            Imgs = torch.zeros((len(center), target_height, target_width, 3), dtype = torch.uint8)
        
        # Find the gpu
        device = center.device
        Imgs = Imgs.to(device = device)
        
        # Get rotation matrix (R * x is from orignal to current)
        Rot_matrix = torch.stack([
            torch.stack([torch.cos(rot_angle), torch.sin(rot_angle)], -1),
            torch.stack([-torch.sin(rot_angle), torch.cos(rot_angle)], -1),
        ], -2)

        img_rot_angle = torch.zeros_like(rot_angle)
        img_x_center = torch.ones_like(rot_angle) * 163.66
        img_y_center = torch.ones_like(rot_angle) * -23.66
        Rot_matrix_old = torch.stack([
            torch.stack([torch.cos(img_rot_angle), torch.sin(img_rot_angle)], -1),
            torch.stack([-torch.sin(img_rot_angle), torch.cos(img_rot_angle)], -1),
        ], -2)
        center_old = torch.stack([img_x_center, img_y_center], -1)
        
        # setup meshgrid
        height_fac = (target_height - 1) / 2 
        width_fac = (target_width - 1) / 2           
        Ypx, Xpx = torch.meshgrid(torch.arange(-height_fac, height_fac + 0.5, device = device), 
                                    torch.arange(-width_fac, width_fac + 0.5, device = device),
                                    indexing = 'ij')
        
        Pos_old = torch.stack([Xpx, Ypx], -1).unsqueeze(0)
        Pos_old[...,1] *= -1
        # Pos_old: Position in goal coordinate system in Px.

        n = 250
        loc_M2px = 1 / self.px_per_meter
        for i in range(0, len(center), n):
            Index = torch.arange(i, min(i + n, len(center)), device = device)
            
            # Position in goal coordinate system in Px.
            pos_old = Pos_old * loc_M2px
            
            # Get position in the coordinate system in self.paths
            pos_old = torch.matmul(pos_old, Rot_matrix[Index].unsqueeze(1))
            pos_old = pos_old + center[Index,:].unsqueeze(1).unsqueeze(1)
            
            # Get position im Image aligned coordinate system
            pos_old = torch.matmul(pos_old, Rot_matrix_old[Index].unsqueeze(1))
            pos_old = pos_old + center_old[Index,:].unsqueeze(1).unsqueeze(1)
            
            # Get pixel position in Image
            pos_old = pos_old / loc_M2px
            pos_old[...,1] *= -1
            
            torch.cuda.empty_cache()
            Imgs[Index] = self._interpolate_image(Imgs[Index], pos_old, self.img)
    
        return Imgs

    def extract_images(self, X):
        '''
        X.shape = (n, num_timesteps_in, 2)
        Returns image data 
        img # num_overall_agents, height, width, channels
        img_m_per_px # num_overall_agents
        '''

        # Only use the input positions
        centre = X[:, -1,:].clone()
        x_rel = centre - X[:, -2,:]
        rot = torch.angle(x_rel[:,0] + 1j * x_rel[:,1])

        img_needed = self.return_batch_images(
            centre, 
            rot,
            target_height = self.target_height,
            target_width = self.target_width,
            grayscale = self.grayscale,
        )
        return img_needed


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Generate stochastics outputs for each point x, given the condition x.

        Sample y ~ p(y|x,c)

        Args:
            x: Input tensor with shape (..., d)

        Returns:
            Tensor with shape (..., q) containing the generated samples.
        """

        # Get the batch dimensions
        batch_dims = x.shape[:-1]

        # Flatten the batch dimensions for processing
        x_flat = x.view(-1, x.shape[-1])

        # We need to assure that d can be reshaped to (-1, self.num_timesteps_in, 2)
        assert np.mod(x_flat.shape[-1], self.num_timesteps_in * 2) == 0, "The last dimension of x must be divisible by num_timesteps_in * 2"
        # get proper shape for the model input
        X = x_flat.view(len(x_flat), -1, self.num_timesteps_in, 2)
        # Assume vehicles only
        T = np.full(X.shape[:2], ord('V'), dtype=np.uint8)
        T = torch.from_numpy(T).to(device = self.device)

        # We need to assure that the image is correctly shaped as well
        img = self.extract_images(X[:, 0]) # (num_overall_agents, height, width, 1)
        img = img.permute(0,3,1,2)
        img = img.float() / 255
            

        with torch.no_grad():
            actual_batch_size = len(X)
            past_traj, rot_angles_rad = self.flow_dist._normalize_rotation(X)
            
            x_t = past_traj[:,0,-1:,:]
            x_t = self._repeat_rowwise(x_t, self.num_samples_path_pred)
            x_t = x_t.reshape(actual_batch_size * self.num_samples_path_pred,-1).unsqueeze(1)
            
            rot_angles_rad = rot_angles_rad.repeat_interleave(self.num_samples_path_pred)

            if img is not None: 
                samples_rel, _ = self.flow_dist.sample(self.num_samples_path_pred, X, T, img)
            else:
                samples_rel, _ = self.flow_dist.sample(self.num_samples_path_pred, X, T)
            
            samples_rel = samples_rel.squeeze(0)
                    
            hidden = torch.tile(samples_rel.reshape(-1, self.fut_enc_sz).unsqueeze(0), (self.fut_model.decoder.nl,1,1))
            
            # Decoder part
            x = samples_rel.reshape(-1, self.fut_enc_sz).unsqueeze(1)
            prev_step = torch.tensor([1.0,0.0]).unsqueeze(0).repeat(actual_batch_size * self.num_samples_path_pred,1).to(self.device)
            
            outputs = torch.zeros(actual_batch_size * self.num_samples_path_pred, self.num_timesteps_out, 2).to(device = self.device)
            for t in range(0, self.num_timesteps_out):
                if self.decoder_type == "none":
                    output, hidden = self.fut_model.decoder(x, hidden)
                else:
                    output, hidden = self.fut_model.decoder(prev_step, x, hidden)
                
                prev_step = output.squeeze()
                outputs[:, t, :] = prev_step
                
                x = hidden[-1].unsqueeze(1)
            
            y_hat = self.flow_dist._rel_to_abs(outputs, x_t)

            # invert rotation normalization
            y_hat = self.flow_dist._rotate(y_hat, x_t, -1 * rot_angles_rad.unsqueeze(1))

            y_hat = y_hat.reshape(actual_batch_size, self.num_samples_path_pred, self.num_timesteps_out, 2)
            
            Y_pred = y_hat.detach()
        
        # Add potentially missing batch dimensions back to the output
        if len(Y_pred.shape) == 3:
            Y_pred = Y_pred[None]

        Y = Y_pred * self.std_pos_veh # [:, num_samples_path_pred, num_timesteps_out, 2]

        y = Y.view(len(Y), -1) # Flatten the last dimensions to match the expected output shape (:, q)

        # Unflatten the output to match the original batch dimensions
        y = y.view(*batch_dims, -1) # [..., q]
        return y
    