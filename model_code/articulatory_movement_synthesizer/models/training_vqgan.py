import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from discriminator import Discriminator
from vqgan import VQGAN
from utils import load_data, weights_init
from torch.utils.tensorboard import SummaryWriter


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
workspace_root = Path(
    os.environ.get("COVERT_READING_WORKSPACE", REPOSITORY_ROOT / "workspace")
).expanduser().resolve()
clean_data_path = Path(
    os.environ.get("COVERT_READING_DATA_ROOT", workspace_root / "model_data")
).expanduser().resolve()
class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args,n_layers=2).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        band = args.band 
        base_path = clean_data_path / "decoding_experiment" / f"HS{args.HS}_{args.reading_name}_{args.elec_type}_{band}"
        self.checkpoints_path = clean_data_path / "checkpoints"
        infer_path = Path(f"{base_path}_infer_{args.used_sound}")
        fold_path = Path(f"{base_path}_{args.fold_ind}")
        percentage_path = Path(f"{base_path}_{args.fold_ind}_percentage_{args.percent}")
        for path in (base_path, infer_path, fold_path, percentage_path, self.checkpoints_path):
            path.mkdir(parents=True, exist_ok=True)
        if args.is_infer == True:
            self.writer = SummaryWriter(log_dir=infer_path, comment=f"HS{args.HS}")
        elif args.is_percentage == True:
            self.writer = SummaryWriter(log_dir=percentage_path, comment=f"HS{args.HS}")
        elif args.is_fold == True:
            self.writer = SummaryWriter(log_dir=fold_path, comment=f"HS{args.HS}")
        else:
            self.writer = SummaryWriter(log_dir=base_path, comment=f"HS{args.HS}")
        

        self.train_test(args)
 
    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    def get_mean_r(self,trace,trace_predict):
        r=0
        r_list =[]
        for i in range(13):
        # print(np.corrcoef(np.vstack((trace_predict[:, i].reshape(-1), trace[:, i].reshape(-1))))[0, 1])
            r_list.append(np.corrcoef(np.vstack((trace_predict[0][:,i].reshape(-1), trace[0][:,i].reshape(-1))))[0,1])

        r=np.nanmean(r_list)
        
        return r
       
    def train_test(self, args):
        train_dataset,val_dataset,_ = load_data(args)
        steps_per_epoch = len(train_dataset)
        steps_per_epoch_val = len(val_dataset)
        best_r = 0

        for epoch in range(args.epochs):
            self.vqgan.train()
            self.discriminator.train()
            train_r = 0.0
            val_r = 0.0
            train_vq_loss = 0.0
            val_vq_loss = 0.0
            train_gan_loss = 0.0
            val_gan_loss = 0.0
            train_size = 0
            val_size = 0
            with tqdm(range(len(train_dataset))) as pbar:
                for i, ecogs_trace in zip(pbar, train_dataset):
                    
                    ecogs = ecogs_trace[0].unsqueeze(1).to(device=args.device)
                    trace = ecogs_trace[1].unsqueeze(1).to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(ecogs)

                    


                    disc_real = self.discriminator(trace)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)
                    # perceptual_loss = self.perceptual_loss(trace, decoded_images)
                    # print(trace.shape, decoded_images.shape)
                    rec_loss = torch.abs(trace - decoded_images)
                    
                    # perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    adaptive_weight = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * adaptive_weight * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                   
                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3),
                        HS = args.HS
                    )
                    pbar.update(0)
                    # Accumulate sample-weighted training metrics.
                    train_vq_loss += vq_loss.cpu().detach().numpy().item() * ecogs.size(0)
                    train_gan_loss += gan_loss.cpu().detach().numpy().item() * ecogs.size(0)
                    
                    for ii in range(len(trace)):
                        train_r += self.get_mean_r(trace[ii].cpu().detach().numpy() ,decoded_images[ii].cpu().detach().numpy())
                    train_size += ecogs.size(0)
                    
            
            with tqdm(range(len(val_dataset))) as pbar:
                with torch.no_grad():
                    self.vqgan.eval()
                    self.discriminator.eval()
                    for i, ecogs_trace in zip(pbar, val_dataset):
                        
                        ecogs = ecogs_trace[0].unsqueeze(1).to(device=args.device)
                        trace = ecogs_trace[1].unsqueeze(1).to(device=args.device)
                        decoded_images, _, q_loss = self.vqgan(ecogs)

                        


                        disc_real = self.discriminator(trace)
                        disc_fake = self.discriminator(decoded_images)

                        disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch_val+i, threshold=args.disc_start)
                        # perceptual_loss = self.perceptual_loss(trace, decoded_images)
                        # print(trace.shape, decoded_images.shape)
                        rec_loss = torch.abs(trace - decoded_images)
                        
                        # perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                        perceptual_rec_loss = args.rec_loss_factor * rec_loss
                        perceptual_rec_loss = perceptual_rec_loss.mean()
                        g_loss = -torch.mean(disc_fake)

                        adaptive_weight = 0
                        vq_loss = perceptual_rec_loss + q_loss + disc_factor * adaptive_weight * g_loss

                        d_loss_real = torch.mean(F.relu(1. - disc_real))
                        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                        gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                        pbar.set_postfix(
                            VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                            GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                        )
                        pbar.update(0)
                        # Accumulate sample-weighted validation metrics.
                        val_vq_loss += vq_loss.cpu().detach().numpy().item() * ecogs.size(0)
                        val_gan_loss += gan_loss.cpu().detach().numpy().item() * ecogs.size(0)
                        
                        for ii in range(len(trace)):
                            val_r += self.get_mean_r(trace[ii].cpu().detach().numpy() ,decoded_images[ii].cpu().detach().numpy())
                        val_size += ecogs.size(0)
                    
            # Record epoch-level losses and mean correlations.
            info = {
                'train_vq_loss': train_vq_loss / train_size,
                'train_gan_loss': train_gan_loss / train_size,
                'train_r': train_r / train_size,

                'val_vq_loss': val_vq_loss / val_size,
                'val_gan_loss': val_gan_loss / val_size,
                'val_r': val_r / val_size}

            for tag, value in info.items():
                self.writer.add_scalar(tag, value, epoch)
            # Save the checkpoint with the highest validation correlation.

            current_val_r = val_r / val_size
            if current_val_r > best_r:
                band = args.band
                if args.is_infer == True:
                    torch.save(self.vqgan.state_dict(), os.path.join(self.checkpoints_path, f"best_vqgan{args.HS}_{args.reading_name}_{args.elec_type}_{band}_infer_{args.used_sound}.pt"))
                elif args.is_percentage == True:
                    torch.save(self.vqgan.state_dict(), os.path.join(self.checkpoints_path, f"best_vqgan{args.HS}_{args.reading_name}_{args.elec_type}_{band}_{args.fold_ind}_percentage_{args.percent}.pt"))
                elif args.is_fold == True:
                    torch.save(self.vqgan.state_dict(), os.path.join(self.checkpoints_path, f"best_vqgan{args.HS}_{args.reading_name}_{args.elec_type}_{band}_{args.fold_ind}.pt"))
                else:
                    torch.save(self.vqgan.state_dict(), os.path.join(self.checkpoints_path, f"best_vqgan{args.HS}_{args.reading_name}_{args.elec_type}_{band}.pt"))
                best_r = current_val_r

            if epoch==79:
                band = args.band 
                if args.is_infer == True:
                    torch.save(self.vqgan.state_dict(), os.path.join(self.checkpoints_path, f"vqgan{args.HS}_epoch_{epoch}_{args.reading_name}_{args.elec_type}_{band}_infer_{args.used_sound}.pt"))
                elif args.is_percentage == True:
                    torch.save(self.vqgan.state_dict(), os.path.join(self.checkpoints_path, f"vqgan{args.HS}_epoch_{epoch}_{args.reading_name}_{args.elec_type}_{band}_{args.fold_ind}_percentage_{args.percent}.pt"))
                elif args.is_fold == True:
                    torch.save(self.vqgan.state_dict(), os.path.join(self.checkpoints_path, f"vqgan{args.HS}_epoch_{epoch}_{args.reading_name}_{args.elec_type}_{band}_{args.fold_ind}.pt"))
                else:
                    torch.save(self.vqgan.state_dict(), os.path.join(self.checkpoints_path, f"vqgan{args.HS}_epoch_{epoch}_{args.reading_name}_{args.elec_type}_{band}.pt"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN Training Script")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension of VQGAN bottleneck (default: 256)')
    parser.add_argument('--HS', type=int, default=54, help='Subject number (default: 54)')
    parser.add_argument('--time-length', type=int, default=192, help='Input time length for each sample (default: 192)')
    parser.add_argument('--num-codebook-vectors', type=int, default=256, help='Number of codebook vectors in VQGAN (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss coefficient for VQGAN (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels in input images/traces (default: 1)')
    parser.add_argument('--dataset-path', type=Path, default=None, help='Dataset directory (default: workspace/model_data/dataset_for_decoding_trace)')
    parser.add_argument('--device', type=str, default="cuda:1", help='CUDA device for training, e.g. "cuda:0"')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=80, help='Total number of training epochs (default: 80)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate for optimizer (default: 2.25e-5)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam optimizer beta1 parameter (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam optimizer beta2 parameter (default: 0.9)')
    parser.add_argument('--disc-start', type=int, default=10000, help='Iteration to start discriminator training (default: 10000)')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for discriminator loss (default: 1.0)')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weight for reconstruction loss (default: 1.0)')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weight for perceptual loss (default: 1.0)')
    parser.add_argument('--lastdim', type=float, default=1., help='Output dimension for encoder (default: 1.0)')
    parser.add_argument('--elec_type', type=str, default="all_elecs", help='Electrode type: "all_elecs", "covert_sig", etc.')
    parser.add_argument('--reading_name', type=str, default="covert", help='Reading type: "covert" or "overt"')
    parser.add_argument('--is-infer', '--is_infer', dest='is_infer', action=argparse.BooleanOptionalAction, default=False, help='Run the held-sound inference experiment')
    parser.add_argument('--used_sound', type=str, default='ba', help='Sound used for inference (default: "ba")')
    parser.add_argument('--is-fold', '--is_fold', dest='is_fold', action=argparse.BooleanOptionalAction, default=True, help='Run fold-indexed cross-validation (default: enabled)')
    parser.add_argument('--fold_ind', type=int, default=1, help='Fold index for cross-validation (default: 1)')
    parser.add_argument('--band', type=str, default='high gamma', help='Frequency band, e.g. "high gamma"')
    parser.add_argument('--is-percentage', '--is_percentage', dest='is_percentage', action=argparse.BooleanOptionalAction, default=False, help='Run the reduced-training-percentage experiment')
    parser.add_argument('--percent', type=float, default=1.0, help='Fraction of training data used in the percentage experiment (default: 1.0)')

    args = parser.parse_args()
    if args.dataset_path is None:
        args.dataset_path = clean_data_path / "dataset_for_decoding_trace"
    args.dataset_path = str(args.dataset_path.expanduser().resolve())
    
    args.fp16=False

    band= args.band
    last_dim = np.load(Path(args.dataset_path) / f"last_dim_{band}.npy", allow_pickle=True).item()
    check_points_dir = clean_data_path / "checkpoints"

    HS_list =  []
    if args.reading_name=='overt' and args.elec_type in ["covert_sig",'downsample_covert_sig','SI-specific']:
        pass 
    elif args.reading_name=='covert' and args.elec_type in ["overt_sig",'downsample_overt_sig','SA-specific']:
        pass
    else:
        for HS in HS_list:
            args.HS = HS
            if args.elec_type in last_dim:
                if f'HS{HS}' not in last_dim[args.elec_type]:
                    print(f"HS{HS} has no {args.elec_type} electrodes, padding it.")
                    args.lastdim = 1
                else:
                    if args.elec_type == "all_elecs":
                        args.lastdim = 16 if args.band in ['high gamma','beta1'] else 32
                    elif args.elec_type == "downsampled_all_elecs":
                        args.lastdim = 4 if args.band in ['high gamma','beta1'] else 8
                    else:
                        args.lastdim = last_dim[f'{args.elec_type}'][f"HS{args.HS}"]
                    args.is_fold = True if args.is_infer==False else False

            if args.is_percentage == True:
                for fold_ind in range(5):
                    for percent in [0.2,0.4,0.6,0.8]:
                        args.fold_ind = fold_ind
                        args.percent = percent
                        print(f"HS{args.HS}_{args.reading_name}_{args.elec_type}_{args.fold_ind}, last dim:{args.lastdim}")
                        # Skip a configuration when its final checkpoint exists.
                        band = args.band if hasattr(args, 'band') else 'high_gamma'
                        epoch = args.epochs - 1
                        pt_path = os.path.join(check_points_dir, f"vqgan{args.HS}_epoch_{epoch}_{args.reading_name}_{args.elec_type}_{band}_{args.fold_ind}_percentage_{percent}.pt")
                        if os.path.exists(pt_path):
                            print(f"Skip training: {pt_path} already exists.")
                            continue
                        train_vqgan = TrainVQGAN(args)


            elif args.is_fold == True:
                for fold_ind in range(5):
                    args.fold_ind = fold_ind
                    print(f"HS{args.HS}_{args.reading_name}_{args.elec_type}_{args.fold_ind}, last dim:{args.lastdim}")
                    # Skip a configuration when its final checkpoint exists.
                    band = args.band if hasattr(args, 'band') else 'high_gamma'
                    epoch = args.epochs - 1
                    if args.is_infer == True:
                        pt_path = os.path.join(check_points_dir, f"vqgan{args.HS}_epoch_{epoch}_{args.reading_name}_{args.elec_type}_{band}_infer_{args.used_sound}.pt")
                    elif args.is_fold == True:
                        pt_path = os.path.join(check_points_dir, f"vqgan{args.HS}_epoch_{epoch}_{args.reading_name}_{args.elec_type}_{band}_{args.fold_ind}.pt")
                    else:
                        pt_path = os.path.join(check_points_dir, f"vqgan{args.HS}_epoch_{epoch}_{args.reading_name}_{args.elec_type}_{band}.pt")
                    if os.path.exists(pt_path):
                        print(f"Skip training: {pt_path} already exists.")
                        continue

                    train_vqgan = TrainVQGAN(args)
            elif args.is_infer == True:
                if HS < 70:
                    all_sound_list =['ba','da','ga','bu','du','gu']
                    two_sound_list = [('ba','bu'),('da','du'),('ga','gu')]
                    tri_sound_list = [('ba','da','ga'),('bu','du','gu')]
                else:
                    all_sound_list = ['ba','da','ga','pa','ka','ta','sha','sa']
                    two_sound_list = [('ba','pa'),('da','ta'),('ga','ka'),('sha','sa')]
                    tri_sound_list = [('ba','da','ga'),('pa','ta','ka')]


                for sound_used in all_sound_list:
                    args.used_sound = sound_used
                    print(f"HS{args.HS}_{args.reading_name}_{args.elec_type}_infer_{args.used_sound}, last dim:{args.lastdim}")
                    band = args.band if hasattr(args, 'band') else 'high_gamma'
                    epoch = args.epochs - 1
                    pt_path = os.path.join(check_points_dir, f"vqgan{args.HS}_epoch_{epoch}_{args.reading_name}_{args.elec_type}_{band}_infer_{args.used_sound}.pt")
                    if os.path.exists(pt_path):
                        print(f"Skip training: {pt_path} already exists.")
                        continue
                    train_vqgan = TrainVQGAN(args)

                for sound_used in two_sound_list:
                    args.used_sound = sound_used
                    print(f"HS{args.HS}_{args.reading_name}_{args.elec_type}_infer_{args.used_sound}, last dim:{args.lastdim}")
                    epoch = args.epochs - 1
                    pt_path = os.path.join(check_points_dir, f"vqgan{args.HS}_epoch_{epoch}_{args.reading_name}_{args.elec_type}_{band}_infer_{args.used_sound}.pt")
                    if os.path.exists(pt_path):
                        print(f"Skip training: {pt_path} already exists.")
                        continue
                    train_vqgan = TrainVQGAN(args)

                for sound_used in tri_sound_list:
                    args.used_sound = sound_used
                    print(f"HS{args.HS}_{args.reading_name}_{args.elec_type}_infer_{args.used_sound}, last dim:{args.lastdim}")
                    epoch = args.epochs - 1
                    pt_path = os.path.join(check_points_dir, f"vqgan{args.HS}_epoch_{epoch}_{args.reading_name}_{args.elec_type}_{band}_infer_{args.used_sound}.pt")
                    if os.path.exists(pt_path):
                        print(f"Skip training: {pt_path} already exists.")
                        continue
                    train_vqgan = TrainVQGAN(args)
