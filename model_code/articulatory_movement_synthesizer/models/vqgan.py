import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook


class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.args = args
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = Codebook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self._ensure_consistent_dtype()

    def _ensure_consistent_dtype(self):
        """Keep all model parameters in the configured floating-point dtype."""
        if self.args.fp16:
            self.half()
        else:
            self.float()

    def forward(self, ecogs):
        if self.args.fp16 and ecogs.dtype != torch.float16:
            ecogs = ecogs.half()
        elif not self.args.fp16 and ecogs.dtype != torch.float32:
            ecogs = ecogs.float()
        encoded_images = self.encoder(ecogs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        """Calculate the adaptive GAN weight with guarded gradient handling."""
        try:
            last_layer = self.get_last_layer()
            if last_layer is None:
                print("Warning: Last layer not found, returning default lambda=1.0")
                return 1.0
                
            last_layer_weight = last_layer.weight
            
            if not last_layer_weight.requires_grad:
                print("Warning: Last layer weights don't require grad, returning default lambda=1.0")
                return 1.0
            
            perceptual_loss_grads = torch.autograd.grad(
                outputs=perceptual_loss,
                inputs=last_layer_weight,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]
            
            gan_loss_grads = torch.autograd.grad(
                outputs=gan_loss,
                inputs=last_layer_weight,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]
            
            if perceptual_loss_grads is None:
                print("Warning: perceptual_loss_grads is None, returning default lambda=1.0")
                return 1.0
                
            if gan_loss_grads is None:
                print("Warning: gan_loss_grads is None, returning default lambda=1.0")
                return 1.0
            
            if torch.isnan(perceptual_loss_grads).any() or torch.isinf(perceptual_loss_grads).any():
                print("Warning: perceptual_loss_grads contains NaN or inf, returning default lambda=1.0")
                return 1.0
                
            if torch.isnan(gan_loss_grads).any() or torch.isinf(gan_loss_grads).any():
                print("Warning: gan_loss_grads contains NaN or inf, returning default lambda=1.0")
                return 1.0
            
            perceptual_norm = torch.norm(perceptual_loss_grads)
            gan_norm = torch.norm(gan_loss_grads)
            
            if gan_norm.item() == 0:
                print("Warning: gan_norm is zero, returning default lambda=1.0")
                return 1.0
            
            adaptive_weight = perceptual_norm / (gan_norm + 1e-6)
            adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1e4)
            return adaptive_weight.item()
            
        except Exception as e:
            print(f"Error in calculate_lambda: {e}, returning default lambda=1.0")
            return 1.0

    def get_last_layer(self):
        """Return the decoder output layer used for adaptive weighting."""
        try:
            if hasattr(self.decoder, 'model') and len(self.decoder.model) > 0:
                return self.decoder.model[-1]
            elif hasattr(self.decoder, 'layers') and len(self.decoder.layers) > 0:
                return self.decoder.layers[-1]
            elif hasattr(self.decoder, 'conv_out'):
                return self.decoder.conv_out
            else:
                return None
        except Exception:
            return None

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))








