## DDPM consists basic UNet, GaussianDiffusion, and a few other components.
## Test on stanford cars dataset.

# Brief Recap of Diffusion Process
# 실제 물리현상에서의 확산은 무작위적인 입자의 이동을 통해 일어난다. -> 이는 다음 바로 직후의 입자의 위치를 가우시안에서 추출한다라고 생각할 수도 있다. 
# 그리고 이러한 과정의 역과정 또한 가우시안에서 추출하는 형태라는 것은 수학적으로 증명되어있다.
# 이런 노이즈를 주입하는 과정상의 한 스텝은 q(x_t|x_t-1) 로 표현할 수 있는데, 이는 결국 이전 상태가 주어졌을 때 가우시안 노이즈가 더해진 상태를 나타내는 확률분포이고
# 이는 다시말하면 그냥 Gaussian Distribution이다.
# 이 때 가우시안을 얼마나 주입할지를 Beta로 결정하고 NCSN에서 점점 큰 노이즈를 넣는 이유를 수학적으로 설명한다.
# 앞서 이 역과정 또한 가우시안이라고 증명되어있다고 하였으나, 가우시안 분포라는 것은 알지만 실제 어떤 Gaussian인지는 Forward와 다르게 알 수가 없으므로
# 결국 이 Reverse 과정에서의 가우시안을 찾기위해 딥러닝으로 푸는 것이다. 
# 논문은 결국 이런 가우시안을 찾는 방법론들을 위한 방식을 제시하는 것이다. (이런 샘플링 기반 딥러닝론에서 역전파를 위해 사용하는 Reparemeterization Trick, ELBO, 수식 정리 등을 사용하는 것이다.)

from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger

from torchvision import transforms
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelSummary
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange 
import math
import wandb
import os

def show_raw_images(dataset, num_samples=8, cols=4):
    plt.figure(figsize=(8,5))
    plt.axis('off')
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples / cols)+1, cols, i+1)
        plt.imshow(img[0])
    plt.show()

def load_dataset():    
    data_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) -1) # Tensor가 0~1 사이 값을 가지는데, -1 ~ 1로 확장해줌
    ])
    
    train_dataset = torchvision.datasets.StanfordCars(root='/Users/junwankim/server/ddpm/', download=False, transform=data_transforms)
    test_dataset = torchvision.datasets.StanfordCars(root='/Users/junwankim/server/ddpm/', download=False, split='test', transform=data_transforms)
    
    return torch.utils.data.ConcatDataset([train_dataset, test_dataset]) # Trainig diffusion does not need test datasets


# Forward Diffusion : q(X_{1:T}|X_0) := ∏ᵀₜ₌₁ q(Xₜ |Xₜ₋₁ ), where q(X_t|X_t-1) = N(X_t; \sqrt{1-\beta_t}X_t-1, beta_t⋅ I) 
# by reparametrization trick above function goes to sqrt(1-beta_t)X_t-1 + sqrt(beta_t) * epsilon_t01 where epsilon_t-1 ~ N(0, I)
# Forward Process는 임의의 t 스텝까지 한번에 갈 수 있고 이는 논문 수식 4를 참고하면 된다. 
class ForwardDiffusion:
    def __init__(self, T=300):
        betas = self.linear_beta_schedule(timesteps=T)
        self.T = T
        alphas = 1. - betas     # αₜ := 1 - βₜ
        alphas_cumprod = torch.cumprod(alphas, dim=0) # ᾱₜ := ∏ᵗₖ₌₁ αₖ
        self.alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # √ᾱₜ
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod) # 1-√ᾱₜ
        
        # self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Shared Terms
    # 논문 14Page 참고, we chose a linear schedule from β_1=10^−4 to β_T=0.02
    def linear_beta_schedule(self, timesteps, start=1e-4, end=0.02):
        return torch.linspace(start, end, timesteps)
    
    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        # return out.reshape(batch_size, *((1,) * (len(x_shape)-1))).to(self.device) # dimension 맟추기용
        return out.reshape(batch_size, *((1,) * (len(x_shape)-1)))
    
    def get_step_t(self, x_0, t): # t에 따른 noise 계산
        noise = torch.randn_like(x_0).to(self.device)
        # noise = torch.randn_like(x_0).to(self.device)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)    
        
        # sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.to(self.device)
        # sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(self.device)
        # x_0 = x_0.to(self.device)
        # noise = noise.to(self.device)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t
        x_0 = x_0
        noise = noise
        
        
        # q(xₜ|x₀) = N(xₜ; √ᾱₜx₀, √(1-ᾱₜ)I) 
        #Reparameterization Trick
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    def show_step_t(self, image, sample_steps):
        stepsize = int(self.T / sample_steps)
        plt.figure(figsize=(10, 3))
        plt.axis('off')
        
        reverse_transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.ToPILImage()
        ])
            
        
        for idx in range(0, self.T, stepsize):
            plt.subplot(1, sample_steps+1, int(idx/stepsize)+1)
            t = torch.Tensor([idx]).type(torch.int64)
            img, _ = self.get_step_t(image, t)
            
            if len(img.shape) == 4:
                img = img[0]
            
            plt.imshow(reverse_transform(img))
        
        plt.show()


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        if up:
            self.conv1 = nn.Conv2d(2*in_channels, out_channels, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1) # Stide 2 Kernel 2?
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x, t):
        h = self.conv1(x)
        h = self.relu(h)
        h = self.bnorm1(h)  # BxOxHxW
        
        time_emb = self.relu(self.time_mlp(t))    # BxO
        time_emb = time_emb[(...,) + (None,) * 2] # BxOxHxW
        h = h + time_emb
        
        h = self.conv2(h)
        h = self.relu(h)
        h = self.bnorm2(h)
        
        return self.transform(h)   

class SinusoidalPositionEmbeddings(nn.Module):
    # Basic Sinusoidal Position Embeddings
    # f(t)^i = sin(w_k * t) for i = 2k if i = 2k, cos(w_k * t) if i = 2k+1
    # w_k = 1 / 10000^(2k/d) = exp(-2k * log(10000) / d)
    def __init__(self, dim):
        super().__init__()
        self.dim = dim 
    
    def forward(self, t):
        half_dim = self.dim // 2
        pe = torch.zeros(t.shape[0], self.dim)
        embeddings = math.log(10000) / half_dim
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        pe[:, 0::2] = embeddings.sin()
        pe[:, 1::2] = embeddings.cos()
        
        return pe

class UNet(pl.LightningModule):
    def __init__(self, T=300, batch_size=128):
        super().__init__()
        in_channels = 3
        self. T = T
        self.batch_size = batch_size
        down_channels = [64, 128, 256, 512, 1024]
        up_channels = [1024, 512, 256, 128, 64]
        out_channels = 3
        time_emb_dim = 32
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )
        
        self.forward_process = ForwardDiffusion(T=T)
        
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels) - 1)])
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)])
        self.output = nn.Conv2d(up_channels[-1], out_channels, 1)
    
    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
    
    def training_step(self, batch):
        t = torch.randint(0, self.T, (self.batch_size,)).long()
        
        with torch.no_grad():
            noised, noise = forward_process.get_step_t(batch, t)
            
        noise_pred = self.forward(noised, t)
        loss = F.l1_loss(noise, noise_pred)
        
        self.log('train_loss', loss)
        self.log('lr', self.lr_schedulers().get_lr()[0], on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return [self.optimizer]

    
if __name__ == '__main__':
    
    show_forward = False
    
    if show_forward:
        # Testing Dataset
        train_dataset = torchvision.datasets.StanfordCars(root='/Users/junwankim/server/ddpm/', download=False)
        show_raw_images(train_dataset)
        
        # Testing Forward Process
        
        train = load_dataset()
        train_loader = DataLoader(train, batch_size=128, shuffle=True, drop_last=True)
        image = next(iter(train_loader))[0]
        
        forward_process = ForwardDiffusion()
        forward_process.show_step_t(image, 10)
        
        # Testing UNet
    
        backward_process = UNet()
        sample_input = torch.randn(10, 3, 64, 64)
        sample_output = backward_process(sample_input, torch.Tensor([1]))
        print(sample_output.shape)
    
    
    train_loader = DataLoader(train, batch_size=128, shuffle=True, drop_last=True)
    os.makedirs('./results', exist_ok=True)
    wandb.init(project='DDPM', name='DDPM')
    
    print("Start Training")
    
    model = UNet(T=300, batch_size=128)
    wandb.finish()
    
    wandb_logger = WandbLogger(log_model="all", project='DDPM', name='lightning')
    trainer = pl.Trainer(accelerator='gpu', logger=wandb_logger, max_epochs=100)
    trainer.fit(model= model, train_dataloaders=train_loader)
    
    
    
    
    
    