# Codes for Rebuttal of paper 1972
We provide:
The simplified training and inference codes of LatentHSI.
The comparison between LatentHSI, PLRDiff and HIRDiff.

## 1 LatentHSI

### 1.1 Training

#### 1.1.1 Band selection

We can use any number of bands (i.e. s).

``` python
def get_idxs(Channels,s):
    idxs=torch.linspace(0,Channels-1,s)
    idxs=torch.round(idxs).to(torch.int32)
    return idxs
    # example: print(get_idxs(128,8))
    # tensor([  0,  18,  36,  54,  73,  91, 109, 127], dtype=torch.int32)
```

#### 1.1.2 Training of VAE

VAE is trained in a self-supervised manner, using only the clean HSIs without the information of Degraded HSI. 

clean_HSI stands for $\mathcal{X}$ and reduced_HSI stands for $\mathcal{A}$ in the main paper.

``` python 
# simplified training code of VAE
import torch 
import torch.nn.functional as nF

VAE.kl_weight=1e-6
VAE=VAE.cuda()
s=50 # this means we only use 50 bands for any datasets
optimizer=optim.Adam(VAE.parameters(),lr=1e-4)

VAE=99
VAE.kl_weight=1e-6
device=torch.device("cuda:0")

VAE=VAE.to(device)
s=50 # this means we only use 50 bands for any datasets
optimizer=torch.optim.Adam(VAE.parameters(),lr=1e-4)

def training_step(VAE,inputs):
    # Eq.(18) in the main paper
    # inputs: \mathcal{A} [B,s,H,W]
    reconstruction, posterior = VAE(inputs)
    rec_loss=nF.mse_loss(inputs,reconstruction)
    kl_loss=posterior.kl()
    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
    return rec_loss+VAE.kl_weight*kl_loss,rec_loss

for epoch in range(epochs):
    total_loss=0.
    rec_total_loss=0.
    VAE.train()

    for clean_HSI in dataloader: # dataloader returns clean HSIs(\mathcal{X}), [B,C,H,W]
        Channels=clean_HSI.shape[1]
        idxs=get_idxs(Channels,s)
        reduced_HSI=clean_HSI[:,idxs,:,:] #we get the reduced HSI via bands selection from clean HSI
        loss,rec_loss=training_step(VAE,reduced_HSI)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### 1.1.3 Training of Diffuion Model (UNet)

We only use clean HSIs to train the UNet. 

``` python
# simplified training code of Diffusion Model
# diffusion stands for the diffusion model, containing a UNet
optimizer=torch.optim.Adam(diffusion.parameters(),lr=1e-4)

def get_losses(diffusion, z0):
    t = torch.randint(0, diffusion.num_timesteps, (z0.shape[0],))
    noise = torch.randn_like(z0)
    zt = perturb_x(z0, t, noise)
    estimated_noise = diffusion.UNet(zt, t)
    loss = nF.mse_loss(estimated_noise, noise)
    return loss

# training
for ep in range(epochs):
    for clean_HSI in dataloader: # clean_HSI: [B,C,H,W]
        Channels=clean_HSI.shape[1] 
        idxs=get_idxs(Channels,s) 
        reduced_HSI=clean_HSI[:,idxs,:,:] # gets \mathcal{A}, [B,s,H,W]
        
        ### our diffusion model is trained in the latent space

        latent_representation=VAE.encode(reduced_HSI) #z0 = Encoder(\mathcal{A}) [B,c,h,w]
        loss = get_losses(diffusion,latent_representation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# utils 
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def perturb_x(diffusion, x, t, noise):
    return (
        extract(diffusion.sqrt_alphas_cumprod, t, x.shape) * x +
        extract(diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise)    
```

### 1.2 Inference

#### 1.2.1 Get the Coefficient Matrix $E$

``` python
def get_E(img,s):
    # Eq.(17) in the main paper 
    # the shape of img should be B,C,H,W 
    # the shape of E will be B,C,s
    device=img.device 
    B,C,H,W=img.shape
    idxs=get_idxs(C,s).to(device)
    bimg = img[:,idxs,:,:].reshape(B, s, -1) 
    # estimate coefficient matrix E by solving least square problem.
    # The least square problem has an explicit solution, so we don't need update iterations.
    t1 = torch.matmul(bimg, bimg.transpose(1,2)) + 1e-4*torch.eye(s,device=device).type(bimg.dtype) # For numerical stability, ensuring t1 is invertible.

    t2 = torch.matmul(img.reshape(B, C, -1), bimg.transpose(1,2))
    E = torch.matmul(t2, torch.inverse(t1))
    return E 
```

#### 1.2.2 Restoration Process

We conduct the reverse sampling process in the latent space, and use a final correction with Adam to make the sampling more stable.

``` python
from sampling_utils import loss_tv # TV Loss function 
import torch 
import torch.nn.functional as nF
import torch.nn as nn

def sample_ddim(self,batch_size,s,coder,degraded_HSI,correction_times,sampling_timesteps,D_func,eta1,eta2):
    # self: the Diffusoin Model
    # coder: the Variational Autoencoder 
    # s: number of used bands
    # D_func degradation function. For denoising is Identity mapping, for SR is downsampling.
    # correction times: The number of times z is updated with Adam
    print("sample with ddim...")
    # initial value, get z_T from standard Gaussian Distribution
    z = torch.randn(batch_size, self.c, self.h, self.w) 
	
    if sampling_timesteps is None:
        sampling_timesteps = self.num_timesteps
    # get time steps
    times = torch.linspace(-1, self.num_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    # We get E directly
    E=get_E(degraded_HSI,s) 
    # sampling starts
    for time, time_next in time_pairs:
        if time_next < 0 : # time_next < 0 means that we already get z_0
            Pa=para(nn.Parameter(z.requires_grad_(True)))
            optimizer=torch.optim.Adam(Pa.parameters(),lr=0.2)
            # final correction with Adam, to make the sampling more stable
            for _ in range(correction_times):
                estimation=coder.decode(Pa.z)
                estimation=res_from_E(estimation,E)
		# Eq.(24) in the main paper, cauculate the gradient
                loss1=nF.mse_loss(D_func(estimation),degraded_HSI)
                loss2=loss_tv(estimation)
                loss=eta1*loss1+eta2*loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            answer_reduced=coder.decode(Pa.z) # \hat{\mathcal{A}} = Decoder(\hat{z_0})
            answer_HSI=res_from_E(estimation,E) # \hat{\mathcal{X}} = \hat{\mathcal{A}} \otimes E
            return answer_HSI
        else :
            with torch.no_grad():
            # DDIM sampling 
                time_cond = torch.full((batch_size,), time, device = device, dtype = torch.long)
                pred_noise, z_0 = self.model_predictions(z, time_cond)
                z_0 = self.predict_start_from_noise(z, time_cond, pred_noise)
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                #sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next).sqrt()

            # Update \hat{z_0} 
            z_0=z_0.requires_grad_(True)
            estimation=coder.decode(z_0) # \hat{\mathcal{A}} = Decoder(\hat{z_0})
            estimation=res_from_E(estimation,E) # \hat{\mathcal{X}} = \hat{\mathcal{A}} \otimes E
            loss1=nF.mse_loss(D_func(estimation),degraded_HSI)
            loss2=loss_tv(estimation)
            loss=eta1*loss1+eta2*loss2
            grad=torch.autograd.grad(loss,z_0)[0]
            with torch.no_grad():
                z_0-=grad
                z = z_0 * alpha_next.sqrt() + c * pred_noise # get z_{t-1} from z_0 and \epsilon_{\theta} 
                


 # utils 
def res_from_E(img,E):
    # return img \otimes E 
    # the shape of img should be B,s,H,W
    # E:B,C,s
    # Return:B,C,H,W 
    B,s,H,W=img.shape
    B,C,s=E.shape
    img=img.reshape(B,s,-1)
    img=E@img 
    img=img.reshape(B,C,H,W)
    return img 

class para(nn.Module):
    def __init__(self,img):
        super().__init__()
        self.img=img 
    
    def forward(self):
        return self.img
```



## 2 PLRDiff

### 2.1 Training

PLRDiff uses a pretrained Diffusion Model, addtional training on the pretrained model can lead to a decrease in performance. So it is not trainable.

### 2.2 Inference

#### 2.2.1 Bands selection

PLRDiff only uses the information of 3 bands.
``` python
# PLRDiff selects 3 bands at equal intervals. Ch means the number of channels, and Rr stands for the rank. In PLRDiff Rr is set to 3.
inters = int((Ch+1)/(Rr+1)) # interval
selected_bands = [(t+1)*inters-1 for t in range(Rr)]
param['Band'] = torch.Tensor(selected_bands).type(torch.int).to(device)  
```
#### 2.2.2 Restoration Process

PLRDiff only works for Pansharpening. It directly samples in the pixel space and only use the pristine Gradient Descent.
``` python
for i in timesteps: 
    t = torch.tensor(i * shape[0], device=device)
    # re-instantiate requires_grad for backpropagation
    img = img.requires_grad_()
    # sample A_{t-1} from p(A_{t-1}|A_t)
    out = self.ddpm_sample(
            model,
            img,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn
        )
    baseA = (out["pred_xstart"] +1)/2  # base tensor A = \hat{A}_0 (img ranges in [-1,1]. here we move out["pred_xstart"] back to range [0,1])
    xhat_1 = torch.matmul(E, baseA.reshape(Bb, Rr, -1)).reshape(*shape) + add_res  # \hat{X}_0 = Ax_3 E + R
    xhat_2 = blur(xhat_1) 
    xhat_3 = down(xhat_2) # D(B(\hat{X}))
    norm1 = torch.norm(LRHS - xhat_3) # ||Y - D(B(Ax_3 E + R))||
    xhat_4 = torch.matmul((xhat_1).permute(0,2,3,1), param["PH"]).permute(0,3,1,2) # \hat{X}_3 r
    norm2 = torch.norm(PAN - xhat_4) # ||P - \hat{X}_3 r||
    # get the gradient
    likelihood = norm1 + (param['eta2']/param['eta1'])*norm2
    norm_gradX = grad(outputs=likelihood, inputs=img)[0] 
    # Gradient Descent
    out["sample"] = out["sample"] - param['eta1']*norm_gradX 

    yield out, E, add_res
    img = out["sample"]

```

## 3 HIRDiff

### 3.1 Training

HIRDiff uses the same pretrained Diffusion Model as PLRDiff and is not trainable.

### 3.2 Inference

#### 3.2.1 Bands selection

HIRDiff uses a RRQR algorithm to choose 3 bands, which is essentially the same as PLRDiff. It only uses the information of 3 bands.
```python
if not opt['no_rrqr']:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.cd(r'matlab')
        res = eng.sRRQR_rank(E[0].cpu().numpy().T, 1.2, 3, nargout=3)
        param['Band'] = torch.Tensor(np.sort(list(res[-1][0][:3]))).type(torch.int).to(device)-1
```
#### 3.2.2 Restoration Process

HIRDiff directly samples in the pixel space. Despite using DDIM and a TV regularization, the rest is the same as PLRDiff.

```python
for iteration, (i, j) in pbar: # this means the time steps
        t = torch.tensor([i] * shape[0], device=device)
        t_next = torch.tensor([j] * shape[0], device=device)
        # re-instantiate requires_grad for backpropagation
        img = img.requires_grad_()

        x, eps = img, 1e-9
        B = x.shape[0]

        alphas_bar = torch.FloatTensor([self.alphas_cumprod_prev[int(t.item()) + 1]]).repeat(B, 1).to(x.device)
        alphas_bar_next = torch.FloatTensor([self.alphas_cumprod_prev[int(t_next.item()) + 1]]).repeat(B, 1).to(x.device)
        # DDIM: Algorithm 1 in the paper
        model_output = model(x, alphas_bar)
        pred_xstart = (x - model_output * (1 - alphas_bar).sqrt()) / alphas_bar.sqrt()
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        # update
        xhat = (pred_xstart + 1) / 2
        xhat = torch.matmul(E, xhat.reshape(Bb, Rr, -1)).reshape(*shape)
        # parameters
        eta = 0
        c1 = (
                eta * (
                (1 - alphas_bar / alphas_bar_next) * (1 - alphas_bar_next) / (1 - alphas_bar + eps)).sqrt()
        )
        c2 = ((1 - alphas_bar_next) - c1 ** 2).sqrt()
        xt_next = alphas_bar_next.sqrt() * pred_xstart + c1 * torch.randn_like(x) + c2 * model_output

        param['iteration'] = iteration
        if param['task'] == 'sr': # the TV Regularization is contained in these loss functions
            loss_condition = self.loss_sr(param, model_condition, xhat)
        elif param['task'] == 'denoise':
            loss_condition = self.loss_denoise(param, model_condition, xhat)
        elif param['task'] == 'inpainting':
            loss_condition = self.loss_inpainting(param, model_condition, xhat)
        else:
            raise ValueError('invalid task name')
		# Gradient Descent update
        norm_gradX = grad(outputs=loss_condition, inputs=img)[0]
        xt_next = xt_next - norm_gradX

        out = {"sample": xt_next, "pred_xstart": pred_xstart}
        yield out, E
```


