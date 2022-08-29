#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, Optional

import hydra
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchvision import datasets, transforms
import warnings

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def diag(x:torch.Tensor):
    idx = torch.arange(x.shape[-1])
    return x[...,idx,idx]

def trace(x:torch.Tensor):
    return diag(x).sum(-1)

class BentIdentity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, z:torch.Tensor, logdet:torch.Tensor):
        logdet = logdet - torch.log(z/(2*torch.sqrt(z.square()+1)) + 1).sum(-1)
        z = (torch.sqrt((z.square()+1)) - 1)/2 + z
        return z, logdet
    
    def infer(self, z:torch.Tensor):
        return (4*z+2-torch.sqrt(4*(z.square()+z+1)))/3
    
class LieLinear(torch.nn.Module):
    def __init__(self, in_features, expdim, **kwargs) -> None:
        super().__init__()
        self.in_features = in_features
        self.expdim = expdim
        # self.w = torch.nn.parameter.Parameter(
        #     torch.zeros(self.in_features, self.in_features, **kwargs))
        self.v = torch.nn.parameter.Parameter(
            torch.zeros(self.in_features//self.expdim, self.expdim, self.expdim, **kwargs))
        self.b = torch.nn.parameter.Parameter(
            torch.zeros(self.in_features, **kwargs))
        self.inv = None
        self.idx = np.arange(self.in_features)
        np.random.shuffle(self.idx)
        self.reset_parameters()
        
    def forward(self, z, logdet):
        self.inv = None
        # z = z@torch.linalg.inv(self.w)
        z = torch.reshape(z, (-1, self.in_features//self.expdim, 1, self.expdim))
        z = z@torch.matrix_exp(self.v)
        z = torch.reshape(z, (-1, self.in_features))
        # z = z@self.w
        z = z + self.b
        z = z[:,self.idx]
        logdet = logdet - trace(self.v).sum(0)
        
        return z, logdet
    
    def infer(self, z):
        if self.inv is None:
            # self.inv = torch.linalg.inv(self.w)
            self.inv = torch.eye(self.in_features, device=self.v.device)
            self.inv = torch.reshape(self.inv, (self.in_features, self.in_features//self.expdim, 1, self.expdim))
            self.inv = self.inv@torch.matrix_exp(-self.v)
            self.inv = torch.reshape(self.inv, (self.in_features, self.in_features))
            # self.inv = self.inv@self.w
            
        return (z[:,self.idx.argsort()] - self.b)@self.inv
    
    def reset_parameters(self):
        # torch.nn.init.normal_(self.w, std=np.sqrt(1./self.in_features))
        torch.nn.init.normal_(self.v, std=np.sqrt(1./self.in_features))
        torch.nn.init.zeros_(self.b)
    
class LieFlow(torch.nn.Module):
    def __init__(self, input_shape, expdim, layersize, **kwargs) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.in_features = np.array(input_shape).prod()
        self.layersize = layersize
        self.layers = torch.nn.ModuleList([LieLinear(self.in_features, expdim, **kwargs)])
        for _ in range(self.layersize):
            self.layers.append(BentIdentity())
            self.layers.append(LieLinear(self.in_features, expdim, **kwargs))
        
    def forward(self, data):
        z = torch.reshape(data, (-1, self.in_features))
        logdet = torch.zeros(data.shape[0], device=z.device)
        for i,layer in enumerate(self.layers):
            z, logdet = layer(z, logdet)
            
        return z, logdet     
    
    def infer(self, z):
        for i,layer in enumerate(reversed(self.layers)):
            z = layer.infer(z)
        
        data = torch.reshape(z, (-1,)+self.input_shape)
        return data

class LieFlowModule(pl.LightningModule):
    def __init__(self, input_shape, expdim, layersize, var):
        super().__init__()
        self.model = self.create_model(input_shape, expdim, layersize)
        self.save_hyperparameters({'expdim':expdim,'layersize':layersize}, logger=False)
        self.sigma = np.sqrt(var)
        self.a = 1/(2*var)
        self.b = np.log(2*torch.pi*var)/2
      
    def create_model(self, input_shape, expdim, layersize):
        model = LieFlow(input_shape, expdim, layersize)
        print(model)
        return model
    
    def gaussian_nll(self, z:torch.Tensor):
        return z.square().mean(0).sum()*self.a + self.b
      
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        z, logdet = self.forward(x)
        latent_nll = self.gaussian_nll(z)
        logdet = logdet.mean()
        loss = latent_nll + logdet
        self.log('train/latent_nll', latent_nll, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log('train/logdet', logdet, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        z, logdet = self.forward(x)
        latent_nll = self.gaussian_nll(z)
        logdet = logdet.mean()
        loss = latent_nll + logdet
        self.log('val/latent_nll', latent_nll, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log('val/logdet', logdet, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss
    
    def validation_epoch_end(self, outputs) -> None:
        avg_acc = torch.stack(outputs).mean()
        self.logger.log_hyperparams(self.hparams, metrics={'val/loss':avg_acc})
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
   
class SaveGeneratedData(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        
    def on_save_checkpoint(self, trainer:pl.Trainer, pl_module:LieFlowModule, checkpoint: Dict[str, Any]) -> Optional[dict]:
        z = torch.randn((25,pl_module.model.in_features), device=pl_module.device)*pl_module.sigma
        data = pl_module.model.infer(z)
        fig,ax = plt.subplots(5,5)
        for i in range(5):
            for j in range(5):
                ax[i,j].imshow(data[5*i+j,0].cpu().detach().numpy())
        plt.close(fig)
        trainer.logger.experiment.add_figure("Generated data", fig, pl_module.current_epoch)
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

@hydra.main(config_path=os.path.join(root, 'conf'),config_name='config.yaml')
def main(cfg):
    warnings.simplefilter('ignore')
    
    transform = transforms.Compose([transforms.ToTensor(),])
    data_dir = os.path.join(root, 'data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(data_dir, train=False, transform=transform)    
    
    callback1 = SaveGeneratedData()
    callback2 = ModelCheckpoint(**cfg.model_checkpoint)

    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=cfg.batch_size,
        num_workers=4)
    
    model = LieFlowModule((1,28,28), cfg.expdim, cfg.layersize, cfg.var)
    
    logger = TensorBoardLogger(**cfg.logger)
    trainer = pl.Trainer(**cfg.trainer, logger=logger, callbacks=[callback1, callback2])
        
    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
