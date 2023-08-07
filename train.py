import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Horse2ZebraDataset
from utils import save_checkpoint, load_checkpoint
from generator_model import Generator
from discriminator_model import Discriminator
import os
import config
from torchvision.utils import save_image
from time import time

def train_fn(discHorse, discZebra, genHorse, genZebra, loader, optDiscriminator, optGenerator, l1, mse, dScalar, gScalar):
    
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        with torch.cuda.amp.autocast_mode.autocast():
            fakeHorse = genHorse(zebra) #generate fake horse starting from zebra
            
            discRealHorseOutput = discHorse(horse)
            discFakeHorseOutput = discHorse(fakeHorse.detach())

            discRealHorseLoss = mse(discRealHorseOutput, torch.ones_like(discRealHorseOutput))
            discFakeHorseLoss = mse(discFakeHorseOutput, torch.zeros_like(discFakeHorseOutput))
            discHorseLoss = discRealHorseLoss + discFakeHorseLoss

            

            fakeZebra = genZebra(horse)

            discRealZebraOutput = discZebra(zebra)
            discFakeZebraOutput = discZebra(fakeZebra.detach())

            discRealZebraLoss = mse(discRealZebraOutput, torch.ones_like(discRealZebraOutput))
            discFakeZebraLoss = mse(discFakeZebraOutput, torch.ones_like(discFakeZebraOutput))

            discZebraLoss = discFakeZebraLoss + discRealZebraLoss

            discLoss = discZebraLoss + discHorseLoss / 2

        optDiscriminator.zero_grad()
        dScalar.scale(discLoss).backward()
        dScalar.step(optDiscriminator)
        dScalar.update() 
        



        with torch.cuda.amp.autocast_mode.autocast():
            #adversarial loss
            discFakeHorse = discHorse(fakeHorse)
            discFakeZebra = discZebra(fakeZebra)

            genHorseLoss = mse(discFakeHorse, torch.ones_like(discFakeHorse))
            genZebraLoss = mse(discFakeZebra, torch.ones_like(discFakeZebra))


            #cycle loss
            cycleZebra = genZebra(fakeHorse)
            cycleHorse = genHorse(fakeZebra)

            cycleZebraLoss = l1(zebra, cycleZebra)
            cycleHorseLoss = l1(horse, cycleHorse)


            #identity loss
            identityZebra = genZebra(zebra)
            identityHorse = genHorse(horse)
            
            identityZebraLoss = l1(zebra, identityZebra)
            identityHorseLoss = l1(horse, identityHorse)

            ganLoss = genHorseLoss + genZebraLoss + cycleZebraLoss * config.LAMBDA_CYCLE + cycleHorseLoss * config.LAMBDA_CYCLE + identityZebraLoss * config.LAMBDA_IDENTITY + identityHorseLoss * config.LAMBDA_IDENTITY

        optGenerator.zero_grad()
        gScalar.scale(ganLoss).backward()
        gScalar.step(optGenerator)
        gScalar.update()

        if idx % 200 == 0:
            save_image(fakeHorse * 0.5 + 0.5, f"./results/{time()}_horse_{idx}.png")
            save_image(fakeZebra * 0.5 + 0.5, f"./results/{time()}_zebra_{idx}.png")
            

        loop.set_postfix(
            D_H = discHorseLoss.item(),
            D_Z = discZebraLoss.item(),
            G = ganLoss.item()
        )






def main():
    discHorse = Discriminator().to(config.DEVICE)
    discZebra = Discriminator().to(config.DEVICE)

    genHorse = Generator(imgChannels=3).to(config.DEVICE)
    genZebra = Generator(imgChannels=3).to(config.DEVICE)

    optDiscriminator = optim.Adam(
        list(discZebra.parameters()) + list(discHorse.parameters()),      #Concat of two list
        lr= config.LEARNING_RATE,
        betas = (0.5, 0.999)
    )

    optGenerator = optim.Adam(
        list(genZebra.parameters()) + list(genHorse.parameters()), 
        lr = config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1  = nn.L1Loss()
    MSE = nn.MSELoss()


    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, genHorse, optGenerator, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, genZebra, optGenerator, config.LEARNING_RATE)
        
        load_checkpoint(config.CHECKPOINT_DISC_H, discHorse, optDiscriminator, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_Z, discZebra, optDiscriminator, config.LEARNING_RATE)
        
        
    dataset     = Horse2ZebraDataset(rootHorse= config.TRAIN_DIR + "/horse", rootZebra= config.TRAIN_DIR + "/zebra", transform=config.transforms)

    datasetVal  = Horse2ZebraDataset(rootHorse=config.VAL_DIR + "/horse", rootZebra= config.VAL_DIR + "/zebra", transform=config.transforms)

    trainLoader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    valLoader   = DataLoader(datasetVal, batch_size=1, shuffle=False, pin_memory=True)


    gScalar = torch.cuda.amp.grad_scaler.GradScaler()
    dScalar = torch.cuda.amp.grad_scaler.GradScaler()


    for epoch in range(config.NUM_EPOCHS):
        train_fn(discHorse, discZebra, genHorse, genZebra, trainLoader, optDiscriminator, optGenerator, L1, MSE, dScalar, gScalar)

        if config.SAVE_MODEL:
            save_checkpoint(genHorse, optGenerator, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(genZebra, optGenerator, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(discHorse, optDiscriminator, filename=config.CHECKPOINT_DISC_H)
            save_checkpoint(discZebra, optDiscriminator, filename=config.CHECKPOINT_DISC_Z)


if __name__ == "__main__":
    main()