

class HyperParameters(object):
    def __init(self):
        self.num_epochs = 5
        self.criterion =
        self.real_label = 0.8

        self.optimizerD = None
        self.optimizerG = None

        self.shortCP = 50
        self.longCP = 500

        self.testName = "None"
        self.saveFilePref = "../results/"

def train(hps, dataloader, netD, netG):
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(hps.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data['image'].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), hps.real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = hps.criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            hps.optimizerD.step()
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            hps.optimizerG.step()
            
            # Output training stats
            if i % hps.shortCP == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % hps.longCP == 0) or ((epoch == hps.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, nrow=8, padding=2, normalize=True))
                plt.imshow(img_list[-1].permute(1,2,0))
                if ((epoch==num_epochs-1) and (i==len(dataloader)-1)):
                    plt.savefig(hps.saveFilePref + "../" + hps.testName + "_final_" + str(epoch) + "_" + str(iters))
                    plt.draw()
                    plt.pause(0.001)
                else:
                    plt.savefig(hps.saveFilePref + hps.testName +  str(epoch) + "_" + str(iters))
                    plt.draw()
                    plt.pause(0.001)
                
            

            iters += 1

