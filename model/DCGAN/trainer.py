import os
import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model_g, model_d, optimizer_g, optimizer_d, noise_dim, loss_func, epochs, device, save_path, logger):
        self.model_g = model_g.to(device)
        self.model_d = model_d.to(device)
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.noise_dim = noise_dim
        self.loss_func = loss_func
        self.epochs = epochs
        self.device = device
        self.save_path = save_path
        self.logger = logger
        self.writer = SummaryWriter(log_dir=save_path)
        self.lowest_g_loss = float('inf')
        self.lowest_d_loss = float('inf')
        self.best_model_g = None
        self.best_model_d = None
    
    def main(self, train_loader, valid_loader, test_loader):
        for epoch in range(self.epochs):
            train_g_loss, train_d_loss = self.train(train_loader)
            self.logger.info(f"Train {epoch + 1}/{self.epochs}:: g_loss: {train_g_loss:.4f} d_loss: {train_d_loss:.4f}")

            valid_g_loss, valid_d_loss = self.valid(valid_loader)
            self.logger.info(f"Valid {epoch + 1}/{self.epochs}:: g_loss: {valid_g_loss:.4f} d_loss: {valid_d_loss:.4f}")

            self.writer.add_scalars('Losses', {
                'Train_G_Loss': train_g_loss,
                'Train_D_Loss': train_d_loss,
                'Valid_G_Loss': valid_g_loss,
                'Valid_D_Loss': valid_d_loss,
            }, epoch)

            if valid_g_loss < self.lowest_g_loss:
                self.lowest_g_loss = valid_g_loss
                self.lowest_d_loss = valid_d_loss

                self.best_model_g = self.model_g.state_dict()
                self.best_model_d = self.model_d.state_dict()
                torch.save(self.best_model_g, os.path.join(self.save_path, 'best_model_g.pth'))
                torch.save(self.best_model_d, os.path.join(self.save_path, 'best_model_d.pth'))

            if (epoch + 1) % (self.epochs // 10) == 0 or epoch == self.epochs - 1:
                test_g_loss, test_d_loss = self.test(test_loader)
                self.logger.info(f"Test {epoch + 1}/{self.epochs}:: g_loss: {test_g_loss:.4f} d_loss: {test_d_loss:.4f}")

                self.writer.add_scalars('Test Losses', {
                    'Test_G_Loss': test_g_loss,
                    'Test_D_Loss': test_d_loss,
                }, epoch)

            self.logger.info(f"Current Best Loss:: g_loss: {self.lowest_g_loss:.4f} d_loss: {self.lowest_d_loss:.4f}")

    def train(self, train_loader):
        g_losses = 0.0
        d_losses = 0.0
        for (x, _) in train_loader:
            x = x.to(self.device)

            batch_size = x.size(0)
            """
            train discriminator
            """
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)

            self.optimizer_d.zero_grad()
            real_loss = self.loss_func(self.model_d(x), real_labels)

            z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
            fake_x = self.model_g(z)
            fake_loss = self.loss_func(self.model_d(fake_x), fake_labels)

            d_loss = real_loss + fake_loss
            d_losses += d_loss.item()
            d_loss.backward()

            self.optimizer_d.step()
            """
            train generator
            """
            self.optimizer_g.zero_grad()

            z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
            fake_x = self.model_g(z)
            g_loss = self.loss_func(self.model_d(fake_x), real_labels)
            g_losses += g_loss.item()
            g_loss.backward()

            self.optimizer_g.step()

        return g_losses / len(train_loader), d_losses / len(train_loader)

    def valid(self, valid_loader):
        g_losses = 0.0
        d_losses = 0.0
        with torch.no_grad():
            for (x, _) in valid_loader:
                x = x.to(self.device)

                batch_size = x.size(0)                
                """
                valid discriminator
                """
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                real_loss = self.loss_func(self.model_d(x), real_labels)

                z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
                fake_x = self.model_g(z)
                fake_loss = self.loss_func(self.model_d(fake_x), fake_labels)

                d_loss = real_loss + fake_loss
                d_losses += d_loss.item()
                """
                valid generator
                """
                z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
                fake_x = self.model_g(z)
                g_loss = self.loss_func(self.model_d(fake_x), real_labels)
                g_losses += g_loss.item()

        return g_losses / len(valid_loader), d_losses / len(valid_loader)

    def test(self, test_loader):
        g_losses = 0.0
        d_losses = 0.0
        with torch.no_grad():
            for (x, _) in test_loader:
                x = x.to(self.device)

                batch_size = x.size(0)
                """
                test discriminator
                """
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                real_loss = self.loss_func(self.model_d(x), real_labels)

                z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
                fake_x = self.model_g(z)
                fake_loss = self.loss_func(self.model_d(fake_x), fake_labels)

                d_loss = real_loss + fake_loss
                d_losses += d_loss.item()
                """
                test generator
                """
                z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
                fake_x = self.model_g(z)
                g_loss = self.loss_func(self.model_d(fake_x), real_labels)
                g_losses += g_loss.item()

        return g_losses / len(test_loader), d_losses / len(test_loader)
    