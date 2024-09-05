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
    
    def main(self, train_loader, valid_loader, test_loader, lambda_gp=10, n_critic=5):
        for epoch in range(self.epochs):
            train_g_loss, train_d_loss = self.train(train_loader, lambda_gp, n_critic)
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

    def train(self, train_loader, lambda_gp, n_critic):        
        g_losses = 0.0
        d_losses = 0.0
        for (x, _) in train_loader:
            x = x.to(self.device)
            
            batch_size = x.size(0)
            """
            train discriminator
            learns more often than its creator (repeat n_critic times)
            """
            for _ in range(n_critic):
                z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
                fake_x = self.model_g(z)

                x_score = self.model_d(x).mean()
                fake_x_score = self.model_d(fake_x).mean()

                d_loss = -(x_score - fake_x_score) # 실제 데이터를 더 높은 점수로 평가하고 가짜 데이터를 더 낮은 점수로 평가하는 방법을 알아보세요.
                gp = self.gradient_penalty(x, fake_x)  # 실제 데이터와 가짜 데이터 간의 그래디언트 페널티 계산
                d_loss += lambda_gp * gp  # 판별기 손실에 그래디언트 페널티 추가

                self.optimizer_d.zero_grad()
                d_losses += d_loss.item()
                d_loss.backward()

                self.optimizer_d.step()
            """
            train discriminator
            """
            self.optimizer_g.zero_grad()

            z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
            fake_x = self.model_g(z)

            g_loss = -self.model_d(fake_x).mean() # 판별자가 가짜 데이터를 진짜로 평가할 확률을 최대화합니다.
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
                z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
                fake_x = self.model_g(z)

                x_score = self.model_d(x).mean()
                fake_x_score = self.model_d(fake_x).mean()

                d_loss = -(x_score - fake_x_score) # 실제 데이터를 더 높은 점수로 평가하고 가짜 데이터를 더 낮은 점수로 평가하는 방법을 알아보세요.
                d_losses += d_loss.item()
                """
                valid generator
                """
                z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
                fake_x = self.model_g(z)

                g_loss = -self.model_d(fake_x).mean() # 판별자가 가짜 데이터를 진짜로 평가할 확률을 최대화합니다.
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
                z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
                fake_x = self.model_g(z)

                x_score = self.model_d(x).mean()
                fake_x_score = self.model_d(fake_x).mean()

                d_loss = -(x_score - fake_x_score) # 실제 데이터를 더 높은 점수로 평가하고 가짜 데이터를 더 낮은 점수로 평가하는 방법을 알아보세요.
                d_losses += d_loss.item()
                """
                test generator
                """
                z = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
                fake_x = self.model_g(z)

                g_loss = -self.model_d(fake_x).mean() # 판별자가 가짜 데이터를 진짜로 평가할 확률을 최대화합니다.
                g_losses += g_loss.item()

        return g_losses / len(test_loader), d_losses / len(test_loader)

    def gradient_penalty(self, x, fake_x):
        batch_size = x.size(0)
        
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device, requires_grad=True) # epsilon은 0과 1 사이의 랜덤한 값으로, 배치 크기만큼 샘플링

        interpolated = epsilon * x + (1 - epsilon) * fake_x
        interpolated = interpolated.to(self.device)
        
        interpolated_score = self.model_d(interpolated)

        gradients = torch.autograd.grad(
            outputs=interpolated_score,  # 판별자의 출력값
            inputs=interpolated,  # 보간된 입력 데이터
            grad_outputs=torch.ones(interpolated_score.size()).to(self.device),  # 손실의 기울기를 1로 설정
            create_graph=True,  # 역전파를 위한 계산 그래프 생성
            retain_graph=True,  # 동일한 그래프에서 여러 번의 역전파를 허용
            only_inputs=True  # 입력에 대해서만 기울기를 계산
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
    