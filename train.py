import os
import time

from module.utils import *
from module.config import *
from module.DNCM_encoder import *

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def main():
    arg = get_args() 

    # Define model and initialize optimizer

    DNCM_Encoder = DNCM_encoder()
    optimizer = torch.optim.Adam(DNCM_Encoder.parameters(), lr=0.001)

    # Initialize hyperparameter

    EPOCH = arg.epoch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize optimizer

    content_directory_path = arg.content_dir
    content_image_path = []

    for file_name in os.listdir(content_directory_path):
        file_path = os.path.join(content_directory_path, file_name)
        content_image_path.append(file_path)

    train_dataset = CustomDataset(content_image_path)

    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = 16)

    writer = SummaryWriter(log_dir = arg.log_dir)

    for epoch in range(EPOCH):
        DNCM_Encoder.train()

        for i, (original_content_image, colored_content_image) in enumerate(train_loader):
            optimizer.zero_grad()

            original_content_image = original_content_image.to(DEVICE)
            colored_content_image = colored_content_image.to(DEVICE)
            DNCM_Encoder = DNCM_Encoder.to(DEVICE)

            Z_ori = DNCM(DNCM_Encoder, original_content_image, colored_content_image, option='nDNCM')
            Z_colored = DNCM(DNCM_Encoder, colored_content_image, original_content_image, option='nDNCM')

            Y_ori = DNCM(DNCM_Encoder, Z_ori, colored_content_image, option='sDNCM')
            Y_colored = DNCM(DNCM_Encoder, Z_colored, original_content_image, option='sDNCM')

            L_con = calculate_L_con(Z_ori, Z_colored).mean()
            L_rec = calculate_L_rec(Y_ori, Y_colored, colored_content_image, original_content_image).mean()

            Loss = (L_rec + 10*L_con)

            Loss.backward()
            nn.utils.clip_grad_norm_(DNCM_Encoder.parameters(), max_norm=10.0)
            optimizer.step()

            if i % 30 == 0:
                loss = {'L_con': L_con,
                        'L_rec': L_rec,
                        'Loss' : Loss}
                
                denormalized_original_content_image = denormalize(original_content_image.detach())[0]
                denormalized_colored_content_image = denormalize(colored_content_image.detach())[0]
                denormalized_colored_original_content_image = denormalize(Y_ori.detach())[0]
                denormalized_colored_colored_content_image = denormalize(Y_colored.detach())[0]

                images = np.concatenate([
                    denormalized_original_content_image, 
                    denormalized_colored_content_image, 
                    denormalized_colored_original_content_image, 
                    denormalized_colored_colored_content_image], axis=1)
                images = np.transpose(images, (2, 0, 1))

                recode(writer, images, loss, epoch, i, 'Training')
    
            if i % 200==0:
                save_weight(DNCM_Encoder.state_dict(), epoch, i)

    writer.close()


if __name__ == "__main__":
    main()