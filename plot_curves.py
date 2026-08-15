import matplotlib.pyplot as plt
import os

epochs = [9, 10]
train_loss = [0.5019, 0.4352]
val_loss = [0.4605, 0.4188]
val_psnr = [25.28, 28.46]
val_ssim = [0.7633, 0.7831]

os.makedirs('results', exist_ok=True)

plt.figure()
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Val Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('results/loss_curve.png')

plt.figure()
plt.plot(epochs, val_psnr, label='Val PSNR', marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.title('Validation PSNR')
plt.legend()
plt.savefig('results/psnr_curve.png')

plt.figure()
plt.plot(epochs, val_ssim, label='Val SSIM', marker='o', color='purple')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.title('Validation SSIM')
plt.legend()
plt.savefig('results/ssim_curve.png')

print("Training curves generated successfully.")
