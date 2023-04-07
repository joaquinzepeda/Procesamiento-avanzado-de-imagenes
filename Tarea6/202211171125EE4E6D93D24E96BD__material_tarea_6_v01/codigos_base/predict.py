dir_img = Path('/content/training/image_2/')
dir_mask = Path('/content/training/semantic/')
checkpoint_file = '/content/checkpoints/checkpoint_epoch2.pth'

dataset_global = KittiDataset(dir_img, dir_mask, 'gray', scale = 1.0)

net = UNet(n_channels=3, n_classes=12, bilinear = True)

device = torch.device('cpu')
logging.info(f'Using device {device}')
net.to(device=device)
net.load_state_dict(torch.load(checkpoint_file, map_location=device))

net.cpu()

for i in range(1):
  mysize = dataset_global[i]['image'].size()

  img_global = dataset_global[i]['image'].reshape((1,mysize[0],mysize[1],mysize[2])) #.cuda()
  img_gmask = dataset_global[i]['mask'].reshape((1,mysize[1],mysize[2])) #.cuda()

  print("Tamaño del tensor de entrada:", img_global.size())
  img_pred = net.forward(img_global)
  img_np = img_pred.cpu().detach().numpy()
  print("Tamaño del tensor de salida:", img_pred.size())

  # Por hacer: dibujar img_global, img_gmask y img_np usando plt.imshow()
  # Para esto, es necesario modificar las dimensiones de las imágenes, y transformar
  # las imágenes que contienen labels a formato np.ubyte
  # Además img_global debe ser multiplicada por 255