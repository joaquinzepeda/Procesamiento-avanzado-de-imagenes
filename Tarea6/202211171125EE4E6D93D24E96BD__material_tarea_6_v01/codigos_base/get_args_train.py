def get_args_train():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.epochs = 2
    parser.batch_size = 1
    parser.lr = 0.0001
    parser.load = False
    parser.scale = 1.0
    parser.val = 10.0
    parser.amp = False
    parser.bilinear = True
    parser.classes = 12

    return parser