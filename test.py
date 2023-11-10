import os, argparse, time, json
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn

from tqdm import tqdm

import data
from model import load_model
import numpy as np 
import random 

from rich import print
from rich.console import Console

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0], -1 for CPU',
            default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    parser.add_argument('--start', dest='start', help='start.',
          default=1, type=int)
    parser.add_argument('--num_epoch', dest='num_epoch', help='num_epoch.',
          default=160, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=64, type=int)
    parser.add_argument('--crop', dest='crop', help='crop.',
          default=1, type=int)
    parser.add_argument('--ad', dest='ad', help='ad.',
          default=0.2, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='AFLW2000', type=str)
    parser.add_argument('--crop_size', dest='crop_size', help='crop_size.', default=224, type=int)
    parser.add_argument('--input_size', dest='input_size', help='input_size.', default=224, type=int)
    parser.add_argument('--efficient', dest='efficient', help='efficient.',
          default=4, type=int)

    args = parser.parse_args()

    return args

def test(console, model, pose_dataset):

    args = parse_args()
    
    seed = 0
    cudnn.enabled = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    gpu = args.gpu_id

    # Validation
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=2)
    
    if args.gpu_id != -1:
        model.cuda(gpu)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    console.log(f"[*] The model has { pytorch_total_params } parameters.")
    console.log('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(20)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor))

    ten_idx_tensor = np.array([idx for idx in range(10)])
    ten_idx_tensor = Variable(torch.FloatTensor(ten_idx_tensor))

    if args.gpu_id != -1:
        idx_tensor = idx_tensor.cuda(gpu)
        ten_idx_tensor = ten_idx_tensor.cuda(gpu)
        
    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    softmax = nn.Softmax()

    if args.gpu_id != -1:
        softmax = softmax.cuda(gpu)

    start = time.time()
    with torch.no_grad():
        if True:
            for i, (images, labels, cont_labels, raw_img, index) in enumerate(tqdm(test_loader)):
                images = Variable(images)
                if args.gpu_id != -1:
                    images = images.cuda(gpu)

                label_yaw = cont_labels[:,0].float()
                label_pitch = cont_labels[:,1].float()
                label_roll = cont_labels[:,2].float()

                # Forward pass
                yaw_coarse, yaw_shift, pitch_coarse, pitch_shift, roll_coarse, roll_shift =  model(images)

                # Continuous
                # Coarse
                ten_yaw = softmax(yaw_coarse)
                ten_pitch = softmax(pitch_coarse)
                ten_roll = softmax(roll_coarse)

                yaw_predicted = torch.sum(ten_yaw * ten_idx_tensor, 1) * 20 - 100
                pitch_predicted = torch.sum(ten_pitch * ten_idx_tensor, 1) * 20 - 100
                roll_predicted = torch.sum(ten_roll * ten_idx_tensor, 1) * 20 - 100

                # Shift
                shifted_yaw = softmax(yaw_shift)
                shifted_pitch = softmax(pitch_shift)
                shifted_roll = softmax(roll_shift)

                yaw_predicted += torch.sum(shifted_yaw * idx_tensor, 1) 
                pitch_predicted += torch.sum(shifted_pitch * idx_tensor, 1)
                roll_predicted += torch.sum(shifted_roll * idx_tensor, 1)

                # Mean absolute error
                p_gt_deg = label_pitch.cuda(gpu) if gpu != -1 else label_pitch
                y_gt_deg = label_yaw.cuda(gpu) if gpu != -1 else label_yaw 
                r_gt_deg = label_roll.cuda(gpu) if gpu != -1 else label_roll
                p_pred_deg = pitch_predicted
                y_pred_deg = yaw_predicted
                r_pred_deg = roll_predicted
                pitch_error += torch.sum(torch.abs(p_pred_deg - p_gt_deg))
                yaw_error += torch.sum(torch.abs(y_pred_deg - y_gt_deg))
                roll_error += torch.sum(torch.abs(r_pred_deg - r_gt_deg))

        total = len(pose_dataset)
        usage = time.time() - start
        console.log(f"[*] finish within { usage } seconds. Per image { usage/len(pose_dataset) }.")
        yaw_error, pitch_error ,roll_error = [x.cpu() for x in [yaw_error, pitch_error ,roll_error]]
        mean_error = ((yaw_error + pitch_error + roll_error)) / 3
        console.log('Test error in degrees of the model on the ' + str(total) +
        ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, Mean: %.4f' % (yaw_error / total,
        pitch_error / total, roll_error / total, mean_error / total))

        result = {
            "total_image": total,
            "yaw_error": yaw_error.item() / total,
            "pitch_error": pitch_error.item() / total,
            "roll_error": roll_error.item() / total,
            "mean_error": mean_error.item() / total
        }

    return result
    
if __name__ == '__main__':

    # console
    console = Console()

    args = parse_args()
    console.log('Loading data.')

    #if False:
    arr_transform = []
    if args.crop == 1:
        arr_transform = [transforms.Resize((args.input_size, args.input_size))]
        arr_transform.append(transforms.CenterCrop((args.crop_size, args.crop_size)))
    else:
        arr_transform = [transforms.Resize((args.input_size, args.input_size))]
        
    arr_transform = [
        *arr_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transformations = transforms.Compose(arr_transform)

    if args.dataset == 'AFLW2000':
        pose_dataset = data.AFLW2000(args.data_dir, args.filename_list, None, transformations, sixd = False, ad = args.ad)
    elif args.dataset == 'BIWI':
        pose_dataset = data.BIWI_kinect(args.data_dir, args.filename_list, None, transformations, mode = 'val', sixd = False)
    else:
        pose_dataset = data.BIWI(args.data_dir, None, transformations, mode = 'overall_val') #'test')

    snapshot_path = os.path.join(args.snapshot)
    fileext = snapshot_path.split('.')[-1]
    
    if fileext == 'pkl':
        # EfficientNetV2 structure
        if args.efficient == 4:
            network = 'efficientnet_v2_s'
        elif args.efficient == 3:
            network = 'efficientnet_v2_m'
        else:
            network = 'efficientnet_v2_l'

        model = load_model(pretrained=False, 
                            network = network)

        # Load snapshot
        with console.status("[bold green]Loading snapshot...") as status:
            saved_state_dict = torch.load(snapshot_path, map_location = f"cuda:{ args.gpu_id }" if args.gpu_id != -1 else "cpu")
            model.load_state_dict(saved_state_dict['model'] if 'model' in saved_state_dict else saved_state_dict)

        result = test(console, model, pose_dataset)
    else:
        total_result = []
        lowest = { 'epoch': 0, 'value': 0 }
        try:
            for epoch in range(args.start, args.num_epoch + 1):
                console.log(f"Epoch { epoch }")
                
                # EfficientNetV2 structure
                if args.efficient == 4:
                    network = 'efficientnet_v2_s'
                elif args.efficient == 3:
                    network = 'efficientnet_v2_m'
                else:
                    network = 'efficientnet_v2_l'

                model = load_model(pretrained=False, 
                                    network = network)

                # Load snapshot
                with console.status("[bold green]Loading snapshot...") as status:
                    saved_state_dict = torch.load(os.path.join(snapshot_path, f"{ str(epoch) }.pkl"), map_location = f"cuda:{ args.gpu_id }" if args.gpu_id != -1 else "cpu")
                    model.load_state_dict(saved_state_dict['model'] if 'model' in saved_state_dict else saved_state_dict)

                result = test(console, model, pose_dataset)
                total_result.append({
                    **result,
                    "epoch": epoch
                })

                mean_error = result['mean_error']
                if epoch == 1 or epoch == args.start or mean_error < lowest['value']:
                    lowest = {
                        **result,
                        "epoch": epoch,
                        "value": mean_error
                    }
        except KeyboardInterrupt as e:
            console.log(f"[*] Bye.")

        f = open(os.path.join(snapshot_path, f"{ args.dataset }_epoch.json"), "w+")
        f.write(json.dumps(total_result))
        f.close()

        print(lowest)
