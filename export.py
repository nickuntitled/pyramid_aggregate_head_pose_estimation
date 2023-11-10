import torch, argparse, os
from model import load_model

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--efficient', dest='efficient', help='efficient.',
          default=4, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='./deeplab_notorig_8020.pkl', type=str)
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    snapshot_path = args.snapshot

    if args.efficient == 4:
        network = 'efficientnet_v2_s'
    elif args.efficient == 3:
        network = 'efficientnet_v2_m'
    else:
        network = 'efficientnet_v2_l'

    model = load_model(pretrained=False, network = network)
        
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path, map_location = 'cpu')
    model.load_state_dict(saved_state_dict['model'] if 'model' in saved_state_dict else saved_state_dict)

    if not os.path.exists("exported"):
        os.makedirs("exported")

    torch.save(
        model.state_dict(), "exported/exported.pkl"
    )