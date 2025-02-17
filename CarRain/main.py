import os
import argparse
import yaml

from CARLASimulator import simulator

# Read config from args and yaml file
def get_args():
    parser = argparse.ArgumentParser(description='CARLA Run')

    parser.add_argument('-c', '--config',
                        type=str,
                        default=None,
                        help='The config file')    
    
    args = parser.parse_args()

    if args.config:
        assert os.path.exists(args.config), ("The config file is missing.", args.config)
        with open(args.config,"r") as f:
            cfg = yaml.safe_load(f)['scene']
        for key in cfg:
            args.__dict__[key] = cfg[key]

    return args


if __name__=="__main__":
    args = get_args()
    # Create CARLASimulator
    simulator.init(args)
    simulator.run()