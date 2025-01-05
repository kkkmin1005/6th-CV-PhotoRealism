import argparse

def get_args():
    
    parse = argparse.ArgumentParser()
    
    # Hyper paramter
    parse.add_argument('--epoch', type=int, required=True)

    # Directory path
    parse.add_argument('--content_dir', type=str, required=True)
    parse.add_argument('--log_dir', type=str, required=True)
    
    cfg = parse.parse_args()
    return cfg