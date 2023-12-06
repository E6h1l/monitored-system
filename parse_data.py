import argparse

def conditions_parser():
    parser = argparse.ArgumentParser(description='Run training script')
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help='Path to CONDITIONS.json'
    )

    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help='Conditions file name: NAME.json'
    )

    return parser