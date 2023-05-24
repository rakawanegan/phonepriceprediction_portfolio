import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="sample.ini")
    parser.add_argument("--name", type=str, default="knnsample")
    args = parser.parse_args()
    return args

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed with error: {e.stderr}")
        return None


def main():
    args = parse_args()
    output = run_command(f"python3 main.py --path {args.path} --name {args.name} > results/log/{args.name}.log")
if __name__ == '__main__':
    main()