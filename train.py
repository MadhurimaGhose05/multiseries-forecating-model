import argparse
from training.train_global import train as train_global
from training.train_chronos import train_or_prepare
from training.evaluate_chronos import evaluate_chronos

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="data")
    p.add_argument("--artifact-dir",default="artifacts")
    p.add_argument("--skip-chronos",action="store_true")
    a=p.parse_args()
    train_global(a.data_dir,a.artifact_dir)
    if not a.skip_chronos:
        train_or_prepare(a.artifact_dir)
        evaluate_chronos(a.data_dir,a.artifact_dir)
if __name__=="__main__": main()
