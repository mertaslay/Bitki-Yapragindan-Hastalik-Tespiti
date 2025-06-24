# main.py
import argparse

# Bu iki dosyayı birazdan göstereceğim gibi fonksiyonlaştıracağız
from train import train
from test import test

def cli():
    parser = argparse.ArgumentParser(description="Leaf-Disease Recognition Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Calistirilacak adim: train veya test"
    )
    # Ek parametreler (örnek)
    parser.add_argument("--epochs", type=int, default=6, help="Epoch sayisi")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch boyutu")
    args = parser.parse_args()
    return args

def main():
    args = cli()

    if args.mode == "train":
        train(epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == "test":
        test()
    else:
        raise ValueError("mode train veya test olmalı")

if __name__ == "__main__":
    main()
