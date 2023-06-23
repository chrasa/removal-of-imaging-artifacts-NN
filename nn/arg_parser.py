from argparse import ArgumentParser

class ArgParser(ArgumentParser):
    def __init__(self) -> None:
        super().__init__(description='Train model to remove imaging artifacts')
        self.add_argument('imaging_method', type=str, help="Used imaging algorithm. Either RTM or ROM")
        self.add_argument('model', type=str, help='Name of the model')
        self.add_argument('loss', type=str, help='Name of the loss function')
        self.add_argument('-stride', type=int, default=5, help='Stride used for the model')
        self.add_argument('-nimages', type=int, default=2000, help="Number of images used for training")
        self.args = self.parse_args()
