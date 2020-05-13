import argparse

import trainer
import data_util
from application import Application


# ------------------ #
# WEBCAM APPLICATION #
# ------------------ #
def parse_inputs():
    """
    Parse the input arguments for actions tu run.

    :return: Namespace containing the actions to run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collect-data", action="store_true", help="Collect new data")
    parser.add_argument("-i", "--index-dataset", action="store_true", help="Create a new dataset")
    parser.add_argument("-r", "--review-dataset", action="store_true", help="Review the dataset")
    parser.add_argument("-b", "--launch-batch", action="store_true", help="Launch a batch of tests")
    parser.add_argument("--load-model", action="store_true", help="Load a model")
    parser.add_argument("--launch-application", action="store_true", help="Launch the application")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse inputs
    args = parse_inputs()

    # Initialize
    util = data_util.ImageProcessor(erase_index=False)
    model = trainer.Model()

    # Check args for actions
    if args.collect_data:
        util.collect_data()
        util.index_dataset()

    elif args.review_dataset:
        util.index_dataset()
        util.review_dataset(model.MODEL_IMAGE_SIZE)

    elif args.launch_batch:
        util.index_dataset()
        model.launch_training_batch(args.load_model)

    elif args.launch_application:
        app = Application(model, util)
        app.run()
