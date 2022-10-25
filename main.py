# Set working directory
import argparse
from os.path import dirname, abspath, join
import yaml

from Code.training import compute_embedding
import Data.datasets as datasets
import Code.visualization as viz
from Code.dataloader import dot_to_dictionary
import matplotlib.pyplot as plt

try:
    import wandb
    if wandb.__file__ is not None:
        usewandb = True
    else:
        usewandb = False
except ImportError as err:
    usewandb = False


def main(config):
    dataset_name = config['dataset_name']
    method_name = config['method_name']
    topo_losses = config["topo"]

    if usewandb:
        wandb.init(project='topo_experiments',
                entity="heitere",
                tags=[dataset_name, method_name] + topo_losses,
                config=config)

    config = dot_to_dictionary(config)

    # Data
    dataset_generator = datasets.str_to_data(dataset_name)
    data, labels = dataset_generator(**config.get(dataset_name, {}))

    # topological loss
    loss_config = {}
    for loss_name in topo_losses:
        loss_config[loss_name] = config.get(loss_name, {})

    Y, result_dict = compute_embedding(data,
                                       method_name=method_name,
                                       method_config=config.get(
                                           method_name, {}),
                                       loss_config=loss_config,
                                       topo_weight=config.get(
                                           'topo_weight', 0),
                                       training_config=config.get(
                                           "training", {}),
                                       random_state=config['random_state'],
                                       final_loss_config=None,
                                       emb_init=None,
                                       verbosity=1,
                                       return_loss_history=False)
    fig_embedding = viz.plot_paper(Y, colors=labels)
    plt.show()
    
    if usewandb:
        fig_persistence = viz.plot_persistence(Y)
        wandb.log({'embedding_time': result_dict["emb_time"]})
        wandb.log({"Final embedding": wandb.Image(fig_embedding)})
        wandb.log({"Persistence diagram": wandb.Image(fig_persistence)})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', type=str, default="default.yaml", 
                        help="Path to .yaml config file.")
    args = parser.parse_args()

    config_file_path = join(dirname(abspath(__file__)), args.config)
    
    with open(config_file_path) as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    main(parse_args())
