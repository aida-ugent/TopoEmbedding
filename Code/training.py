import torch
import random
import pandas as pd
import time
from Code.losses import get_topological_loss, combine_topological_losses
from Code.topembed import str_to_method

try:
    import wandb
    if wandb.__file__ is not None:
        usewandb = True
    else:
        usewandb = False
except ImportError as err:
    usewandb = False

def train_model(model, data, num_epochs, learning_rate,
                random_state=None, eps=1e-07, weight_decay=0,
                batch_size=0, verbosity=1):

    if batch_size > data.shape[0]:
        raise ValueError(
            f"Batch size of {batch_size} is larger than dataset size {data.shape[0]}")

    # Initialize the optimization
    if random_state is not None:
        random.seed(random_state)
        torch.manual_seed(random_state)

    # Initialize model
    data = model.initialize(data)
    data = torch.tensor(data).type(torch.float)

    # Track initial loss
    loss, loss_components = model(data)
    if verbosity > 0:
        print("Initial loss ", end="")
        for key, value in loss_components.items():
            if key != "W":
                print(f"{key}: {value:.4f}, ", end="")
        print("")
    loss_components.update({'epoch': 0})
    if usewandb:
        wandb.log(loss_components)
    loss_df = pd.DataFrame.from_records([loss_components])

    if batch_size > 0:
        dataLoader = torch.utils.data.DataLoader(dataset=data,
                                                 batch_size=batch_size,
                                                 shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay,
                                 eps=eps)

    for epoch in range(1, num_epochs+1):
        model.train()

        try:
            if batch_size > 0:
                for data in dataLoader:
                    loss, loss_components = model(data)

                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    model.on_optimizer_step()
            else:
                loss, loss_components = model(data)

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.on_optimizer_step()

            loss_components.update({'epoch': epoch})
            loss_df = pd.concat((loss_df,
                                pd.DataFrame.from_records([loss_components])),
                                ignore_index=True)

            if epoch % (int(num_epochs / 10)) == 0 and verbosity > 0:
                print(f"Epoch {epoch:4d}: ", end="")
                for key, value in loss_components.items():
                    if (key != "epoch" and key != "W"):
                        print(f"{key}: {value:.4f}, ", end="")
                print("")
                if usewandb:
                    wandb.log(loss_components)
        except RuntimeError as e:
            # d-dimensional feature is not existent
            if "element 0 of tensors" in e.args[0]:
                print("Error: the specified d-dimensional features does not exist in the persistence diagram.")
                break
            else:
                raise
    model.on_train_end()
    return model.encode(data).detach().numpy(), loss_df


def compute_embedding(data,
                      method_name,
                      method_config,
                      loss_config,
                      topo_weight,
                      training_config,
                      random_state=42,
                      final_loss_config=None,
                      emb_init=None,
                      verbosity=1,
                      return_loss_history=False):

    # construct loss
    topo_losses = list(loss_config.keys())
    loss_functions = [get_topological_loss(
        loss_name, **loss_config[loss_name]) for loss_name in topo_losses]
    topo_loss = combine_topological_losses(loss_functions)

    # Embedding method
    model = str_to_method(method_name)(**method_config,
                                       topo_loss=topo_loss,
                                       topo_weight=topo_weight,
                                       random_state=random_state,
                                       verbosity=verbosity
                                       )

    start_time = time.time()

    data = model.initialize(data, emb_init)
    if method_name == "ManoptPCA":
        Y, loss_df = model.train(verbosity=verbosity,
                                 **training_config)
    else:
        Y, loss_df = train_model(model=model,
                                 data=data,
                                 verbosity=verbosity,
                                 **training_config)

    embedding_time = time.time() - start_time
    if verbosity > 0:
        mm, ss = divmod(embedding_time, 60)
        hh, mm = divmod(mm, 60)
        print(f"Embedding time {hh:2.0f}h {mm:2.0f}m {ss:2.4f}s")
        print("")

    if final_loss_config is not None:
        final_loss_functions = [get_topological_loss(
            loss_name, **loss_config[loss_name]) for loss_name in topo_losses]
        final_topo_loss = combine_topological_losses(final_loss_functions)
        tloss = final_topo_loss(torch.tensor(Y).type(torch.float)).item()
        print(f"Topological (using final loss): {tloss}")
    
    if verbosity > 0:
        final_row = loss_df.loc[loss_df.epoch == max(loss_df.epoch)].iloc[0]
        print(final_row)

    res = {"emb_loss": final_row.at["emb_loss"],
           "topo_loss": tloss if final_loss_config is not None else final_row.at["topo_loss"],
           "span_matrix": model.get_span_matrix(),
           "emb_time": embedding_time,
           "embedding": Y,
           "total_loss": final_row.at["weighted_total_loss"],
           "loss_config": loss_config,
           "topo_weight": topo_weight,
           "method_config": method_config,
           "data_dim": data.shape,
           "emb_init": emb_init}
    
    if return_loss_history:
        return (Y, res, loss_df)
    else:
        return (Y, res)
