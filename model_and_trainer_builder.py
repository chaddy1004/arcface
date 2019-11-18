from network.network import Res50
from trainers.trainer import Trainer
from dataloader.dataloader import Dataloader

def get_network_builder(config):
    if config.model.architecture == "ResNet50":
        return Res50(config)
    else:
        raise ValueError(f"unknown network architecture {config.model.architecture}")


def build_model_and_trainer(config, data_loader):
    network_builder = get_network_builder(config=config)
    backbone = network_builder.define_backbone(model_name="backbone")
    full_model = network_builder.build_full(backbone=backbone, model_name="full")
    # TODO: Finish implementing the trainer
    data_loader = Dataloader(config=config)
    trainer = Trainer(data_loader=data_loader, config=config, backbone=backbone, full=full_model)
    return backbone, full_model, trainer
