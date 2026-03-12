import torch
import torch.nn as nn
import torchvision.models as models

def transfer_model(num_classes=94):
    '''
    Load the ResNet50 model and replace its output layer to classify in custom number of classes.

    Returns the modified model
    '''

    # Use IMAGENET1K_V2 for highest pre-training accuracy
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Lock the preexisting parameters as they already work for general image recognition
    for param in model.parameters():
        param.requires_grad = False

    # fc is the output layer of ResNet50
    num_features = model.fc.in_features

    # New custom output layer, to be trained
    model.fc = nn.Linear(num_features, num_classes)

    return model

def defrost_top_layers(model):
    '''Unfreezes layer4 of transferred ResNet50 so it can learn on the mushroom set'''

    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True

    return model


if __name__ == "__main__":
    test_model = transfer_model(94)
    print(test_model.fc)