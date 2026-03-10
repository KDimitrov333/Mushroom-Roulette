import torch
import torch.nn as nn
import torchvision.models as models
import timm

def transfer_model(num_classes=94):
    '''
    Load the Xception model and replace its output layer to classify in custom number of classes.

    Returns the modified model
    '''

    model = timm.create_model('legacy_xception', pretrained=True)

    # Lock the preexisting parameters as they already work for general image recognition
    for param in model.parameters():
        param.requires_grad = False

    # Let timm safely replace the classifier head
    model.reset_classifier(num_classes)

    return model

def defrost_top_layers(model):
    '''Unfreezes the deepest layers of transferred Xception so it can learn on the mushroom set'''
    
    # Xception uses 'blockX' and 'convX' instead of 'layerX'
    target_layers = ["block11", "block12", "conv3", "conv4", "bn3", "bn4"]

    for name, param in model.named_parameters():
        if any(target in name for target in target_layers):
            param.requires_grad = True

    return model


if __name__ == "__main__":
    test_model = transfer_model(94)
    print(test_model.fc)