

def get_activation(activation, name):
    """
    Sets a forward hook for computing activations for a given layer by name.
    """
    def hook(module, input, output):
        B, C = output.shape[0], output.shape[1]

        # Save feature activations
        if name in activation:
            activation[name].append(output.view(B, C, -1).mean(dim=2))
        else:
            activation[name] = [output.view(B, C, -1).mean(dim=2)]

    return hook


def return_hooks(model, extracted_layers):
    """
    Returns handles to forward hooks for computing activations
    as well as a dictionary of activations and co-activations.
    """
    handle_list = []
    activation = {}
    for name, module in list(model.named_modules()):
        if name in extracted_layers:
            handle_list += [module.register_forward_hook(
                get_activation(activation, name))]
    return handle_list, activation
