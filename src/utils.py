import transformers
#==================================#
#init_lr=3e-5
#Pooler and regressor: init_lr*3.6=1.08e-4
#Layer 0,1,2,3: init_lr=3e-5
#Layer 4,5,6,7: init_lr*1.75
#Layer 8,9,10,11: init_lr*3.5=1.05e-4
#==================================#
def AdamW_grouped_LLRD(model, init_lr):
    # init_lr= 3e-5
    opt_parameters = []  
    named_parameters = list(model.named_parameters())
    modelname = 'bert.'
    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_1 = ["layer.0", "layer.1", "layer.2", "layer.3"]
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]

    for i, (name, params) in enumerate(named_parameters):
        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01
        if name.startswith(modelname + "embeddings") or name.startswith(modelname + "encoder"):

            lr = init_lr # set_1

            lr = init_lr * 1.75 if any(p in name for p in set_2) else lr

            lr = init_lr * 3.5 if any(p in name for p in set_3) else lr

            opt_parameters.append({"params": params,
                                           "weight_decay": weight_decay,
                                           "lr": lr})
            continue

        if name.startswith(modelname + "regressor") or name.startswith(modelname + "pooler"):
            lr = init_lr * 3.6
            opt_parameters.append({"params": params,
                                           "weight_decay": weight_decay,
                                           "lr": lr})

    return transformers.AdamW(opt_parameters, lr=init_lr)
