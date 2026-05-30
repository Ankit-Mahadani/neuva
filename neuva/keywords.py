KEYWORDS = {
    "model","layer","train","load","predict","save","print",
    "let","fn","return","if","else","for","while",
    "on","to","epochs","lr","loss","true","false",
}
TYPES = {"int","float","bool","string","tensor","matrix"}
ACTIVATIONS = {"relu","sigmoid","softmax","tanh","linear","leaky_relu","gelu"}
LOSS_FUNCTIONS = {"crossentropy","mse","mae","huber","binary_crossentropy"}
LAYER_TYPES = {"dense","conv","pool","dropout","flatten","norm"}
BUILTINS = {"load","save","predict","accuracy","split","normalize","shuffle","shape","print"}
ALL_RESERVED = KEYWORDS | TYPES | ACTIVATIONS | LOSS_FUNCTIONS
