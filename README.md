## ONNXKnet.jl : Read ONNX graphs in Knet

ONNXKnet.jl is an [ONNX](http://onnx.ai/) backend for Knet as it provides model import functionalities for the [Knet.jl](https://github.com/denizyuret/Knet.jl) machine learning framework. This is heavily inspired by [ONNX.jl](https://github.com/FluxML/ONNX.jl).

## Loading ONNX serialized models:

```
>>> using Knet, ONNXKnet

>>> ONNXKnet.load_model("model.onnx") # produces two files: weights.bson and model.jl

>>> weights = ONNXKnet.load_weights("weights.bson")

>>> model = include("model.jl")
```

`model` is the corresponding Knet model, and can be called just like any other Knet model: `model(ip)`, where `ip` is the input
with appropriate dimensions.

## To Do:

This package is still under development and quite a few things need to be done here:

1. Add support for other larger models : VGG, face detection, object detection, emotion ferplus and so on.

2. Look into ways of removing code redundacy : Discuss if it's a good idea to add ONNX.jl as a dependency, so that the model interface code can be shared.

## Contributing

Contributors are always welcome. Discussion takes place on Julia Slack, either on the #Knet or #ONNX channels.
