import module_ai
import torch
import visualtorch

ai = module_ai.ai()

input = torch.randn(2,3)
ai.model_init(input)
img = visualtorch.layered_view(ai.model.to("cpu"), input.shape, spacing=25, legend=True)
img.save("model_layeredview.png")
img = visualtorch.graph_view(ai.model.to("cpu"), input.shape)
img.save("model_graphview.png")
img = visualtorch.lenet_view(ai.model.to("cpu"), input.shape)
img.save("model_lenetview.png")