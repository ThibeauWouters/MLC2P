import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

output = traced_script_module(torch.ones(10, 3, 224, 224))
print('First elements of the prediction: ', output[0, :5])

traced_script_module.save("traced_model.pt")
