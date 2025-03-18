import torch
from torch import nn
import numpy as np
from typing import Union, Dict, Tuple
import torchvision
import transformers
# import timm
import math

class UniversalModelAnalyzer:
    """
    Attributes:
        model (nn.Module): PyTorch model。
        input_spec (Union[Tuple, Dict[str, Tuple]]): Input specification, supports tuple (vision model) and dictionary (NLP model) formats
        device (str):device set, "cpu", "cuda" or "xpu"
    """
    def __init__(self, model: nn.Module, input_spec: Union[Tuple, Dict[str, Tuple]], device: str = "xpu"):
        self.model = model.to(device)
        self.device = device
        self.input_spec = input_spec
        self.hooks = []
        self.layer_info = []

        # Register forward pass hook to capture layer information
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward propagation hooks for all leaf modules of the model, 
        use hook functions to capture the input and output shapes of each leaf module
        """
        def hook_wrapper(layer_name):
            """
            Wrapper for hook functions that capture the input and output shapes of a specific layer

            Args:
                layer_name (str): Current Layer。

            Returns:
                hook (function): hook function
            """
            def hook(module, inputs, outputs):
                """
                The actual hook function is executed after the layer forward propagation

                Args:
                    module (nn.Module): Current Module
                    inputs (tuple): Module Input
                    outputs (Tensor or tuple): Module Output。
                """
                input_shapes = []
                for inp in inputs:
                    if isinstance(inp, torch.Tensor):
                        shape = tuple(inp.shape[0:])
                        input_shapes.append(shape)
                    else:
                        input_shapes.append("N/A")

                output_shapes = []
                for out in (outputs if isinstance(outputs, tuple) else [outputs]):
                    if isinstance(out, torch.Tensor):
                        shape = tuple(out.shape[0:])
                        output_shapes.append(shape)
                    else:
                        output_shapes.append("N/A")

                self.layer_info.append({
                    "name": layer_name,
                    "module": module,
                    "input_shapes": input_shapes,
                    "output_shapes": output_shapes
                })
                print(layer_name, module, input_shapes, output_shapes)
                #return hook
            return hook

        # Traverse all leaf modules and register hooks
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules have no submodules
                self.hooks.append(module.register_forward_hook(hook_wrapper(name)))

    def _generate_dummy_input(self):
        """
        Generate dummy input
        """
        if isinstance(self.input_spec, tuple):  # vision model input
            return torch.randn(64, *self.input_spec).to(self.device)
        elif isinstance(self.input_spec, dict):  # NLP model input
            return {
                k: torch.randint(0, 10000, (1024,) + shape).to(self.device)
                for k, shape in self.input_spec.items()
            }
        else:
            raise ValueError("Unsupported input specification type")

    def _calculate_flops(self, module: nn.Module, input_shapes: list) -> int:
        """
        Calculates the FLOPs (floating point operations) of a given module

        Args:
            module (nn.Module): Modules for which FLOPs are to be calculated
            input_shapes (list): List of input shapes for the module

        Returns:
            int: Estimated FLOPs value. If the module type does not support FLOPs calculation, it returns 0 and prints a warning
        """
        flops = 0
        if not input_shapes:
            return 0

        # Conv2d
        if isinstance(module, nn.Conv2d):
            input_shape = input_shapes[0]
            out_h = (input_shape[1] + 2 * module.padding[0] -
                     module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
            out_w = (input_shape[2] + 2 * module.padding[1] -
                     module.dilation[1] * (module.kernel_size[1] - 1) - 1) // module.stride[1] + 1
            flops = 2 * (out_h * out_w * input_shape[0] * module.out_channels *
                     module.kernel_size[0] * module.kernel_size[1] // module.groups)
            if module.bias == True:
                flops += module.out_channels * out_h * out_w

        # Linear
        elif isinstance(module, nn.Linear):
            input_shape = input_shapes[0]
            flops = np.prod(input_shape) * module.out_features * 2
            if module.bias == True:
                flops += input_shape[0] * input_shape[1] * module.out_features

        # MultiheadAttention
        elif isinstance(module, nn.MultiheadAttention):
            print("a1:", input_shapes[0])
            q_shape = input_shapes[0]
            seq_len, embed_dim = q_shape[-2], q_shape[-1]
            flops += 3 * 2 * seq_len * embed_dim * module.embed_dim # Projection
            flops += 2 * seq_len * seq_len * embed_dim # SDPA
            flops += 3 * seq_len * seq_len # softmax
            flops += 2 * seq_len * seq_len * embed_dim # (A = softmax(**))* V
            flops += 2 * seq_len * embed_dim * module.embed_dim # Projection out
            if module.bias == True:
                flops += seq_len * module.embed_dim

        # LayerNorm
        elif isinstance(module, nn.LayerNorm):
            input_shape = input_shapes[0]
            flops = 5 * np.prod(input_shape)

        # BatchNorm
        elif isinstance(module, nn.BatchNorm2d):
            input_shape = input_shapes[0]
            flops = 2 * np.prod(input_shape)

        # Embedding / Identity
        elif isinstance(module, (nn.Embedding, nn.Identity)):
            flops = 0

        # Activation (ReLU, GELU, SiLU)
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Dropout, nn.Tanh)):
            input_shape = input_shapes[0]
            flops = np.prod(input_shape)

        # Pooling (MaxPool2d, AvgPool2d)
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            input_shape = input_shapes[0]
            kh = module.kernel_size
            kw = module.kernel_size
            flops = input_shape[0] * input_shape[1] // module.stride * kh * kw
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            H = input_shapes[0][2]
            W = input_shapes[0][3]
            O, P = module.output_size
            kh = math.ceil(H / O)
            kw = math.ceil(W / P)
            flops = O * P * (kh * kw + 1)
        elif isinstance(module, (transformers.activations.GELUActivation)):
            seq_len = input_shapes[0][1]
            hidden_size = input_shapes[0][2]
            if module.act.__name__ == "gelu":
                flops = 13 * seq_len * hidden_size
            elif module.act.__name__ == "_gelu_python":
                flops = 18 * seq_len * hidden_size
            else:
                raise "Unknown transformers.activations.GELUActivation.act name: {}".format(module.act)
        elif isinstance(module, (transformers.activations.NewGELUActivation)):
            seq_len = input_shapes[0][1]
            hidden_size = input_shapes[0][2]
            flops = 13 * seq_len * hidden_size
        elif isinstance(module, (transformers.pytorch_utils.Conv1D)):
            nf = module.nf
            nx = module.nx
            # weight(nx, nf), bias(nf,), input(N, seq, hidden)
            N, seq_len, hidden_size = input_shapes[0]
            # torchmm = 2mnk
            flops = 2 * seq_len * nx * nf
        else:
            print(f"Warning: FLOPs calculation of {type(module)} is not implemented, return 0")

        return int(flops)
    
    def _calculate_memory_load(self, module: nn.Module, input_shapes: list) -> int:
        """
        Calculate the memory load of a given module
        """
        memory_load = 0
        if not input_shapes:
            return 0
        memory_load = np.prod(input_shapes[0])
        return int(memory_load)

    def analyze(self):
        # Generate dummy input
        dummy_input = self._generate_dummy_input()

        # Run the forward pass to trigger the hook
        self.model.eval()
        with torch.no_grad():  # Disable gradient calculation to reduce memory consumption
            if isinstance(dummy_input, dict):
                self.model(**dummy_input)
            else:
                self.model(dummy_input)

        # Remove all registered hooks after analysis is complete
        for hook in self.hooks:
            hook.remove()

        total_flops = 0
        total_params = 0
        total_memory = 0

        # Print header
        header = f"{'Layer Name':<30} | {'Input Shapes':<40} | {'Output Shapes':<40} | {'FLOPs':<15} | {'Memory load':<15} |  {'Params':<10}"
        print(header)
        print("-" * len(header))

        # Output analysis results layer by layer
        for info in self.layer_info:
            module = info["module"]
            input_shapes = info["input_shapes"]
            output_shapes = info['output_shapes']
            # Compute params
            params = sum(p.numel() for p in module.parameters())
            # Compute FLOPs and memory load
            flops = self._calculate_flops(module, input_shapes)
            memory_load = self._calculate_memory_load(module, input_shapes) + params

            total_flops += flops
            total_params += params
            total_memory += memory_load

            # Print info for every layer
            print(f"{info['name']:<30} | {str(input_shapes):<40} | {str(info['output_shapes']):<40} | "
                  f"{flops:<15,} | {memory_load:<15,} | {params:<10,}")

        # Print summary
        print("-" * len(header))
        print(f"{'TOTAL':<30} | {'':<40} | {'':<40} | {total_flops:<15,} | {total_memory:<15,} | {total_params:<10,}")


if __name__ == "__main__":
    # example1: vision model
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'timm_vision_transformer')
    # model = torchvision.models.__dict__["resnet50"]()
    # # model = torchvision.models.__dict__["squeezenet1_1"]()
    # analyzer = UniversalModelAnalyzer(model, input_spec=(3, 256, 256))
    # analyzer.analyze()
    # example2: NLP model
    from transformers import BertModel
    from transformers import GPT2Tokenizer, GPT2Model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline,
    )
    generator = pipeline(
        "text-generation",
        model= "gpt2",
    )
 
    # model = BertModel.from_pretrained("bert-large-cased")
    # model = GPT2Model.from_pretrained('gpt2')
    # model = AutoModelForSequenceClassification.from_pretrained("bert-large-cased")
    print("model:", generator.model)
    analyzer = UniversalModelAnalyzer(
        generator.model,
        input_spec={"input_ids": (32,), "attention_mask": (32,)}
    )
    analyzer.analyze()
    # example3：Timms model
    # model = timm.create_model('vit_base_patch16_224', pretrained=True)
    # analyzer = UniversalModelAnalyzer(model, input_spec=(3, 224, 224))
    # analyzer.analyze()
