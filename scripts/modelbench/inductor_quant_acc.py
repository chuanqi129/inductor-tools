# Execution Command:
# TORCHINDUCTOR_FREEZING=1 python inductor_quant_acc.py
import torch
import torchvision.models as models
import torch._dynamo as torchdynamo
import torch._inductor as torchinductor
import copy
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e, prepare_qat_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch._export import capture_pre_autograd_graph
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def run_model(model_name, args):
    torchinductor.config.freezing = True
    if args.cpp_wrapper:
        print("using cpp_wrapper")
        torchinductor.config.cpp_wrapper = args.cpp_wrapper
    valdir = "/workspace/benchmark/imagenet/val/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=50, shuffle=False,
    num_workers=4, pin_memory=True)
    cal_loader = copy.deepcopy(val_loader)
    model = models.__dict__[model_name](pretrained=True)
    if args.is_qat:
        model = model.train()
    else:
        model =model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    x = torch.randn(50, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)

    # Calibration
    if args.is_qat:
        print("using qat")
        for i, (images, _) in enumerate(cal_loader):
            exported_model = capture_pre_autograd_graph(
                model,
                (images,)
            )
            if i==10: break
        quantizer = xiq.X86InductorQuantizer()
        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=True))
        prepared_model = prepare_qat_pt2e(exported_model, quantizer)
        lr = 0.0001
        momentum = 0.9
        weight_decay = 1e-4
        optimizer = torch.optim.SGD(prepared_model.parameters(), lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
        optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss()
        for i, (images, target) in enumerate(val_loader):
        # print(" start QAT Calibration step: {}".format(i), flush=True)
            images = images
            target = target
            output = prepared_model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()               
            if i == 1:
                break

        with torch.no_grad():
            converted_model = convert_pt2e(prepared_model)
            torch.ao.quantization.move_exported_model_to_eval(converted_model)
            # Lower into Inductor
            optimized_model = torch.compile(converted_model)
    elif args.is_fp32:
        print("using fp32")
        with torch.no_grad():
            optimized_model = torch.compile(model)     
    else:
        print("using ptq")
        with torch.no_grad():
            exported_model = capture_pre_autograd_graph(
                model,
                example_inputs
            )
            quantizer = xiq.X86InductorQuantizer()
            quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
            # PT2E Quantization flow
            prepared_model = prepare_pt2e(exported_model, quantizer)
            # Calibration
            prepared_model(*example_inputs)
            converted_model = convert_pt2e(prepared_model)
            torch.ao.quantization.move_exported_model_to_eval(converted_model)
            optimized_model = torch.compile(converted_model)
    # Benchmark
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            #output = model(images)
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #top1.update(acc1[0], images.size(0))
            #top5.update(acc5[0], images.size(0))
            quant_output = optimized_model(images)
            quant_acc1, quant_acc5 = accuracy(quant_output, target, topk=(1, 5))
            quant_top1.update(quant_acc1[0], images.size(0))
            quant_top5.update(quant_acc5[0], images.size(0))
        if args.is_fp32:
            print(model_name + " fp32: " + ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=quant_top1, top5=quant_top5))
        else:
            print(model_name + " int8: " + ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=quant_top1, top5=quant_top5))

if __name__ == "__main__":
    model_list=["alexnet","densenet121","mnasnet1_0","mobilenet_v2","mobilenet_v3_large","resnet152","resnet18","resnet50","resnext50_32x4d","shufflenet_v2_x1_0","squeezenet1_1","vgg16"]
    # model_list=["alexnet"]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantize",
        action='store_true',
        help="enable quantize for inductor",
    )
    parser.add_argument(
        "--cpp_wrapper",
        action='store_true',
        help="enable cpp wrapper for inductor",
    )
    parser.add_argument(
        "--is_qat",
        action='store_true',
        help="enable qat quantization for inductor",
    )
    parser.add_argument(
        "--is_fp32",
        action='store_true',
        help="fp32 inductor",
    )
    args = parser.parse_args()
    for model in model_list:
        run_model(model, args)

