# Execution Command:
# TORCHINDUCTOR_FREEZING=1 python inductor_quant_acc.py
import torch
import torchvision.models as models
import torch._dynamo as torchdynamo
import copy
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.pt2e.quantizer import X86InductorQuantizer
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

def run_model(model_name):
    valdir = "/workspace/pt_data/val/"
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
    model = models.__dict__[model_name](pretrained=True).eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    quant_top1 = AverageMeter('Acc@1', ':6.2f')
    quant_top5 = AverageMeter('Acc@5', ':6.2f')
    x = torch.randn(50, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    with torch.no_grad():
        # Generate the FX Module
        exported_model, guards = torchdynamo.export(
            model,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        # Create X86InductorQuantizer
        quantizer = X86InductorQuantizer()
        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
        # PT2E Quantization flow
        prepared_model = prepare_pt2e(exported_model, quantizer)
        # Calibration
        for i, (images, _) in enumerate(cal_loader):
            prepared_model(images)
            if i==10: break
        converted_model = convert_pt2e(prepared_model).eval()
        # Lower into Inductor
        optimized_model = torch.compile(converted_model)
        # Benchmark
        for i, (images, target) in enumerate(val_loader):
            #output = model(images)
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #top1.update(acc1[0], images.size(0))
            #top5.update(acc5[0], images.size(0))
            quant_output = optimized_model(images)
            quant_acc1, quant_acc5 = accuracy(quant_output, target, topk=(1, 5))
            quant_top1.update(quant_acc1[0], images.size(0))
            quant_top5.update(quant_acc5[0], images.size(0))
        #print(model_name + " fp32: ")
        #print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #      .format(top1=top1, top5=top5))
        print(model_name + " int8: ")
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=quant_top1, top5=quant_top5))

if __name__ == "__main__":
    model_list=["alexnet","shufflenet_v2_x1_0","mobilenet_v3_large","vgg16","densenet121","mnasnet1_0","squeezenet1_1","mobilenet_v2","resnet50","resnet152","resnet18","resnext50_32x4d"]
    for model in model_list:
        run_model(model)

