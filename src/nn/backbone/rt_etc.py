
import torch
import torch.nn.utils.prune as prune
from torch import nn

class RTETC:
    """
    Real-Time Efficient Transformer Compression:
    static methods for pruning, quantization, and distillation.
    """
    @staticmethod
    def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
        convs = [(m, 'weight') for m in model.modules() if isinstance(m, nn.Conv2d)]
        prune.global_unstructured(convs,
                                  pruning_method=prune.L1Unstructured,
                                  amount=amount)
        return model

    @staticmethod
    def quantize_model(model: nn.Module) -> nn.Module:
        model.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        return model

    @staticmethod
    def distill(teacher: nn.Module,
                student: nn.Module,
                loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                device: str) -> None:
        teacher.eval()
        criterion = nn.MSELoss()
        for imgs, _ in loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                ft = teacher(imgs)
            fs = student(imgs)
            loss = criterion(ft, fs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()