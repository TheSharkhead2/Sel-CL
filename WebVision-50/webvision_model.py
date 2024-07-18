import sys

sys.path.append('../utils')
import models_webvision as mod


def build_model(args, device):
    if args.network == "RN18":
        model = mod.ResNet18(
            num_classes=args.num_classes,
            low_dim=args.low_dim,
            head=args.headType
        ).to(device)
    elif args.network == "PARN18":
        model = mod.PreActResNet18(
            num_classes=args.num_classes,
            low_dim=args.low_dim,
            head=args.headType
        ).to(device)
    elif args.network == "PARN50":
        model = mod.PreActResNet50(
            num_classes=args.num_classes,
            low_dim=args.low_dim,
            head=args.headType
        ).to(device)
    elif args.network == "RN50":
        model = mod.ResNet50(
            num_classes=args.num_classes,
            low_dim=args.low_dim,
            head=args.headType
        ).to(device)

    return model
