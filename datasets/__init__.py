import torch.utils.data
import torchvision

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'bsi_panoptic':
        from .bsi_panoptic import build as build_bsi_panoptic
        return build_bsi_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
