import pytest
import torch
from PIL import Image
from .data_processing import is_valid_image, process_image


@pytest.mark.parametrize(
    "image",
    [
        Image.new("1", size=(50, 50)),
        Image.new("L", size=(50, 50)),
        Image.new("P", size=(50, 50)),
        Image.new("CMYK", size=(50, 50)),
        Image.new("YCbCr", size=(50, 50)),
        Image.new("LAB", size=(50, 50)),
        Image.new("HSV", size=(50, 50)),
        Image.new("I", size=(50, 50)),
        Image.new("F", size=(50, 50)),
    ],
)
def test_check_for_invalid_image(image):
    assert not is_valid_image(image)


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", size=(50, 50)),
        Image.new("RGBA", size=(50, 50)),
    ],
)
def test_check_for_valid_image(image):
    assert is_valid_image(image)


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", size=(50, 50), color=(20, 40, 60)),
        Image.new("RGB", size=(200, 50), color=(20, 40, 60)),
        Image.new("RGB", size=(500, 500), color=(20, 40, 60)),
        Image.new("RGBA", size=(50, 50), color=(20, 40, 60, 100)),
        Image.new("RGBA", size=(200, 50), color=(20, 40, 60, 100)),
        Image.new("RGBA", size=(500, 500), color=(20, 40, 60, 100)),
    ],
)
def test_process_image(image):
    image_processed = process_image(image)

    assert type(image_processed) == torch.Tensor
    assert image_processed.size() == (1, 3, 224, 224)

    image_processed_check = torch.cat(
        [
            torch.empty(1, 224, 224).fill_(-1.7754),
            torch.empty(1, 224, 224).fill_(-1.3354),
            torch.empty(1, 224, 224).fill_(-0.7587),
        ],
        dim=0,
    ).unsqueeze(0)

    assert torch.allclose(image_processed, image_processed_check, atol=1e-04)
