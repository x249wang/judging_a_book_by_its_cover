from torchvision import transforms

image_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def is_valid_image(image):
    return image.mode in ["RGB", "RGBA"]


def process_image(image, image_transforms=image_transforms):
    image_rgb = image.convert("RGB")
    image_rgb = image_transforms(image_rgb)
    return image_rgb.unsqueeze(0)
