import cv2

def overlay_mask_on_image(image, mask_rgb, alpha=0.5):
    """
    image: tensor [3, H, W] یا numpy [H, W, 3]
    mask_rgb: numpy array [H, W, 3] — ماسک رنگی
    return: overlay image [H, W, 3]
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        image = (image * 255).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

