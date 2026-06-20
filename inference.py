import argparse
import numpy as np
import torch
from PIL import Image
import io

from config import DEVICE, WEIGHTS_FINAL
from streams import all_streams
from fusion import fusion_model
from forensics import compute_ela
from dataset import VAL_TRANSFORM


STREAM_NAMES = [
    'Xception/ResNet (spatial)',
    'EfficientNet   (spatial)',
    'F3-Net         (frequency)',
    'ViT            (global context)',
    'Swin           (hierarchical)',
    'CapsNet        (structural)',
    'U-Net encoder  (geometry)',
    'Forensics      (ELA+PRNU)',
]


def load_image(image_path: str):
    """Load image as both a normalised tensor and raw PIL (for forensics)."""
    pil = Image.open(image_path).convert('RGB')
    tensor = VAL_TRANSFORM(pil).unsqueeze(0).to(DEVICE)  
    return pil, tensor


def save_ela_heatmap(pil_image: Image.Image, save_path: str = 'ela_heatmap.png'):
    """
    Save the ELA heatmap as a colourised PNG for visual inspection.
    Bright = potentially manipulated region.
    """
    ela_flat = compute_ela(pil_image.resize((224, 224)))  
    ela_map  = ela_flat.reshape(28, 28)
   
    ela_pil  = Image.fromarray((ela_map * 255).astype(np.uint8)).resize(
        (224, 224), Image.NEAREST
    )
    ela_pil.save(save_path)
    print(f'[ELA] Heatmap saved → {save_path}')


def predict(image_path: str, weights_path: str = WEIGHTS_FINAL):
    """
    Run full ensemble prediction on a single image.

    Args:
        image_path:   path to input image
        weights_path: path to trained fusion weights

    Returns:
        dict with keys: verdict, confidence, stream_weights, ela_path
    """

    # Load weights
    import os
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f'Weights file not found: {weights_path}')
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image file not found: {image_path}')

    print(f'[Inference] Loading weights: {weights_path}')
    print(f'[Inference] Loading image  : {image_path}')

    try:
        state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=DEVICE)

    fusion_model.load_state_dict(state)
    fusion_model.eval()
    all_streams.eval()

    pil, tensor = load_image(image_path)

    #forensic features
    from forensics import extract_forensic_features
    forensic_np   = extract_forensic_features(pil)
    forensic_feat = torch.tensor(forensic_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    #  neural stream 
    with torch.no_grad():
        neural_feats = all_streams(tensor)
        stream_feats = neural_feats + [forensic_feat]

        prob, alphas = fusion_model.predict(stream_feats)

    prob_val   = prob.item()
    alpha_vals = alphas.squeeze(0).cpu().numpy()

    # Verdict: outputs P(real), therefore >0.5 = real
    verdict    = 'REAL' if prob_val > 0.5 else 'DEEPFAKE'
    confidence = prob_val if prob_val > 0.5 else (1 - prob_val)

    # ELA  heatmap
    ela_path = image_path.rsplit('.', 1)[0] + '_ela.png'
    save_ela_heatmap(pil, ela_path)

    # Print report
    print('\n' + '='*55)
    print(f'  VERDICT:    {verdict}')
    print(f'  CONFIDENCE: {confidence*100:.1f}%')
    print(f'  Raw P(real):{prob_val:.4f}')
    print('='*55)
    print('  Stream attention weights (higher = more influential):')
    for name, weight in sorted(
        zip(STREAM_NAMES, alpha_vals), key=lambda x: x[1], reverse=True
    ):
        bar = '█' * int(weight * 40)
        print(f'    {name:<38} {weight:.3f}  {bar}')
    print('='*55)

    return {
        'verdict':        verdict,
        'confidence':     confidence,
        'prob_real':      prob_val,
        'stream_weights': dict(zip(STREAM_NAMES, alpha_vals.tolist())),
        'ela_path':       ela_path,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepfake detector — single image')
    parser.add_argument('--image',   required=True, help='Path to input image')
    parser.add_argument('--weights', default=WEIGHTS_FINAL,
                        help=f'Path to fusion weights (default: {WEIGHTS_FINAL})')
    args = parser.parse_args()

    result = predict(args.image, args.weights)
