"""
WorldSim — Local Pygame Player
================================
Play your trained world model live with WASD keys.

Usage:
    python play.py

    or with custom paths:
    python play.py --model path/to/worldmodel_v2.pt --start path/to/frame_00000.jpg

Requirements:
    pip install -r requirements.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="WorldSim pygame player")
parser.add_argument("--model",  default="worldmodel_v2.pt", help="Path to .pt model file")
parser.add_argument("--start",  default="frame_00000.jpg",  help="Path to starting frame image")
parser.add_argument("--width",  type=int, default=640,      help="Window width")
parser.add_argument("--height", type=int, default=480,      help="Window height")
parser.add_argument("--fps",    type=int, default=15,       help="Target FPS (match training FPS)")
args = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[WorldSim] Device: {DEVICE}")

# ── Hyperparameters — must match training exactly ─────────────────────────────
IMG_H, IMG_W = 128, 128
HIDDEN_DIM   = 512
LATENT_DIM   = 256
ACTION_DIM   = 64
N_ACTIONS    = 9

# ── Action space ──────────────────────────────────────────────────────────────
# dx, dy ∈ {-1, 0, 1}  →  index = (dx+1)*3 + (dy+1)
def action_idx(dx, dy):
    return (dx + 1) * 3 + (dy + 1)

ACTION_LABELS = {
    (0,  0): "STOP",
    (0,  1): "W  — forward",
    (0, -1): "S  — backward",
    (-1, 0): "A  — left",
    (1,  0): "D  — right",
    (-1, 1): "W+A",
    (1,  1): "W+D",
    (-1,-1): "S+A",
    (1, -1): "S+D",
}

# ── Model architecture (identical to training) ────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GroupNorm(8, ch),
        )
    def forward(self, x): return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   32, 4, 2, 1), nn.SiLU(), ResBlock(32),
            nn.Conv2d(32,  64, 4, 2, 1), nn.SiLU(), ResBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(), ResBlock(128),
            nn.Conv2d(128,256, 4, 2, 1), nn.SiLU(), ResBlock(256),
            nn.Conv2d(256,512, 4, 2, 1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, LATENT_DIM),
        )
    def forward(self, x): return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc  = nn.Linear(HIDDEN_DIM + LATENT_DIM, 512 * 4 * 4)
        self.net = nn.Sequential(
            ResBlock(512),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.SiLU(), ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(), ResBlock(128),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.SiLU(), ResBlock(64),
            nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.SiLU(), ResBlock(32),
            nn.ConvTranspose2d( 32,   3, 4, 2, 1), nn.Tanh(),
        )
    def forward(self, z): return self.net(self.fc(z).view(-1, 512, 4, 4))


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder      = Encoder()
        self.decoder      = Decoder()
        self.action_emb   = nn.Embedding(N_ACTIONS, ACTION_DIM)
        self.gru          = nn.GRUCell(LATENT_DIM + ACTION_DIM, HIDDEN_DIM)
        self.prior_fc     = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM * 2),
        )
        self.posterior_fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM, HIDDEN_DIM), nn.SiLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM * 2),
        )

    @torch.no_grad()
    def step(self, h, frame, a_idx):
        """One step: current frame + action → next frame + updated hidden state."""
        a    = torch.tensor([a_idx], device=DEVICE)
        ae   = self.action_emb(a)
        z    = self.encoder(frame)
        mu, _= self.posterior_fc(torch.cat([h, z], -1)).chunk(2, -1)
        h    = self.gru(torch.cat([mu, ae], -1), h)
        return h, self.decoder(torch.cat([h, mu], -1))


# ── Load model ────────────────────────────────────────────────────────────────
model_path = Path(args.model)
if not model_path.exists():
    print(f"[WorldSim] ERROR: Model not found: {model_path}")
    print("  → Train first using notebooks/train.ipynb on Kaggle")
    print("  → Download worldmodel_v2.pt to this folder")
    sys.exit(1)

model = RSSM().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"[WorldSim] Model loaded: {model_path}  ({n_params/1e6:.1f}M params)")

# ── Load starting frame ───────────────────────────────────────────────────────
start_path = Path(args.start)
if not start_path.exists():
    print(f"[WorldSim] ERROR: Starting frame not found: {start_path}")
    print("  → Copy any frame_XXXXX.jpg from your dataset to this folder")
    sys.exit(1)

transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

start_frame = transform(Image.open(start_path).convert("RGB")).unsqueeze(0).to(DEVICE)
print(f"[WorldSim] Starting frame: {start_path}")

# ── Game state ────────────────────────────────────────────────────────────────
h     = torch.zeros(1, HIDDEN_DIM, device=DEVICE)
frame = start_frame.clone()

def reset():
    global h, frame
    h     = torch.zeros(1, HIDDEN_DIM, device=DEVICE)
    frame = start_frame.clone()
    print("[WorldSim] Reset.")

def tensor_to_surface(t, w, h_px):
    arr = (t.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((w, h_px), Image.LANCZOS)
    return pygame.surfarray.make_surface(np.array(img).swapaxes(0, 1))

# ── Pygame init ───────────────────────────────────────────────────────────────
pygame.init()
WIN_W, WIN_H = args.width, args.height
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("WorldSim  |  WASD = move  |  R = reset  |  ESC = quit")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 16)
font_s = pygame.font.SysFont("monospace", 13)

print("\n[WorldSim] Controls:")
print("  W / S / A / D  — move camera")
print("  R              — reset to starting frame")
print("  ESC / Q        — quit\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
running        = True
current_action = (0, 0)

while running:
    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            if event.key == pygame.K_r:
                reset()

    # Continuous key input
    keys = pygame.key.get_pressed()
    dx, dy = 0, 0
    if keys[pygame.K_w]: dy =  1
    if keys[pygame.K_s]: dy = -1
    if keys[pygame.K_a]: dx = -1
    if keys[pygame.K_d]: dx =  1
    current_action = (dx, dy)

    # Model step
    h, frame = model.step(h, frame, action_idx(dx, dy))

    # Render
    screen.blit(tensor_to_surface(frame, WIN_W, WIN_H), (0, 0))

    # Semi-transparent HUD background
    hud = pygame.Surface((220, 80), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 140))
    screen.blit(hud, (8, 8))

    screen.blit(font.render(f"FPS:    {clock.get_fps():.0f}", True, (255, 220, 0)), (14, 12))
    screen.blit(font.render(f"Action: {ACTION_LABELS[current_action]}", True, (255, 220, 0)), (14, 30))
    screen.blit(font_s.render("R=reset   ESC=quit", True, (180, 180, 180)), (14, 52))
    screen.blit(font_s.render(f"Device: {DEVICE}", True, (140, 140, 140)), (14, 68))

    pygame.display.flip()
    clock.tick(args.fps)

pygame.quit()
print("[WorldSim] Bye!")
