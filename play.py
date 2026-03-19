-    python play.py

    or with custom paths:
    python play.py --model worldmodel_v2.pt --start frame_00000.jpg

Requirements:
    pip install pygame torch torchvision pillow numpy
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
parser.add_argument("--model",  default="worldmodel_v2.pt")
parser.add_argument("--start",  default="frame_00000.jpg")
parser.add_argument("--width",  type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps",    type=int, default=15)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[WorldSim] Device: {DEVICE}")

# ── Hyperparameters (must match training) ────────────────────────────────────
IMG_H, IMG_W = 128, 128
HIDDEN_DIM   = 512
LATENT_DIM   = 256
ACTION_DIM   = 64
N_ACTIONS    = 9

def action_idx(dx, dy):
    return (dx + 1) * 3 + (dy + 1)

# ── Architecture ──────────────────────────────────────────────────────────────
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
    sys.exit(1)

model = RSSM().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
model.eval()
print(f"[WorldSim] Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

# ── Load start frame ──────────────────────────────────────────────────────────
start_path = Path(args.start)
if not start_path.exists():
    print(f"[WorldSim] ERROR: Start frame not found: {start_path}")
    sys.exit(1)

transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
start_frame = transform(Image.open(start_path).convert("RGB")).unsqueeze(0).to(DEVICE)
print(f"[WorldSim] Start frame: {start_path}")

# ── Game state ────────────────────────────────────────────────────────────────
h     = torch.zeros(1, HIDDEN_DIM, device=DEVICE)
frame = start_frame.clone()
step_count = 0

def reset():
    global h, frame, step_count
    h, frame, step_count = torch.zeros(1, HIDDEN_DIM, device=DEVICE), start_frame.clone(), 0
    print("[WorldSim] Reset.")

def tensor_to_surface(t, w, hh):
    arr = (t.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((w, hh), Image.LANCZOS)
    return pygame.surfarray.make_surface(np.array(img).swapaxes(0, 1))

# ── WASD arrow pad renderer ───────────────────────────────────────────────────
def draw_action_pad(surface, dx, dy, x, y):
    """
    Draw a D-pad style arrow indicator at position (x, y) — bottom-right corner.
    dx, dy: current action vector
    """
    PAD   = 38   # size of each arrow button
    GAP   = 4    # gap between buttons
    R     = 10   # corner radius

    # Colors
    COL_BG       = (20,  20,  20,  180)   # dark background panel
    COL_INACTIVE = (60,  60,  60,  200)   # inactive arrow
    COL_ACTIVE   = (255, 220,  0,  255)   # active arrow (yellow)
    COL_ARROW    = (255, 255, 255, 255)   # arrow symbol

    # Panel background
    panel_w = PAD * 3 + GAP * 4
    panel_h = PAD * 3 + GAP * 4
    panel   = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    pygame.draw.rect(panel, COL_BG, (0, 0, panel_w, panel_h), border_radius=12)

    # Arrow button positions: (label, col, row, action_dx, action_dy)
    buttons = [
        ("▲", 1, 0,  0,  1),   # W — top center
        ("◀", 0, 1, -1,  0),   # A — middle left
        ("●", 1, 1,  0,  0),   # STOP — center
        ("▶", 2, 1,  1,  0),   # D — middle right
        ("▼", 1, 2,  0, -1),   # S — bottom center
    ]

    for symbol, col, row, bdx, bdy in buttons:
        bx = GAP + col * (PAD + GAP)
        by = GAP + row * (PAD + GAP)
        is_active = (bdx == dx and bdy == dy and not (bdx == 0 and bdy == 0 and (dx != 0 or dy != 0)))
        if symbol == "●":
            is_active = (dx == 0 and dy == 0)
        btn_col = COL_ACTIVE if is_active else COL_INACTIVE
        pygame.draw.rect(panel, btn_col, (bx, by, PAD, PAD), border_radius=R)
        # Arrow symbol
        font_arrow = pygame.font.SysFont("segoeui", 20, bold=True)
        sym_surf = font_arrow.render(symbol, True, COL_ARROW if is_active else (150, 150, 150))
        sym_rect = sym_surf.get_rect(center=(bx + PAD // 2, by + PAD // 2))
        panel.blit(sym_surf, sym_rect)

    surface.blit(panel, (x - panel_w, y - panel_h))

# ── Pygame init ───────────────────────────────────────────────────────────────
pygame.init()
WIN_W, WIN_H = args.width, args.height
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("WorldSim  |  WASD = move  |  R = reset  |  ESC = quit")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 15)
font_s = pygame.font.SysFont("monospace", 12)

print("\n[WorldSim] Controls: WASD = move | R = reset | ESC = quit\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
running        = True
current_action = (0, 0)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            if event.key == pygame.K_r:
                reset()

    # Continuous WASD input
    keys = pygame.key.get_pressed()
    dx, dy = 0, 0
    if keys[pygame.K_w]: dy =  1
    if keys[pygame.K_s]: dy = -1
    if keys[pygame.K_a]: dx = -1
    if keys[pygame.K_d]: dx =  1
    current_action = (dx, dy)

    # Model step
    h, frame = model.step(h, frame, action_idx(dx, dy))
    step_count += 1

    # Render frame
    screen.blit(tensor_to_surface(frame, WIN_W, WIN_H), (0, 0))

    # ── Top-left HUD ──────────────────────────────────────────────────────────
    hud = pygame.Surface((160, 60), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 140))
    screen.blit(hud, (8, 8))
    screen.blit(font.render(f"FPS:  {clock.get_fps():.0f}", True, (255, 220, 0)), (14, 13))
    screen.blit(font.render(f"Step: {step_count}", True, (255, 220, 0)), (14, 31))
    screen.blit(font_s.render("R=reset  ESC=quit", True, (160, 160, 160)), (14, 52))

    # ── Bottom-right action pad ───────────────────────────────────────────────
    draw_action_pad(screen, dx, dy, WIN_W - 10, WIN_H - 10)

    pygame.display.flip()
    clock.tick(args.fps)

pygame.quit()
print("[WorldSim] Bye!")
