
import numpy as np, cv2, torch, torch.nn as nn, gradio as gr
from torchvision import models

BUNDLE = "footprint_bundle.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

B = torch.load(BUNDLE, map_location="cpu", weights_only=False)
IN_H, IN_W = B["in_h"], B["in_w"]
MEAN = np.array(B["mean"], np.float32)
STD = np.array(B["std"], np.float32)
REG = B["reg_targets"]                      # ['height_cm', 'weight_kg', 'age']
FOLD_STATS = B["fold_stats"]                # per fold: {target: [mean, std]}


# ----------------------------------------------------------------- the model
class Net(nn.Module):
    def __init__(self, arch="resnet50"):
        super().__init__()
        bb = (models.resnet50() if arch == "resnet50" else models.resnet18())
        self.stem = nn.Sequential(*list(bb.children())[:-1])
        d = bb.fc.in_features
        self.drop = nn.Dropout(0.5)
        self.reg = nn.Linear(d, 3)
        self.cls = nn.Linear(d, 1)
        self.sid = nn.Linear(d, 1)

    def forward(self, x):
        f = self.drop(self.stem(x).flatten(1))
        return self.reg(f), self.cls(f).squeeze(1), self.sid(f).squeeze(1)


NETS = []
for sd in B["state_dicts"]:
    n = Net(B["arch"])
    n.load_state_dict(sd)
    n.eval().to(DEV)
    NETS.append(n)
print(f"loaded {len(NETS)} folds on {DEV}")


# --------------------------------------------------- preprocessing (identical
# to the notebook — any difference here silently wrecks the predictions)
def paper_crop(bgr):
    h, w = bgr.shape[:2]
    s = 700 / max(h, w)
    sm = cv2.resize(bgr, None, fx=s, fy=s)
    L, A, Bc = cv2.split(cv2.cvtColor(sm, cv2.COLOR_BGR2LAB))
    chroma = np.sqrt((A.astype(np.float32) - 128) ** 2 + (Bc.astype(np.float32) - 128) ** 2)
    paper = ((L > 110) & (chroma < 32)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, k, 3)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, k, 2)
    n, lb, st, _ = cv2.connectedComponentsWithStats(paper, 8)
    if n <= 1:
        return bgr
    i = 1 + int(np.argmax(st[1:, 4]))
    if st[i, 4] < 0.25 * paper.size:
        return bgr
    x, y, bw, bh, _ = st[i]
    x, y, bw, bh = [int(v / s) for v in (x, y, bw, bh)]
    return bgr[max(y, 0):min(y + bh, h), max(x, 0):min(x + bw, w)]


def preprocess(bgr):
    sheet = paper_crop(bgr)
    if sheet.size == 0:
        sheet = bgr
    h, w = sheet.shape[:2]
    sheet = sheet[int(0.15 * h):, :]           # blank the handwriting band
    h, w = sheet.shape[:2]
    s = min(IN_H / h, IN_W / w)
    nh, nw = max(int(h * s), 1), max(int(w * s), 1)
    canvas = np.full((IN_H, IN_W, 3), 255, np.uint8)
    t, l = (IN_H - nh) // 2, (IN_W - nw) // 2
    canvas[t:t + nh, l:l + nw] = cv2.resize(sheet, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas


# ------------------------------------------------------------------ inference
@torch.no_grad()
def predict(rgb):
    if rgb is None:
        return "Upload a footprint image first.", None
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    canvas = preprocess(bgr)
    x = ((canvas.astype(np.float32) / 255.0) - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(DEV)
    xf = torch.flip(x, dims=[3])               # flip-TTA, same as the notebook

    regs, sexes, sides = [], [], []
    for net, stats in zip(NETS, FOLD_STATS):
        r1, g1, s1 = net(x)
        r2, g2, s2 = net(xf)
        r = ((r1 + r2) / 2)[0].cpu().numpy()
        # each fold z-scored with its own train-fold mean/std — undo it per fold
        regs.append([r[i] * stats[c][1] + stats[c][0] for i, c in enumerate(REG)])
        sexes.append(float((torch.sigmoid(g1) + torch.sigmoid(g2)) / 2))
        # a mirrored image swaps the side, so invert the flipped prediction
        sides.append(float((torch.sigmoid(s1) + (1 - torch.sigmoid(s2))) / 2))

    reg = np.mean(regs, axis=0)
    p_male = float(np.mean(sexes))
    p_right = float(np.mean(sides))
    v = dict(zip(REG, reg))

    out = (
        f"## Prediction\n\n"
        f"| target | value | confidence |\n|---|---|---|\n"
        f"| height | **{v['height_cm']:.1f} cm** | ±7 cm typical error |\n"
        f"| weight | **{v['weight_kg']:.1f} kg** | ±10 kg typical error |\n"
        f"| age | **{v['age']:.1f} years** | ±5 years typical error |\n"
        f"| sex | **{'Male' if p_male > .5 else 'Female'}** | {max(p_male, 1-p_male)*100:.0f}% |\n"
        f"| foot | **{'Right' if p_right > .5 else 'Left'}** | {max(p_right, 1-p_right)*100:.0f}% |\n\n"
    )
    return out, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


with gr.Blocks(title="BDFP footprint model") as demo:
    gr.Markdown("# Footprint → height, weight, age, sex, left/right\n"
                "Upload a photo of an inked footprint on a white sheet.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="footprint photo")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            txt = gr.Markdown()
            prep = gr.Image(label="what the model actually sees (448×320)")
    btn.click(predict, inp, [txt, prep])
    inp.upload(predict, inp, [txt, prep])

if __name__ == "__main__":
    demo.launch()
