"""Detector de Nodulos Pulmonares (LUNA16) - Interface Gradio."""

import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from luna_data import load_candidates, xyz_to_irc, XYZ
from model import LunaModel
from inference import load_model

# --- Inicializacao ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL, MODEL_INFO = load_model(ROOT / "checkpoints" / "luna_model_best.pt", DEVICE)
ALL_CANDIDATES = load_candidates(require_on_disk=False)

print(f"Modelo carregado: epoca {MODEL_INFO['epoch']}, F1={MODEL_INFO['best_f1']:.4f}")
print(f"Candidatos: {len(ALL_CANDIDATES)}")
print(f"Device: {DEVICE}")


# --- Funcoes de classificacao ---

def classify_uploaded_ct(mhd_path, raw_path, batch_size=64):
    """Classifica candidatos de um CT a partir de arquivos .mhd/.raw."""
    tmpdir = tempfile.mkdtemp()
    mhd_dst = Path(tmpdir) / Path(mhd_path).name
    raw_dst = Path(tmpdir) / Path(raw_path).name
    shutil.copy2(mhd_path, mhd_dst)
    shutil.copy2(raw_path, raw_dst)

    series_uid = mhd_dst.stem

    candidates = [c for c in ALL_CANDIDATES if c.series_uid == series_uid]
    if not candidates:
        shutil.rmtree(tmpdir)
        return [], None

    ct_mhd = sitk.ReadImage(str(mhd_dst))
    hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
    hu_a.clip(-1000, 1000, hu_a)

    origin_xyz = XYZ(*ct_mhd.GetOrigin())
    vx_size_xyz = XYZ(*ct_mhd.GetSpacing())
    direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    shutil.rmtree(tmpdir)

    crop_size = (32, 48, 48)
    crops = []
    for c in candidates:
        center_irc = xyz_to_irc(c.center_xyz, origin_xyz, vx_size_xyz, direction_a)
        slices = []
        for axis, center_val in enumerate(center_irc):
            start = int(round(center_val - crop_size[axis] / 2))
            end = int(start + crop_size[axis])
            if start < 0:
                start, end = 0, int(crop_size[axis])
            if end > hu_a.shape[axis]:
                end = hu_a.shape[axis]
                start = int(end - crop_size[axis])
            slices.append(slice(start, end))
        crops.append(hu_a[tuple(slices)])

    crops_t = torch.from_numpy(np.stack(crops)).float().unsqueeze(1)

    all_probs = []
    with torch.no_grad():
        for start in range(0, len(crops_t), batch_size):
            batch = crops_t[start:start + batch_size].to(DEVICE)
            _, probs = MODEL(batch)
            all_probs.extend(probs[:, 1].cpu().numpy())

    results = []
    for c, prob, crop in zip(candidates, all_probs, crops):
        results.append({
            "series_uid": c.series_uid,
            "center_xyz": c.center_xyz,
            "probability": float(prob),
            "is_nodule": c.is_nodule,
            "crop": crop,
        })

    results.sort(key=lambda r: r["probability"], reverse=True)
    return results, series_uid


# --- Funcoes de apresentacao ---

def build_dataframe(suspects):
    """Monta DataFrame apenas com os candidatos suspeitos."""
    rows = []
    for i, r in enumerate(suspects):
        rows.append({
            "#": i + 1,
            "Probabilidade": f"{r['probability']:.4f}",
            "X": f"{r['center_xyz'][0]:.1f}",
            "Y": f"{r['center_xyz'][1]:.1f}",
            "Z": f"{r['center_xyz'][2]:.1f}",
        })
    return pd.DataFrame(rows)


def build_figure(suspects):
    """Monta grid com fatia axial central de cada suspeito (3 por linha)."""
    n = len(suspects)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.text(0.5, 0.5, "Nenhum candidato suspeito",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    for i in range(n):
        r = suspects[i]
        crop = r["crop"]
        axes[i].imshow(crop[crop.shape[0] // 2], cmap="gray")
        axes[i].set_title(f"#{i+1}  prob={r['probability']:.3f}", fontsize=11)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Candidatos suspeitos", fontsize=13)
    plt.tight_layout()
    return fig


# --- Funcao principal ---

def analyze_ct(files, threshold):
    """Funcao principal da interface Gradio."""
    if files is None or len(files) < 2:
        return "Faca upload de 2 arquivos: um .mhd e um .raw do mesmo CT.", None, None

    mhd_path = None
    raw_path = None
    for f in files:
        name = f if isinstance(f, str) else f.name
        if name.endswith(".mhd"):
            mhd_path = name
        elif name.endswith(".raw"):
            raw_path = name

    if mhd_path is None or raw_path is None:
        return "Nao encontrei os arquivos .mhd e .raw. Verifique o upload.", None, None

    results, uid = classify_uploaded_ct(mhd_path, raw_path)

    if not results:
        return (
            "Nenhum candidato encontrado para este CT.\n"
            "Isso pode significar que o CT nao faz parte do dataset LUNA16, "
            "ou que o series_uid nao esta no candidates.csv.",
            None, None,
        )

    suspects = [r for r in results if r["probability"] >= threshold]

    resumo = (
        f"CT: {uid}\n"
        f"Candidatos analisados: {len(results)}\n"
        f"Suspeitos (prob >= {threshold}): {len(suspects)}\n"
    )
    if suspects:
        resumo += f"Maior probabilidade: {suspects[0]['probability']:.4f}"
    else:
        resumo += "Nenhuma regiao suspeita encontrada."

    df = build_dataframe(suspects)
    fig = build_figure(suspects)

    return resumo, df, fig


# --- Interface Gradio ---

with gr.Blocks(title="Detector de Nodulos Pulmonares") as demo:
    gr.Markdown(
        "## Detector de Nodulos Pulmonares (LUNA16)\n"
        "Faca upload dos arquivos `.mhd` e `.raw` de um CT scan do dataset LUNA16. "
        "O modelo analisa todas as regioes candidatas e mostra as suspeitas.\n\n"
        "**Importante**: so funciona com CTs do LUNA16 (precisa das coordenadas do `candidates.csv`)."
    )

    with gr.Row():
        file_input = gr.File(
            file_count="multiple",
            file_types=[".mhd", ".raw"],
            label="Arquivos do CT (.mhd e .raw)",
        )
        threshold_slider = gr.Slider(
            minimum=0.1, maximum=0.95, value=0.5, step=0.05,
            label="Threshold",
        )

    btn = gr.Button("Classificar", variant="primary")

    resumo_output = gr.Textbox(label="Resumo", lines=4)
    df_output = gr.Dataframe(label="Regioes suspeitas")
    plot_output = gr.Plot(label="Visualizacao dos suspeitos")

    btn.click(
        fn=analyze_ct,
        inputs=[file_input, threshold_slider],
        outputs=[resumo_output, df_output, plot_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
