import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    fig, ax = plt.subplots(figsize=(11, 14), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color palette - Academic publication style
    c_stream1_bg = "#EBF3FB"
    c_stream1_border = "#1B4F72"
    c_stream2_bg = "#EAFAF1"
    c_stream2_border = "#1E8449"
    c_gfu_bg = "#FEF9E7"
    c_gfu_border = "#B7950B"
    c_head_bg = "#FDEDEC"
    c_head_border = "#922B21"
    c_box = "#FFFFFF"
    c_text = "#17202A"

    # Background Subgraphs
    # Subgraph Stream 1
    rect_s1 = patches.FancyBboxPatch((3, 56), 44, 41, boxstyle="round,pad=1,rounding_size=2",
                                     linewidth=1.5, edgecolor=c_stream1_border, facecolor=c_stream1_bg, alpha=0.4, linestyle="--")
    ax.add_patch(rect_s1)
    ax.text(25, 95.5, "Stream 1: Technical Price Dynamics", fontsize=11, fontweight='bold', color=c_stream1_border, ha='center')

    # Subgraph Stream 2
    rect_s2 = patches.FancyBboxPatch((53, 56), 44, 41, boxstyle="round,pad=1,rounding_size=2",
                                     linewidth=1.5, edgecolor=c_stream2_border, facecolor=c_stream2_bg, alpha=0.4, linestyle="--")
    ax.add_patch(rect_s2)
    ax.text(75, 95.5, "Stream 2: Behavioral Sentiment Dynamics", fontsize=11, fontweight='bold', color=c_stream2_border, ha='center')

    # Subgraph GFU
    rect_gfu = patches.FancyBboxPatch((15, 23), 70, 23, boxstyle="round,pad=1,rounding_size=2",
                                      linewidth=1.8, edgecolor=c_gfu_border, facecolor=c_gfu_bg, alpha=0.45)
    ax.add_patch(rect_gfu)
    ax.text(50, 44.5, "Gated Cross-Modality Fusion Unit (GFU)", fontsize=11.5, fontweight='bold', color=c_gfu_border, ha='center')

    # Subgraph Head
    rect_head = patches.FancyBboxPatch((25, 2), 50, 17, boxstyle="round,pad=1,rounding_size=2",
                                       linewidth=1.5, edgecolor=c_head_border, facecolor=c_head_bg, alpha=0.4, linestyle="--")
    ax.add_patch(rect_head)
    ax.text(50, 17.5, "Classification Head & Optimization", fontsize=11, fontweight='bold', color=c_head_border, ha='center')

    # Helper function for drawing boxes
    def draw_box(x, y, w, h, title, subtitle, edgecolor="#2C3E50", facecolor="#FFFFFF", title_bold=True, fontsize=9):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1",
                                      linewidth=1.2, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, fontsize=fontsize, fontweight='bold' if title_bold else 'normal',
                    color=c_text, ha='center', va='center')
            ax.text(x + w/2, y + h*0.30, subtitle, fontsize=fontsize-1.5, color="#566573", ha='center', va='center')
        else:
            ax.text(x + w/2, y + h/2, title, fontsize=fontsize, fontweight='bold' if title_bold else 'normal',
                    color=c_text, ha='center', va='center')

    # Helper function for arrows
    def draw_arrow(x1, y1, x2, y2, color="#2C3E50", style="-|>", lw=1.5, dashed=False):
        arrowprops = dict(arrowstyle=style, color=color, lw=lw,
                          linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrowprops)

    # STREAM 1 BOXES
    # Input 1
    draw_box(7, 87, 36, 6, r"Stationary Price Input $\mathbf{X}_{\mathrm{price}}$", r"$W=18$ bars $\times$ 4 features (Return, GK-Vol, RSI, MACD)", c_stream1_border, "#FFFFFF")
    # LSTM 1A
    draw_box(7, 77, 36, 6.5, "Stacked LSTM Layer 1", "24 Hidden Units | Dropout 0.20 | Rec. Dropout 0.10", c_stream1_border, "#F4F6F6")
    # LSTM 1B
    draw_box(7, 67, 36, 6.5, "Stacked LSTM Layer 2", r"16 Hidden Units | Dropout 0.20 $\to \mathbf{h}_{\mathrm{price}}^{(2)}$", c_stream1_border, "#F4F6F6")
    # LayerNorm 1
    draw_box(9, 58, 32, 5.5, r"Layer Normalization ($\mathrm{LN}$)", r"$\tilde{\mathbf{h}}_{\mathrm{price}} \in \mathbb{R}^{16}$ (Zero-mean, Unit-variance)", c_stream1_border, "#D4E6F1")

    # Arrows Stream 1
    draw_arrow(25, 87, 25, 83.5, c_stream1_border)
    draw_arrow(25, 77, 25, 73.5, c_stream1_border)
    draw_arrow(25, 67, 25, 63.5, c_stream1_border)

    # STREAM 2 BOXES
    # Input 2
    draw_box(57, 87, 36, 6, r"Behavioral Sentiment Input $\mathbf{X}_{\mathrm{sent}}$", r"$W=18$ bars $\times$ 3 features (CFGI Score, EMA-6, Delta)", c_stream2_border, "#FFFFFF")
    # LSTM 2A
    draw_box(57, 77, 36, 6.5, "Stacked LSTM Layer 1", "16 Hidden Units | Dropout 0.20 | Rec. Dropout 0.10", c_stream2_border, "#F4F6F6")
    # LSTM 2B
    draw_box(57, 67, 36, 6.5, "Stacked LSTM Layer 2", r"16 Hidden Units | Dropout 0.20 $\to \mathbf{h}_{\mathrm{sent}}^{(2)}$", c_stream2_border, "#F4F6F6")
    # LayerNorm 2
    draw_box(59, 58, 32, 5.5, r"Layer Normalization ($\mathrm{LN}$)", r"$\tilde{\mathbf{h}}_{\mathrm{sent}} \in \mathbb{R}^{16}$ (Zero-mean, Unit-variance)", c_stream2_border, "#D5F5E3")

    # Arrows Stream 2
    draw_arrow(75, 87, 75, 83.5, c_stream2_border)
    draw_arrow(75, 77, 75, 73.5, c_stream2_border)
    draw_arrow(75, 67, 75, 63.5, c_stream2_border)

    # INTERMEDIATE: Vector Concatenation
    draw_box(30, 48.5, 40, 5, r"Multi-Modal Vector Concatenation", r"$[\tilde{\mathbf{h}}_{\mathrm{price}} \,\|\, \tilde{\mathbf{h}}_{\mathrm{sent}}] \in \mathbb{R}^{32}$", "#2C3E50", "#EBEDEF")
    draw_arrow(25, 58, 38, 53.5, c_stream1_border)
    draw_arrow(75, 58, 62, 53.5, c_stream2_border)

    # GFU BLOCK BOXES
    # Attention Gate
    draw_box(20, 36, 60, 6.5, r"Trainable Cross-Modality Gating Layer ($\mathbf{W}_g \in \mathbb{R}^{16 \times 32}, \mathbf{b}_g \in \mathbb{R}^{16}$)",
             r"$\mathbf{g}_t = \sigma(\mathbf{W}_g [\tilde{\mathbf{h}}_{\mathrm{price}} \,\|\, \tilde{\mathbf{h}}_{\mathrm{sent}}] + \mathbf{b}_g) \in (0, 1)^{16}$", c_gfu_border, "#FCF3CF")
    draw_arrow(50, 48.5, 50, 42.5, "#2C3E50")

    # Convex Combination Fusion
    draw_box(20, 25.5, 60, 6.5, r"Convex Combination Synthesis ($\odot$ Hadamard Product)",
             r"$\mathbf{h}_{\mathrm{fused}} = \mathbf{g}_t \odot \tilde{\mathbf{h}}_{\mathrm{price}} + (\mathbf{1} - \mathbf{g}_t) \odot \tilde{\mathbf{h}}_{\mathrm{sent}} \in \mathbb{R}^{16}$", c_gfu_border, "#FEF5E7")
    draw_arrow(50, 36, 50, 32, c_gfu_border)

    # Skip connections from LN1 and LN2 to Convex Combination
    draw_arrow(18, 58, 18, 28.5, c_stream1_border, dashed=True)
    draw_arrow(18, 28.5, 20, 28.5, c_stream1_border, dashed=True)

    draw_arrow(82, 58, 82, 28.5, c_stream2_border, dashed=True)
    draw_arrow(82, 28.5, 80, 28.5, c_stream2_border, dashed=True)

    # CLASSIFICATION HEAD
    draw_box(30, 10.5, 40, 5.5, "Dense Linear Layer + ReLU", r"$\mathbf{z}_t = \mathrm{Dropout}_{0.2}(\mathrm{ReLU}(\mathbf{W}_h \mathbf{h}_{\mathrm{fused}} + \mathbf{b}_h)) \in \mathbb{R}^{16}$", c_head_border, "#FADBD8")
    draw_arrow(50, 25.5, 50, 16, "#2C3E50")

    # Output Neuron
    draw_box(28, 3.5, 44, 5.5, "Calibrated Posterior Probability Output", r"$\hat{P}_t = \sigma(\mathbf{w}_o^T \mathbf{z}_t + b_o) = P(y_t = 1 \mid \mathbf{X})$", c_head_border, "#F2D7D5")
    draw_arrow(50, 10.5, 50, 9, c_head_border)

    plt.tight_layout()
    plt.savefig("results/figure1_model_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved results/figure1_model_architecture.png at 300 DPI.")

if __name__ == "__main__":
    draw_architecture()
