# ----- tiny helpers -----
@torch.no_grad()
def save_recons_grid(model, loader, device, path, n=8):
    model.eval()
    xb = next(iter(loader))[:n].to(device)
    logits, _, _ = model(xb)
    rec = torch.sigmoid(logits)
    grid = make_grid(torch.cat([xb.cpu(), rec.cpu()], 0), nrow=n, pad_value=1.0)
    save_image(grid, path)

@torch.no_grad()
def save_umap(model, loader, device, path, max_points=4000):
    # collect mus
    model.eval()
    zs=[]
    for x in loader:
        x = x.to(device)
        _, mu, _ = model(x)
        zs.append(mu.cpu())
        if sum(t.size(0) for t in zs) >= max_points: break
    z = torch.cat(zs).numpy()
    # 2D: use z; else reduce with UMAP
    if z.shape[1] == 2:
        emb = z
    else:
        try:
            from umap import UMAP
        except:
            !pip -q install umap-learn
            from umap import UMAP
        emb = UMAP(n_components=2, min_dist=0.1, n_neighbors=15, random_state=0).fit_transform(z)
    plt.figure(figsize=(6,6))
    plt.scatter(emb[:,0], emb[:,1], s=6, alpha=0.6)
    plt.title("Latent manifold (UMAP if z>2)")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
