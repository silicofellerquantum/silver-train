import nbformat, sys, base64, os
path = sys.argv[1]; imgdir = sys.argv[2] if len(sys.argv) > 2 else None
nb = nbformat.read(path, 4)
img_i = 0
for idx, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    print(f"\n======== CELL {idx} ========")
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            print(o.text.rstrip()[:6000])
        elif o.output_type in ("display_data", "execute_result"):
            if "image/png" in o.data and imgdir:
                os.makedirs(imgdir, exist_ok=True)
                p = os.path.join(imgdir, f"cell{idx}_{img_i}.png"); img_i += 1
                open(p, "wb").write(base64.b64decode(o.data["image/png"]))
                print(f"[image saved: {p}]")
            elif "text/plain" in o.data:
                print("[display]", o.data["text/plain"][:500])
        elif o.output_type == "error":
            print("ERROR:", o.ename, o.evalue[:2000])
