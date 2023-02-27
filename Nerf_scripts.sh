# these options from nvdiffrec
DATASET_PATH ="F:\\DifferentiableRendering\\nvdiffrec\\data\\nerd\\rubik"
mkdir "$DATASET_PATH/colmap_text"

colmap model_converter --input_path "$DATASET_PATH/sparse/0" \
    --output_path "$DATASET_PATH/colmap_text" \
    --output_type TXT

python scripts/colmap2nerf.py --colmap_matcher exhaustive \
    --images "$DATASET_PATH/images" \
    --colmap_db "$DATASET_PATH/colmap.db" \
    --text "$DATASET_PATH/colmap_text" \
    --out "$DATASET_PATH/transforms.json"

cd "$DATASET_PATH"

python "$SCRIPT_DIR/scripts/colmap2nerf.py" --colmap_matcher exhaustive \
    --images images \
    --colmap_db colmap.db \
    --text colmap_text \
    --out transforms.json

cd "$SCRIPT_DIR"

python scripts/run.py --scene="$DATASET_PATH/" \
    --mode=nerf \
    --screenshot_transforms="$DATASET_PATH/transforms.json" \
    --screenshot_w=1024 \
    --screenshot_h=1024 \
    --screenshot_dir="$DATASET_PATH/screenshots" \
    --save_snapshot="$DATASET_PATH/snapshot.msgpack" \
    --n_steps=1000