# Simulated flat-lay photos

`S.jpg`, `M.jpg`, `L.jpg`, `XL.jpg` and `XL_mislabeled.jpg` (the XL with a chest 1 cm smaller than L) are **simulated**: drawn by `fitcore.garment.synthetic`, a shirt on a brown surface next to a white A4 sheet, seen through a mild random perspective. They are not photographs of real garments.

`*.landmarks.json` hold the landmark points in photo pixels, so the photos can be measured without a model. `truth.json` holds the true values.

```
python -m fitcore.garment.demo samples/garment/XL_mislabeled.jpg --category shirt --landmarks samples/garment/XL_mislabeled.landmarks.json
```

Rewrite them with `python -m fitcore.garment.demo --make-samples samples/garment`.
