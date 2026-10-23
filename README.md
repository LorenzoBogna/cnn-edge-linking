# CNN + g/theta edge linking

This project implements a learned edge detector followed by geometric edge linking.

```text
RGB image
   |
   +----> TinyEdgeNet ----> edge probability map ---+
   |                                               |
   +----> Sobel ----> g, theta ---------------------+
                                                   |
                                                   v
                                      orientation-aware linking
                                                   |
                                                   v
                                            linked edge map
                                                   |
                                                   v
                                               contours
```

## 1. Training

The CNN receives RGB images only. BSDS500 provides human boundary annotations, which are averaged into a soft boundary target.

Expected dataset structure:

```text
BSDS500/
  data/
    images/
      train/
      val/
      test/
    groundTruth/
      train/
      val/
      test/
```

Normal training:

```bash
python train.py \
  --bsds-root BSDS500 \
  --epochs 30
```

Useful training options:

```text
--batch-size 8
--lr 1e-3
--weight-decay 1e-4
--pos-weight 4.0
--height 320
--width 320
--out checkpoints
```

The best validation checkpoint is saved as:

```text
checkpoints/best.pt
```

The latest checkpoint is saved as:

```text
checkpoints/last.pt
```

## 2. Diagnostic: overfit one image

If the CNN produces poor edge maps, first check whether the training pipeline can deliberately memorize a single BSDS500 training sample.

Run:

```bash
python train.py \
  --bsds-root BSDS500 \
  --overfit-one \
  --overfit-index 0 \
  --overfit-steps 1000 \
  --save-every 100
```

This mode does not run the normal train/validation loop. It repeatedly optimizes the same training image for the requested number of steps.

Diagnostic files are saved under:

```text
checkpoints/overfit_one/
  step_0000/
  step_0001/
  step_0100/
  step_0200/
  ...
  step_1000/
  overfit_last.pt
```

Each saved step contains:

```text
input.png
ground_truth.png
probability.png
threshold.png
panel_input_gt_prob_threshold.png
```

The panel order is:

```text
input | soft ground truth | CNN probability | thresholded prediction
```

The threshold used only for the diagnostic binary image can be changed with:

```bash
--diag-threshold 0.20
```

For example:

```bash
python train.py \
  --bsds-root BSDS500 \
  --overfit-one \
  --overfit-image 2092.jpg \
  --overfit-steps 1000 \
  --save-every 100
```

During this test the program also prints probability statistics and the mean absolute error (MAE) between the CNN probability map and the soft BSDS target:

```text
min, max, mean, p50, p90, p95, p99, MAE
```

### How to interpret the overfit test

A healthy training pipeline should show a clear progression:

```text
step 0       random / almost uniform prediction
step 100     main structures start appearing
step 500     prediction resembles the selected ground truth
step 1000    prediction strongly follows that ground truth
```

The important comparison is `ground_truth.png` versus `probability.png`.

If the network cannot fit even one image after many optimization steps, investigate the training pipeline before testing generalization. Typical candidates are the loss, target construction, resize/interpolation, optimizer settings, or insufficient model capacity.

If the network fits one image well but performs poorly on validation images, the optimization pipeline is probably working and the problem is instead generalization/model capacity/training data.

BSDS500 targets are soft rather than purely binary, so the probability map is not expected to be identical to a one-pixel binary Canny edge map. Also, a threshold of 0.5 is not automatically optimal.

## 3. Testing different class weights

The training loss is weighted BCE + Dice. The positive-class weight is configurable:

```bash
python train.py \
  --bsds-root BSDS500 \
  --epochs 30 \
  --pos-weight 4.0
```

To test whether positive edges are underweighted, compare runs such as:

```bash
--pos-weight 4
--pos-weight 8
--pos-weight 12
```

Do not choose the final value from visual inspection alone; validate it on the BSDS500 validation split.

## 4. Inference

Example:

```bash
python infer.py image.jpg \
  --weights checkpoints/best.pt
```

Outputs:

```text
output/
  01_cnn_probability.png
  02_cnn_threshold.png
  03_linked_edges.png
  04_colored_contours.png
```

The inference thresholds can be changed, for example:

```bash
python infer.py image.jpg \
  --weights checkpoints/best.pt \
  --low-prob 0.10 \
  --high-prob 0.30
```

Because the CNN is trained against soft BSDS500 boundaries, `high-prob=0.50` should not be assumed to be optimal. Thresholds should eventually be selected on the validation set.

## 5. What happens at test time

1. The CNN receives the RGB image.
2. The CNN outputs edge probabilities.
3. Sobel is applied to the original image.
4. Gradient magnitude and orientation are computed:

```text
g = sqrt(Gx^2 + Gy^2)
theta = atan2(Gy, Gx)
```

5. Strong CNN edges are accepted immediately.
6. Weak CNN edges are accepted only when they are:
   - connected to an accepted edge;
   - supported by sufficient gradient magnitude;
   - compatible in orientation.
7. Optional small gaps are bridged along the estimated edge tangent.
8. Contours are extracted.
9. Contours are drawn with different colors.

## 6. Key conceptual point

The CNN and the modified Canny-like stage have different jobs.

The CNN learns **where** perceptually meaningful boundaries are likely to occur.

The `g + theta` linking stage decides **how** candidate edge fragments should be connected geometrically.

Therefore the final algorithm is not:

```text
CNN -> full Canny
```

It is more accurately described as:

```text
CNN candidates + g/theta from original image -> linked edges
```

or:

```text
learned edge detection + gradient-guided edge linking
```


Risultati validation
non aug
thr precision recall F1 
----------------------------------- 
0.05 0.1196 0.7858 0.2077
0.10 0.1692 0.6742 0.2706 
0.15 0.2063 0.6035 0.3075 
0.20 0.2380 0.5479 0.3319 
0.25 0.2670 0.5013 0.3484 
0.30 0.2949 0.4596 0.3593 
0.35 0.3228 0.4217 0.3657 
0.40 0.3509 0.3853 0.3673 
0.45 0.3802 0.3499 0.3644 
0.50 0.4117 0.3152 0.3571 
0.55 0.4453 0.2803 0.3441 
0.60 0.4823 0.2435 0.3236 
0.65 0.5231 0.2050 0.2945 
0.70 0.5698 0.1646 0.2554 
0.75 0.6264 0.1221 0.2043 
0.80 0.6857 0.0787 0.1413 
0.85 0.7426 0.0389 0.0740 
0.90 0.7974 0.0116 0.0229 
BEST threshold=0.40 precision=0.3509 recall=0.3853 F1=0.3673

aug
thr     precision   recall      F1
-----------------------------------
0.05    0.0928      0.8433    0.1671
0.10    0.1395      0.7195    0.2338
0.15    0.1801      0.6267    0.2797
0.20    0.2158      0.5554    0.3109
0.25    0.2480      0.4966    0.3308
0.30    0.2776      0.4451    0.3419
0.35    0.3059      0.3979    0.3459
0.40    0.3348      0.3530    0.3436
0.45    0.3646      0.3084    0.3342
0.50    0.3973      0.2638    0.3170
0.55    0.4336      0.2177    0.2899
0.60    0.4705      0.1678    0.2474
0.65    0.5111      0.1144    0.1870
0.70    0.5588      0.0647    0.1159
0.75    0.6103      0.0286    0.0546
0.80    0.6578      0.0089    0.0175
0.85    0.7566      0.0018    0.0035
0.90    0.0000      0.0000    0.0000

BEST
threshold=0.35 precision=0.3059 recall=0.3979 F1=0.3459

aug light
thr     precision   recall      F1
-----------------------------------
0.05    0.1054      0.8185    0.1868
0.10    0.1538      0.6897    0.2515
0.15    0.1918      0.6014    0.2909
0.20    0.2260      0.5311    0.3171
0.25    0.2574      0.4724    0.3333
0.30    0.2858      0.4214    0.3406
0.35    0.3124      0.3752    0.3410
0.40    0.3385      0.3326    0.3355
0.45    0.3642      0.2918    0.3240
0.50    0.3909      0.2517    0.3062
0.55    0.4209      0.2113    0.2814
0.60    0.4525      0.1688    0.2459
0.65    0.4884      0.1237    0.1974
0.70    0.5307      0.0792    0.1379
0.75    0.5878      0.0418    0.0781
0.80    0.6462      0.0163    0.0317
0.85    0.6757      0.0035    0.0070
0.90    0.7547      0.0003    0.0005

BEST
threshold=0.35 precision=0.3124 recall=0.3752 F1=0.3410

w8
thr     precision   recall      F1
-----------------------------------
0.05    0.0746      0.8999    0.1378
0.10    0.1020      0.8237    0.1816
0.15    0.1257      0.7643    0.2158
0.20    0.1476      0.7117    0.2445
0.25    0.1694      0.6636    0.2699
0.30    0.1919      0.6180    0.2929
0.35    0.2157      0.5741    0.3135
0.40    0.2411      0.5299    0.3314
0.45    0.2694      0.4846    0.3463
0.50    0.3011      0.4380    0.3569
0.55    0.3373      0.3893    0.3615
0.60    0.3797      0.3390    0.3582
0.65    0.4282      0.2851    0.3423
0.70    0.4840      0.2283    0.3102
0.75    0.5490      0.1692    0.2586
0.80    0.6184      0.1094    0.1860
0.85    0.6804      0.0550    0.1017
0.90    0.7434      0.0182    0.0354

BEST
threshold=0.55 precision=0.3373 recall=0.3893 F1=0.3615

pw12

thr     precision   recall      F1
-----------------------------------
0.05    0.0759      0.8974    0.1399
0.10    0.1131      0.8016    0.1983
0.15    0.1434      0.7292    0.2397
0.20    0.1704      0.6701    0.2717
0.25    0.1955      0.6193    0.2972
0.30    0.2200      0.5729    0.3179
0.35    0.2448      0.5293    0.3347
0.40    0.2710      0.4882    0.3485
0.45    0.2991      0.4470    0.3584
0.50    0.3301      0.4049    0.3637
0.55    0.3645      0.3603    0.3624
0.60    0.4055      0.3133    0.3535
0.65    0.4538      0.2608    0.3312
0.70    0.5107      0.2049    0.2924
0.75    0.5729      0.1469    0.2339
0.80    0.6374      0.0904    0.1583
0.85    0.7055      0.0437    0.0823
0.90    0.7649      0.0126    0.0249

BEST
threshold=0.50 precision=0.3301 recall=0.4049 F1=0.3637