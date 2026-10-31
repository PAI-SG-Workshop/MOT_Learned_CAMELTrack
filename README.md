# MOT_Learned_CAMELTrack

**CAMELTrack** - learned association (CAMEL) on top of tracking by detection, trained on KITTI-MOTS.

- Paper: Somers et al., arXiv 2505.01257; official code: `TrackingLaboratory/CAMELTrack`
- Implementation: `common/mot/camel.py` (temporal encoders per cue, GAFFE, InfoNCE, official augmentations and online
  logic, official sizes - 43 M parameters); training `tools/mot/train_camel.py` on the train detections of x,
  cascade_rcnn and v6 with the shared ReID feature; held-out train sequences 0001 / 0019 for the model selection.
- `[Unified]` differences are listed in the module docstring (cues = box + shared ReID; ReID x sqrt(512) and box
  standardisation, without which the embeddings collapse; 10 fps).

### Two attempts, both reported

1. **v1.1-a** (official sample construction, last epoch): HOTA far below the heuristic trackers. Diagnosis on train
   sequences: AssA 0.72 on sequences used for training, 0.40 on held-out ones, 0.37 on val - CAMEL memorised its
   training sequences (loss 0.04); the held-out pair accuracy (0.944) did not reveal it.
2. **v1.1-b**: samples hold what the online tracker sees (unmatched tracklets, new objects, false positives) and the
   epoch is chosen by the **online HOTA on the held-out train sequences** (epoch 15, held-out HOTA
   0.641, threshold 0.510).

### Results (KITTI-MOTS val)

| detector | **CAMEL v1.1-b** HOTA / AssA / IDSW | attempt v1.1-a (memorised) | TrackTrack | best v1.0 tracker |
|---|---|---|---|---|
| v1 | **0.322 / 0.313 / 457** | 0.284 / 0.240 / 936 | 0.374 | tracktrack 0.374 |
| v6 | **0.447 / 0.426 / 413** | 0.387 / 0.320 / 947 | 0.500 | botsort_reid 0.509 |
| v8 | **0.400 / 0.397 / 349** | 0.352 / 0.304 / 779 | 0.426 | botsort_reid 0.443 |
| v9 | **0.429 / 0.447 / 282** | 0.363 / 0.314 / 741 | 0.452 | botsort_reid 0.460 |
| v10 | **0.428 / 0.426 / 332** | 0.354 / 0.292 / 797 | 0.450 | botsort_reid 0.455 |
| v11 | **0.402 / 0.419 / 291** | 0.345 / 0.303 / 778 | 0.430 | hybridsort 0.435 |
| v12 | **0.415 / 0.445 / 281** | 0.337 / 0.290 / 819 | 0.397 | hybridsort 0.424 |
| v13 | **0.356 / 0.379 / 295** | 0.316 / 0.292 / 613 | 0.372 | tracktrack 0.372 |
| v26 | **0.367 / 0.380 / 293** | 0.317 / 0.279 / 681 | 0.315 | hybridsort 0.390 |
| x | **0.417 / 0.412 / 381** | 0.348 / 0.288 / 986 | 0.468 | botsort_reid 0.478 |
| x2 | **0.424 / 0.429 / 301** | 0.350 / 0.286 / 970 | 0.425 | hybridsort 0.452 |
| ppyoloe | **0.441 / 0.429 / 462** | 0.375 / 0.307 / 1042 | 0.487 | tracktrack 0.487 |
| rtmdet | **0.447 / 0.457 / 288** | 0.383 / 0.325 / 848 | 0.440 | hybridsort 0.465 |
| centernet | **0.370 / 0.386 / 262** | 0.313 / 0.264 / 785 | 0.331 | hybridsort 0.395 |
| retinanet | **0.460 / 0.436 / 468** | 0.400 / 0.328 / 1107 | 0.530 | botsort_reid 0.535 |
| fcos | **0.468 / 0.469 / 322** | 0.391 / 0.321 / 879 | 0.487 | botsort_reid 0.493 |
| atss | **0.482 / 0.447 / 431** | 0.414 / 0.332 / 1178 | 0.527 | tracktrack 0.527 |
| gfl | **0.494 / 0.481 / 357** | 0.413 / 0.330 / 981 | 0.516 | hybridsort 0.522 |
| tood | **0.467 / 0.444 / 390** | 0.392 / 0.311 / 967 | 0.499 | hybridsort 0.506 |
| ssd | **0.263 / 0.296 / 186** | 0.233 / 0.226 / 533 | 0.275 | botsort 0.288 |
| faster_rcnn | **0.489 / 0.481 / 581** | 0.412 / 0.348 / 1299 | 0.582 | tracktrack 0.582 |
| cascade_rcnn | **0.532 / 0.497 / 489** | 0.459 / 0.373 / 1145 | 0.625 | tracktrack 0.625 |
| libra_rcnn | **0.484 / 0.472 / 697** | 0.421 / 0.360 / 1267 | 0.599 | tracktrack 0.599 |
| dynamic_rcnn | **0.472 / 0.442 / 600** | 0.401 / 0.321 / 1339 | 0.579 | tracktrack 0.579 |
| deformable_detr | **0.406 / 0.391 / 385** | 0.358 / 0.300 / 955 | 0.480 | botsort_reid 0.482 |

Finding: with 10 training sequences the learned association does not reach the heuristic trackers (gap 0.05-0.09
HOTA on strong detectors); the learning curve was flat after the first epoch, so the limit is the amount of data.

## Usage

```bash
pip install -r requirements.txt
pip install "git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a"
# train detections: tools/mot/dump_dets.py --split train + dump_reid.py (MOT_Detectors_KITTIMOTS)
python tools/mot/train_camel.py --gate --dets dets_train/x_reid.npz
python tools/mot/train_camel.py --epochs 20 --dets dets_train/{x,cascade_rcnn,v6}_reid.npz
CAMEL_WEIGHTS=camel_kittimots.pth python tools/mot/run_tracker.py --source kittimots --dets dets/x_reid.npz --trackers camel
```

Weights (`camel_kittimots.pth`, v1.1-b; `camel_kittimots_attempt_a.pth`): Release v1.1.

Part of the *Physical AI MOT* comparison of the **Statera Working Group** (v1.1 extension: UCMCTrack, CAMELTrack,
Seg2Track-SAM2, AB3DMOT). The v1.0 comparison (10 trackers x 25 detectors, JDT, E2E) and the v1.1 summary are in
`MOT_Detectors_KITTIMOTS` (`RESULTS.md`).

## Protocol (all MOT_* repositories)

- Online / causal only: every method sees the current frame and its own state; checked by re-running prefixes of
  every sequence (the output of frame t must not change when later frames are removed).
- Input 1216 x 384 (KITTI 1242 x 375 letterboxed), detectors trained from scratch on KITTI-MOTS train (12 sequences),
  30 epochs, Adam 1e-4, effective batch 8, seed 777 - the protocol of the *Physical AI Object Detection* series.
- Metrics: HOTA / DetA / AssA (TrackEval classes, car / pedestrian mean), MOTA, IDF1, IDSW.
- Latency: RTX 4070 Laptop GPU, batch 1, fp32, medians over val; pipeline = detector (forward + post-processing) +
  ReID (appearance trackers) + tracker (CPU); image decoding / letterboxing not included.
- Trackers run with their official default parameters; frame-based values given for 30 fps are scaled to 10 fps.
  A sensitivity table with one common score threshold for every tracker is reported separately (chosen on val, so it
  is analysis, not ranking).

## Data: KITTI-MOTS (box level, val = 9 sequences, 2,981 frames)

KITTI tracking images (21 training sequences) with instance / track annotations; boxes = tight boxes of the masks,
classes car and pedestrian, ignore regions (id 10000) neither rewarded nor penalised. The annotations are taken from
the **KITTI-STEP** panoptic maps (same sequences and split, car / person instances merged from KITTI-MOTS; frames,
track counts and pedestrian masks match the KITTI-MOTS paper, car masks differ by 39 of 26,899).

```bash
bash tools/mot/get_kitti_mots.sh            # KITTI tracking images + annotations (KITTI-STEP fallback)
python tools/mot/kittimots_to_voc.py        # VOC-layout copy with tracks + count report
UNIFIED_IMAGE_DATASET=kittimots python tools/mot/count_loader.py     # counts as the loaders read them
```

**Licence of the data and of every trained weight here: KITTI / KITTI-MOTS / KITTI-STEP are CC BY-NC-SA 3.0
(non-commercial).**

Results of this repository: `results/` (evaluation JSON per detector). Sources and licences: `THIRD_PARTY.md`.

## Global Collaboration

Hills Robotics is currently participating in multiple projects in Korea and internationally.

We are actively interested in building relationships with organizations that can complement our technology with global sales networks, distribution channels, system integration capabilities, manufacturing ecosystems, AI infrastructure, robotics technologies, research capabilities, or strategic investment.

Several technology and business collaborations are already underway. Some partners, projects, system architectures, and commercial programs cannot be publicly disclosed because they are subject to confidentiality and business agreements.

We believe that the emerging Physical AI ecosystem cannot be built by a single company.
It requires a global network connecting robotics companies, AI researchers, semiconductor and computing companies, sensor manufacturers, software developers, universities, system integrators, industrial customers, investors, and global distribution partners.

We have now begun our serious exploration of the Physical AI frontier, and we welcome connections with people and organizations who want to build this ecosystem together.

## Research, Contribution, and Donation

The repositories of the Statera Working Group are intended to contribute to the broader technical discussion around Physical AI.

Researchers, engineers, developers, companies, universities, and independent contributors interested in these subjects are welcome to connect with us.

If you would like to collaborate on Physical AI or autonomous robotics, participate in research or technology development, discuss international business cooperation, propose technology or platform integration, contribute technical knowledge, or support this GitHub knowledge initiative through sponsorship or donation, please contact us.

## Contact

**Richard Minsoo Go**
Head of AI, Hills Robotics
Statera Working Group

Personal / Research Contact:
[gominsoo888@gmail.com](mailto:gominsoo888@gmail.com)

Hills Robotics / Business Contact:
[minsoo.go@hillslogis.com](mailto:minsoo.go@hillslogis.com)

Hills Robotics:
https://hillslogis.com/

For inquiries regarding next-generation outdoor autonomous robotic platforms, Physical AI Spider Grid, Intelligence Cabinet, Edge AI, or On-Premise AI architectures that are not yet publicly presented on the company website, please contact us directly.
