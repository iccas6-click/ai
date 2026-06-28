# Real Smartphone Pill Evaluation Set

This folder is for the small real-world evaluation set used to decide whether the
pill recognition pipeline is service-ready. Do not commit captured photos or
annotation JSON files unless the team explicitly agrees to share them.

## Folder Layout

```text
datasets/evaluation/real-smartphone/
├── images/          # put smartphone photos here, ignored by git
└── annotations/     # reviewed JSON labels, ignored by git
```

## Minimum Capture Target

- 30-50 photos.
- 150-300 total pill instances.
- 1-10 pills per photo.
- Include easy and hard cases: white desk, dark desk, paper, hand, indoor light,
  mild shadows, mild rotation, and different distances.
- Keep pills separated. Overlapping pills should be tagged in notes or excluded
  from the first service-readiness score.

## Annotation Workflow

From `inference/`:

```bash
python -m pill_recognition.draft_real_annotations \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --top-k 5

python -m pill_recognition.render_real_annotation_review \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --output-dir outputs/real-review

python -m pill_recognition.validate_real_dataset \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --output outputs/evaluation/real-smartphone-validation.json
```

Then run the comparison suite:

```bash
python -m pill_recognition.run_real_evaluation_suite \
  --dataset-root ../datasets/evaluation/real-smartphone \
  --output-dir outputs/evaluation/real-smartphone-suite \
  --top-k 5
```

## Required Review Fields

Each annotation JSON must follow
`inference/pill_recognition/real_eval_schema.example.json`.

- `image`: image filename under `images/`.
- `allowed_pill_ids`: user's medication scope for that photo, if known.
- `pills[].class_name`: verified AIHub K-ID.
- `pills[].product_name`: verified product name.
- `pills[].bbox_xyxy`: original image pixel bbox `[x1, y1, x2, y2]`.
- `pills[].needs_review`: remove or set false only after human review.

Service-readiness should be judged from reviewed annotations only.
