# Readable Mumbai video validation

The test used one 10-second excerpt of real Mumbai traffic at its original 3840×2160 and 30 fps. The suggested Kaggle download returned HTTP 404. The replacement source and exact preparation steps are in the [video source notes](../input/videos/README_sources.md#readable-mumbai-street-excerpt).

## Results

| Check | Initial pass | Final plate-only export |
| --- | ---: | ---: |
| Frames analyzed | 300 | 300 |
| Plate boxes matched on manually labeled observations | 37 of 37 | 37 of 37 |
| Exactly correct numbers displayed on those observations | 3 of 37 | 19 of 37 |
| Distinct reference plates correctly displayed at least once in the full clip | 7 of 19 | 14 of 19 |

The reference contains 37 manually transcribed plate observations across 18 source frames and 19 distinct vehicles. A box match required intersection over union of at least 0.50. For the full-clip count, the displayed strings and their locations were compared with representative original crops. Five reference plates had unresolved text. Some correct labels lasted one or two frames. The reference covers the checked readable plates in this excerpt.

## Changes verified

- The standard CCT reader dropped characters on some two-line plates and confused some letters with digits. The opt-in profile uses PaddleOCR and checks line scores before accepting a recovery candidate. A number requires agreement with the primary reader or consistent readings across a vehicle track.
- PaddleOCR sometimes treated the small country mark beside a registration as another line of the number. The recovery step uses the dominant registration line where the extra text is much smaller. A separate 22-frame CLI run showed the corrected number for both cars in every frame.
- The final H.264 export used the full run's detection and OCR evidence, with the updated recognition, track voting and overlays replayed over all 300 source frames. Full decoding verified 300 output frames at 30 fps. The JSON and CSV each had 9,207 rows, and the result archive passed CRC validation.
- All 34 tests passed with GPU integration enabled.

The output MP4 SHA-256 is `677bee36423d98763e0963294cedc28ae09bb940b502acbc920c1ea350fd4c22`.
