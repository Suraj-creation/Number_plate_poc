# Pune traffic video benchmark

The test uses a 125-second Pune drive excerpt. The windshield crop keeps the original horizontal pixel scale at 3840×960. The input and output contain 625 frames at 5 fps and run at the original playback speed. The excerpt includes an edit from the source video about 30 seconds in. Both model runs processed all 625 frames. [Source and preparation commands](../input/videos/README_sources.md#pune-traffic-125-second-excerpt)

## Manual reference

The reference contains 40 manually transcribed plate observations across 33 frames and eight distinct registrations. Boxes were matched one to one at an intersection-over-union threshold of 0.50. The first model run used the original Bharat-series validator, and its vehicle-only overlay is the baseline. The second used the updated validator. Its saved observations were rendered through the final overlay logic so the evaluation, collected plate numbers, and delivered MP4 used the same label decisions.

| Measure | Baseline | Updated |
| --- | ---: | ---: |
| Matched plate boxes | 36 of 40 | 36 of 40 |
| Exact displayed numbers on labeled observations | 13 of 40 | 14 of 40 |
| Incorrect displayed numbers on labeled observations | 1 | 0 |
| Matched boxes whose text was withheld | 22 | 22 |
| Distinct registrations verified on the labeled frames | 4 of 8 | 5 of 8 |
| Reference registrations verified anywhere in the clip | 6 of 8 | 7 of 8 |

The last row uses a separate review of source crops at representative frames where each number appeared in the video. The 40 observations are a partial reference. A crop review outside those observations found a letter substitution displayed on two frames. The labeled-frame error count applies only to the manually matched reference.

## Changes measured

- The Bharat-series validator now accepts one- and two-letter suffixes. The one-letter registration in this test gained 18 correct overlay frames. [MoRTH Bharat-series registration rule](https://morth.gov.in/sites/default/files/notifications_document/GSR%20594%28E%29%2026%20.08.2021%20BH%20series%20registration%20mark%20Rules_0.pdf)
- Independent plate tracking adds 48 overlay observations across three of the reviewed registrations compared with the updated vehicle-only overlay. The overlay also withheld two conflicting frame reads when the same plate had established a different consensus. One of those conflicts occurred on a labeled frame.
- PaddleOCR-only recoveries need five time-separated observations for a plate-track overlay when the primary reader has not corroborated the result. The additional evidence requirement filtered two recurring misreads found when reviewing output crops.

The final plate-only output shows accepted registration text and the corresponding plate box. The standard diagnostic overlay and structured evidence remain available. The H.264 delivery MP4 was decoded to verify all 625 frames, the 3840×960 resolution, and the 125.000-second run time.
