# Mixed roadside traffic video validation

This 2:02 edit combines five daytime views filmed from the roadside. Each frame is cropped at the original pixel scale to 3840×1440. The pipeline analyzed all 610 frames at 5 fps. The long run ended before saving its final export, so the demo uses its completed observations through the cut at 1:46 and a fresh run of the final shot. An elevated-camera candidate was dropped after the source crops showed motion blur. [Footage, credits, and reproduction commands](../input/videos/README_sources.md#mixed-roadside-traffic-122-second-edit)

The source reference contains 48 transcribed plate observations across 37 frames and 38 distinct registrations. Detection boxes were matched one to one at 0.50 intersection over union. The automatic overlay matched 37 of 48 boxes, displayed 13 exact reads, and withheld text on the other 24 matched boxes. The automatic overlay made zero wrong reads at these labeled positions. The review of the full run found five other sequences with wrong characters.

I checked representative source crops for each displayed sequence and removed the five misread sequences from the GitHub MP4. I also included five source-verified OCR candidates that the automatic overlay had withheld. These supply the motorcycle, one more scooter, and three more frames of the first bus. The reviewed MP4 shows the OCR numbers and their plate boxes.

| Vehicle type | Distinct plates in source reference | Reference plates read automatically somewhere in the edit | Distinct verified plates in published video | Example |
| --- | ---: | ---: | ---: | --- |
| Car and taxi | 20 | 18 | 31 | 0:02.6 |
| Scooter | 5 | 2 | 3 | 0:07.4 |
| Motorcycle | 1 | 0 | 1 | 0:29.6 |
| Auto-rickshaw | 7 | 1 | 1 | 0:25.6 |
| Goods truck | 3 | 3 | 6 | 0:15.6 |
| Bus | 1 | 1 | 2 | 0:54.2 and 1:41.6 |
| Ambulance | 1 | 1 | 1 | 0:48.4 |
| **Total** | **38** | **26** | **45** | |

Vehicle types were assigned from the source footage during review. The published count also includes verified reads outside the reference frames. The H.264 MP4 decodes to 610 frames over 122 seconds. The repository test suite passed all 37 tests, including the GPU integration test.
