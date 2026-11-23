# Local video acceptance samples

These clips were downloaded for local pipeline QA on 2026-11-19. Pexels allows free use and modification under its [Pexels License](https://www.pexels.com/license/). They contain no ground-truth plate boxes or verified plate text, so they are suitable for integration and throughput checks, not accuracy scoring. The separate original downloads were removed after processing; only the normalized, concatenated test inputs and annotated outputs remain locally.

## Day and general traffic compilation

`india_traffic_acceptance_2m14s_3fps.mp4` combines five distinct clips. Each was resized to fit 1920×1080, converted to 3 fps, had audio removed, then concatenated. The resulting input is 134.002 seconds and 402 frames. The pipeline was run at stride 1, so all 402 frames of this prepared input were inferred and annotated.

| Pexels source | Original duration | Original video |
| --- | ---: | --- |
| [Traffic and pedestrians by CST, Mumbai](https://www.pexels.com/video/traffic-in-the-city-1193769/) | 21.48 s | 1920×1080, 25 fps |
| [Busy Mumbai street with heavy traffic](https://www.pexels.com/video/busy-mumbai-street-with-heavy-traffic-30230719/) | 61.97 s | 2560×1440, 30 fps |
| [Vibrant busy street in Mumbai](https://www.pexels.com/video/vibrant-busy-street-scene-in-mumbai-india-30279179/) | 12.20 s | 2560×1440, 30 fps |
| [Busy urban traffic in Mumbai](https://www.pexels.com/video/busy-urban-traffic-scene-in-mumbai-city-30400506/) | 30.93 s | 2560×1440, 30 fps |
| [Busy Indian city street traffic](https://www.pexels.com/video/busy-urban-traffic-on-indian-city-street-30608961/) | 7.47 s | 2560×1440, 30 fps |

## Night traffic compilation

`india_night_traffic_validation_53s_3fps.mp4` combines the two clips below, normalized to 1920×1080 at 3 fps with audio removed. It is 53.001 seconds and 159 frames; all frames were inferred at stride 1.

| Pexels source | Original duration | Original video |
| --- | ---: | --- |
| [Night street with cars and motorcycles](https://www.pexels.com/video/a-night-view-of-a-busy-street-with-cars-and-motorcycles-17812439/) | 34.77 s | 2560×1440, 29.97 fps |
| [Night market traffic in India](https://www.pexels.com/video/busy-urban-night-market-with-traffic-34793603/) | 18.30 s | 1920×1080, 60 fps |

The 3 fps versions deliberately reduce the source temporal sampling for a practical multi-minute GPU integration run. Results do not establish performance at the original 25–60 fps, nor Indian plate accuracy.

## Readable Mumbai street excerpt

`mumbai_readable_plates.mp4` uses seconds 4–14 of a daylight street clip by aksinfo7 universe. The excerpt keeps the original 3840×2160 resolution and 30 fps. Audio is omitted. [Busy Urban Traffic on Mumbai Street](https://www.pexels.com/video/busy-urban-traffic-on-mumbai-street-30169858/) · [Pexels License](https://www.pexels.com/license/)

```bash
curl -fL https://www.pexels.com/download/video/30169858/ -o /tmp/mumbai_source.mp4
ffmpeg -threads 4 -ss 4 -i /tmp/mumbai_source.mp4 -t 10 -map 0:v:0 \
  -c:v libx264 -crf 15 -preset ultrafast -threads 4 -pix_fmt yuv420p -movflags +faststart \
  input/videos/mumbai_readable_plates.mp4
ANPR_OFFLINE=1 python -m src.cli video --input input/videos/mumbai_readable_plates.mp4 \
  --config configs/readable_video.yaml --output output/readable_plates --stride 1
```

The configuration uses the optional PaddleOCR environment. Set `ANPR_PADDLE_PYTHON` to that environment's Python executable. The download used for the test had SHA-256 `e688aa982373e63c2dc88c1fe2ced4c3a1b9e4851a22075cf09c3e4adc671c0e`.

## Pune traffic 125-second excerpt

The test used seconds 15–140 from one daylight Pune drive. The source includes a transition around 45 seconds. The prepared clip uses the upper 3840×960 of the original 3840×2160 frame to focus on the windshield, samples at 5 fps, and covers the same 125 seconds at normal playback speed. [Pune footage by Drive Wid Ash](https://www.youtube.com/watch?v=XYDrIAZoMGQ)

Given the first 142 seconds of the 2160p VP9 video stream as `pune-source.webm`, prepare and run the input with the same flags used for validation.

```bash
ffmpeg -threads 10 -ss 15 -i pune-source.webm -t 125 -map 0:v:0 \
  -filter_threads 4 -vf 'crop=3840:960:0:0,fps=5' -c:v libx264 -threads 5 \
  -preset veryfast -crf 17 -pix_fmt yuv420p -movflags +faststart input/videos/pune_traffic_125s.mp4
ANPR_OFFLINE=1 python -m src.cli video --input input/videos/pune_traffic_125s.mp4 \
  --config configs/readable_video.yaml --output output/pune_traffic --stride 1
```

The prepared input contained 625 frames and had SHA-256 `c826486f75619320c83e3b684cb6ff49485b1964a7b9aa4a02dedb7157f5a728`. Use the optional PaddleOCR environment described above. [Validation results](../../eval/pune_video_results.md)

## Mixed roadside traffic 122-second edit

The demo joins five outdoor roadside shots in the order below. Each excerpt starts at the beginning of its source. The 3840×1440 crop keeps the original plate pixels in the lower portion of the 4K frames. The 5 fps input preserves playback speed. [Pexels license](https://www.pexels.com/license/)

| Start in edit | Duration | Source |
| --- | ---: | --- |
| 0:00 | 20 s | [Busy Indian Street with Traffic and Vehicles by aksinfo7 universe](https://www.pexels.com/video/busy-indian-street-with-traffic-and-vehicles-30381636/) |
| 0:20 | 30 s | [Busy Urban Traffic Scene in Mumbai City by aksinfo7 universe](https://www.pexels.com/video/busy-urban-traffic-scene-in-mumbai-city-30400506/) |
| 0:50 | 27 s | [Bustling Urban Street Scene in Mumbai Daytime by aksinfo7 universe](https://www.pexels.com/video/bustling-urban-street-scene-in-mumbai-daytime-30262835/) |
| 1:17 | 29 s | [Bustling City Street Traffic in Mumbai by aksinfo7 universe](https://www.pexels.com/video/bustling-city-street-traffic-in-mumbai-30441035/) |
| 1:46 | 16 s | [Busy Street India by Tarun Suthar](https://www.pexels.com/video/busy-street-india-indian-streets-street-27000602/) |

Prepare the input in a local working directory and run from the repository with the optional PaddleOCR environment configured.

```bash
work=$(mktemp -d)
ids=(30381636 30400506 30262835 30441035 27000602)
lengths=(20 30 27 29 16)
for i in "${!ids[@]}"; do
  id=${ids[$i]}
  curl -fL "https://www.pexels.com/download/video/$id/" -o "$work/$id.mp4"
  ffmpeg -v error -threads 4 -i "$work/$id.mp4" -t "${lengths[$i]}" -an \
    -filter_threads 2 -vf 'crop=3840:1440:0:720,fps=5' \
    -c:v libx264 -threads 3 -preset veryfast -crf 16 -pix_fmt yuv420p "$work/mix-$id.mp4"
  printf "file '%s/mix-%s.mp4'\n" "$work" "$id" >> "$work/segments.txt"
done
ffmpeg -v error -f concat -safe 0 -i "$work/segments.txt" -c copy \
  -movflags +faststart "$work/mixed_roadside_122s.mp4"
ANPR_OFFLINE=1 python -m src.cli video --input "$work/mixed_roadside_122s.mp4" \
  --config configs/readable_video.yaml --output output/mixed_roadside --stride 1
```

The benchmark input had 610 frames and SHA-256 `e2bfc09b3c2263d520d56b202e8f42ba96604f135b2b2c84cfef8ba983145d96`. [Per-vehicle results](../../eval/mixed_roadside_results.md)
