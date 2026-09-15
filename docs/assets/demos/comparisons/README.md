# Comparison input media

The [Chinese Blog](../../../index.html#qualitative-showcase) and [English Blog](../../../en.html#qualitative-showcase) retain the selected questions, reference answers and saved model responses.

## Eight-view transformation

Case 05 uses ViewSpatial sample 219. All eight original scene0131_02 input images are included in the original order as image-01.jpg through image-08.jpg. Images can be enlarged in the Blog. The image files are copied byte-for-byte from the source assets.

## Original video

Case 15 uses VSI-Bench sample 1085 and the original arkitscenes/47429977.mp4 from the [official VSI-Bench archive](https://huggingface.co/datasets/nyu-visionx/VSI-Bench/blob/main/arkitscenes.zip), retrieved on 2026-09-14.

The MP4 is unchanged: 480 × 640, 30 fps, 43.63 seconds, SHA-256 2776ad9d9b838db2539031323ecce83674979e7322acd786a0ba779af79725ce. Playback defaults to 1×. A WebM transcode provides browser compatibility. The archive member, CRC and file hashes are recorded in [the input media manifest](../../../web/input-media-manifest.json).

The [32 cached evaluation frames](case-15/frames-32/) remain available as reference inputs. ZDTaichu5.0-9B used 32 frames in the saved evaluation; Qwen3.5-9B and STEP3-VL-10B used 8 frames each. The new video player does not change those saved evaluation settings or model responses.
