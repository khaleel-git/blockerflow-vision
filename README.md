# blockerflow-vision

The nudity detector used by **Blockerflow for Mac** ("Block adult images"), as a small program of its own.

Blockerflow starts `blockerflow-vision` in the background and talks to it over stdin and stdout: it sends
320 x 320 pictures cut from the screen and gets back one score per class of the NudeNet detector.
Blockerflow decides what to do with the scores; this program only runs the model. Everything stays on the
Mac; nothing is sent anywhere.

It contains:

- **NudeNet 320n** (`model/nudenet_320n.onnx`), release `v3.4-weights` of
  [notAI-tech/NudeNet](https://github.com/notAI-tech/NudeNet), GNU AGPL-3.0. It is a YOLOv8-nano detector
  trained with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), AGPL-3.0.
  SHA-256 `c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f`.
- The code to run it with [ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT) through
  [onnxruntime_go](https://github.com/yalue/onnxruntime_go) (MIT), on CoreML when it can.

This program is licensed under the **GNU Affero General Public License v3.0** (see `LICENSE`).

## Build

Needs Go 1.25 and Xcode's command line tools.

```bash
./build.sh            # build/blockerflow-vision, universal (Apple silicon + Intel)
go test ./...         # set BLOCKERFLOW_ORT_LIB=/path/to/libonnxruntime.dylib to include the model tests
```

At run time the ONNX Runtime 1.23.2 library is taken from `BLOCKERFLOW_ORT_LIB`, or else from
`../Frameworks/libonnxruntime.dylib` next to the executable (its place inside `Blockerflow.app`, where this
program sits in `Contents/Helpers`).

## Protocol (version 1)

Little-endian, fixed sizes, over stdin and stdout.

| When | Direction | Bytes |
|---|---|---|
| once, at start | program to client | `"BFV1"`, uint32 version (1), uint32 side (320), uint32 class count (18), uint32 n + n bytes of class names joined by `\n`, uint8 accelerator (0 CPU, 1 CoreML) |
| each picture | client to program | 320 x 320 x 3 bytes, RGB, row-major |
| each reply | program to client | uint8 status; 0: 18 float32 scores in class order (the best score each class reached anywhere in the picture); 1: uint32 n + n bytes of error message |

The program exits when its input closes. Diagnostics go to stderr.
