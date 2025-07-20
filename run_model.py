import argparse
import yaml
import numpy as np
from brain_model.brain import HierarchicalBrainModel
from brain_model.speech import SpeechRecognitionModule, TextToSpeechModule
from brain_model.brain_responder import BrainResponder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--spontaneous', action='store_true', help='run spontaneous cycles')
    parser.add_argument('--capacity', type=float, default=1.0, help='scale neural capacity')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'ne'],
                        help='compute device: cpu, mps (Metal), or ne (Neural Engine)')
    parser.add_argument('--speech', action='store_true', help='use speech input and output')
    parser.add_argument('--camera', action='store_true', help='show live camera feed')
    parser.add_argument('--text', action='store_true', help='interactive text input')
    parser.add_argument('--steps', type=int, default=-1,
                        help='number of steps to run (-1 for infinite)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = HierarchicalBrainModel(
        input_dim=cfg['input_dim'],
        cortex_layers=cfg['cortex_layers'],
        hippocampus_dim=cfg['hippocampus_dim'],
        ach_thresh=cfg.get('ach_thresh', 0.5),
        device=args.device
    )
    if args.capacity != 1.0:
        model.adjust_capacity(args.capacity)
    model.dashboard.launch()
    print("Press Ctrl+C to stop the program")

    cap = None
    read_frame = None
    show_frame = None
    close_camera = None
    if args.camera:
        from brain_model.hardware_interface import (
            open_camera,
            read_frame as _read,
            show_frame as _show,
            close_camera as _close,
        )
        cap = open_camera()
        read_frame = _read
        show_frame = _show
        close_camera = _close

    stt = None
    tts = None
    record_audio = None
    responder = BrainResponder()
    if args.speech:
        from brain_model.hardware_interface import record_audio as _rec
        stt = SpeechRecognitionModule()
        tts = TextToSpeechModule()
        record_audio = _rec

    step = 0
    try:
        while True if args.steps < 0 else step < args.steps:
            frame = None
            amplitude = None
            if args.text:
                try:
                    user_text = input("You: ")
                except EOFError:
                    break
                if user_text.lower() in {"quit", "exit"}:
                    break
                x = np.frombuffer(user_text.encode("utf-8"), dtype=np.uint8)[: cfg["input_dim"]]
                if x.size < cfg["input_dim"]:
                    x = np.pad(x, (0, cfg["input_dim"] - x.size))
                x = x.astype(float) / 255.0
                next_x = np.random.randn(cfg["input_dim"])
                out = model.step(x, reward=0.0, next_x=next_x)
                responder.add(out["activations"], user_text)
                response = responder.respond(out["recall"])
                print("Model:", response)
                if args.speech:
                    tts.speak(response)
            elif args.speech:
                audio, amplitude = record_audio(duration=2.0)
                text = stt.transcribe(audio)
                x = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)[: cfg["input_dim"]]
                if x.size < cfg["input_dim"]:
                    x = np.pad(x, (0, cfg["input_dim"] - x.size))
                x = x.astype(float) / 255.0
                next_x = np.random.randn(cfg["input_dim"])
                out = model.step(x, reward=0.0, next_x=next_x)
                if text:
                    responder.add(out["activations"], text)
                    response = responder.respond(out["recall"])
                    tts.speak(response)
            elif args.spontaneous:
                out = model.step_spontaneous()
            else:
                x = np.random.randn(cfg["input_dim"])
                next_x = np.random.randn(cfg["input_dim"])
                out = model.step(x, reward=0.0, next_x=next_x)

            if args.camera:
                frame = read_frame(cap)
                show_text = None
                if amplitude is not None:
                    show_text = f"amp:{amplitude:.2f} td:{out['td_error']:.2f}"
                else:
                    show_text = f"td:{out['td_error']:.2f}"
                show_frame(frame, show_text)

            print("step", step, "td", out["td_error"], "penalty", out["penalty"])
            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        if cap:
            close_camera(cap)

if __name__ == '__main__':
    main()
