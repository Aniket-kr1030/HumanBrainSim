import argparse
import yaml
import numpy as np
from brain_model.brain import HierarchicalBrainModel
from brain_model.speech import SpeechRecognitionModule, TextToSpeechModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--spontaneous', action='store_true', help='run spontaneous cycles')
    parser.add_argument('--capacity', type=float, default=1.0, help='scale neural capacity')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'ne'],
                        help='compute device: cpu, mps (Metal), or ne (Neural Engine)')
    parser.add_argument('--speech', action='store_true', help='use speech input and output')
    parser.add_argument('--camera', action='store_true', help='show live camera feed')
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

    cam_thread = None
    if args.camera:
        from brain_model.hardware_interface import camera_capture
        import threading
        cam_thread = threading.Thread(target=camera_capture, daemon=True)
        cam_thread.start()

    stt = None
    tts = None
    record_audio = None
    if args.speech:
        from brain_model.hardware_interface import record_audio as _rec
        stt = SpeechRecognitionModule()
        tts = TextToSpeechModule()
        record_audio = _rec

    step = 0
    try:
        while True if args.steps < 0 else step < args.steps:
            if args.speech:
                audio = record_audio(duration=2.0)
                text = stt.transcribe(audio)
                x = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)[:cfg['input_dim']]
                if x.size < cfg['input_dim']:
                    x = np.pad(x, (0, cfg['input_dim'] - x.size))
                x = x.astype(float) / 255.0
                next_x = np.random.randn(cfg['input_dim'])
                out = model.step(x, reward=0.0, next_x=next_x)
                tts.speak(text)
            elif args.spontaneous:
                out = model.step_spontaneous()
            else:
                x = np.random.randn(cfg['input_dim'])
                next_x = np.random.randn(cfg['input_dim'])
                out = model.step(x, reward=0.0, next_x=next_x)
            print('step', step, 'td', out['td_error'], 'penalty', out['penalty'])
            step += 1
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
