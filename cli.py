import argparse
from core.cloner import KokoClone

def main():
    parser = argparse.ArgumentParser(description="KokoClone: Zero-Shot Multilingual Voice Cloning")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--lang", type=str, default="en", help="Language code (en, hi, fr, ja, zh, it, pt, es)")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference audio file (.wav)")
    parser.add_argument("--out", type=str, default="output.wav", help="Output file path (.wav)")

    args = parser.parse_args()

    cloner = KokoClone()
    cloner.generate(
        text=args.text,
        lang=args.lang,
        reference_audio=args.ref,
        output_path=args.out
    )

if __name__ == "__main__":
    main()