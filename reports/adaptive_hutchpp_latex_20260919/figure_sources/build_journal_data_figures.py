"""Label-only revision of validated plots; no new experiments or resampling."""
import argparse
import build_validated_urop_figures as original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    original_header = original.header

    def descriptive_header(fig, title, subtitle):
        subtitle = subtitle.replace('Phase 1A • ', '').replace('Phase 1B • ', '')
        original_header(fig, title, subtitle)

    original.header = descriptive_header
    original.build(args.output_dir)


if __name__ == '__main__':
    main()
