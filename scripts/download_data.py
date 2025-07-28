
"""
This script provides instructions for downloading the necessary datasets.
"""

import os

def main():
    """Prints instructions for downloading the datasets."""

    print("=" * 80)
    print("Instructions for Downloading Datasets")
    print("=" * 80)
    print("\nThis project uses several datasets that need to be downloaded manually due to licensing and distribution restrictions.")
    print("\n1. VLSP 2013 - Word Segmentation")
    print("----------------------------------")
    print("   - Description: A dataset for Vietnamese word segmentation.")
    print("   - How to get: You need to register and download the data from the official VLSP website.")
    print("   - Website: https://vlsp.org.vn/resources/corpora (or search for 'VLSP 2013 word segmentation dataset')")
    print("   - After downloading, place the 'train.txt' and 'test.txt' files in the following directory:")
    print(f"     {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'vlsp2013'))}")

    print("\n2. VLSP 2016 - Sentiment Analysis")
    print("----------------------------------")
    print("   - Description: A dataset for Vietnamese sentiment analysis.")
    print("   - How to get: You need to register and download the data from the official VLSP website.")
    print("   - Website: https://vlsp.org.vn/resources/corpora (or search for 'VLSP 2016 sentiment analysis dataset')")
    print("   - After downloading, place the data in the following directory:")
    print(f"     {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'vlsp2016'))}")

    print("\n3. UIT-ViSFD - Vietnamese Students' Feedback Dataset")
    print("-----------------------------------------------------")
    print("   - Description: A dataset of student feedback for sentiment analysis.")
    print("   - How to get: This dataset is available on Hugging Face and will be downloaded automatically by the data loader.")
    print("   - Hugging Face: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback")

    print("\n" + "=" * 80)
    print("Once you have downloaded the VLSP datasets, the project should run correctly.")
    print("=" * 80)

if __name__ == "__main__":
    main()
