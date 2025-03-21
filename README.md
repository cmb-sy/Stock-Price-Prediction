## Overview

This application predicts future stock prices based on historical stock data. It aims to assist users in making informed investment decisions.

## How to Use

1. Create and activate the `conda` environment:
   ```bash
   conda create --name env
   conda activate env
   ```
2. Install the required dependencies:
   ```bash
   conda install --yes --file requirements.txt
   ```
3. Run the application:
   ```bash
   python run.py
   ```

## Challenges

The current prediction accuracy is suboptimal, leading to uncertain results.

## Future Improvements

To overcome these challenges, the following enhancements are planned:

Incorporate additional parameters such as revenue, employee count, PBR, PER, and equity ratio to improve prediction accuracy.
Extract relevant company information from the latest news and leverage large language models (LLMs) to enhance prediction capabilities.

## Demo
